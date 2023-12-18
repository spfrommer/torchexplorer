from __future__ import annotations

from torch.nn import Module

import re

from typing import Optional, Union
from dataclasses import dataclass

import sys
from loguru import logger

from torchexplorer import utils


config = {'handlers': [{ 'sink': sys.stderr, 'format': '{message}', 'level': 'DEBUG' }]}
logger.configure(**config)  # type: ignore
logger.disable("torchexplorer")


from torchexplorer.core import GradFn, InvocationId, ModuleInvocationStructure

StructureNode = Union[ModuleInvocationStructure, str]


@dataclass
class UpstreamStructureNode:
    node: StructureNode
    output_index: int

    def __str__(self):
        node_str = self.node if isinstance(self.node, str) else self.node.str_impl()
        return f'(Upstream {node_str}):output {self.output_index}'


@dataclass
class DownstreamStructureNode:
    node: StructureNode
    input_index: int
    gradfn: GradFn

    def __str__(self):
        node_str = self.node if isinstance(self.node, str) else self.node.str_impl()
        return f'(Downstream {node_str}):input {self.input_index}'


def extract_structure(
        module: Module, invocation_id: InvocationId=0
    ) -> ModuleInvocationStructure:
    """Module must have had hooks already added and one forward pass."""

    extractor = StructureExtractor(module, invocation_id)
    return extractor.extract_structure()


class StructureExtractor:
    def __init__(self, module: Module, invocation_id: InvocationId):
        self.module = module
        self.invocation_id = invocation_id
        self.structure_id = 0
        self.log_indent_level = 0

    def log(self, message: str, extract_level: str, class_name: str):
        ind = '| ' * self.log_indent_level
        logger.opt(colors=True).debug(
            f'{ind}<green>{extract_level}</green>:<blue>{class_name}</blue> {message}'
        )

    def extract_structure(self) -> ModuleInvocationStructure:
        return self._extract_structure(self.module, self.invocation_id)

    def _extract_structure(
            self, module: Module, invocation_id: InvocationId
        ) -> ModuleInvocationStructure:

        self.log_indent_level += 1

        log_args = ['OUTER', module.__class__.__name__]
        self.log('Start extracting structure', *log_args)

        input_n = len(module.torchexplorer_metadata.input_gradfns[invocation_id])
        output_n = len(module.torchexplorer_metadata.output_gradfns[invocation_id])
        structure = ModuleInvocationStructure(
            module, invocation_id, self.structure_id, input_n, output_n
        )
        self.structure_id += 1
        structure.upstreams_fetched = False


        downstreams = []
        for i, output_gradfn in utils.enum_not_none(_get_output_gradfns(structure)):
            node_name = f'Output {i}'
            downstreams.append(DownstreamStructureNode(
                node=node_name, input_index=i, gradfn=output_gradfn
            ))

        i = 0
        while i < len(downstreams):
            downstream = downstreams[i]
            i += 1

            self.log(f'Processing downstream {downstream}', *log_args)
            upstreams = self._inner_recurse(structure, downstream.gradfn)
            self.log(f'Done inner recurse, got {len(upstreams)} upstreams', *log_args)

            for upstream in upstreams:
                assert structure.inner_graph.has_node(upstream.node)
                assert structure.inner_graph.has_node(downstream.node)

                structure.inner_graph.add_edge(
                    upstream.node,
                    downstream.node,
                    upstream_output_index=upstream.output_index,
                    downstream_input_index=downstream.input_index,
                )

                if is_input_node(upstream.node):
                    continue

                assert isinstance(upstream.node, ModuleInvocationStructure)
                    
                if not upstream.node.upstreams_fetched:
                    self.log(f'Q {upstream.node.module.__class__.__name__}', *log_args)
                    input_gradfns = _get_input_gradfns(upstream.node)
                    for j, input_gradfn in utils.enum_not_none(input_gradfns):
                        self.log(f'Queueing upstream {input_gradfn}', *log_args)
                        downstreams.append(DownstreamStructureNode(
                            node=upstream.node, input_index=j, gradfn=input_gradfn
                        ))
                    
                    upstream.node.upstreams_fetched = True

        self.log('Done extracting structure', *log_args)
        self.log_indent_level -= 1
        return structure

    def _inner_recurse(
            self,
            current_struct: ModuleInvocationStructure,
            gradfn: GradFn,
            gradfn_index: Optional[int]=None
        ) -> list[UpstreamStructureNode]:
        # Some gradfns kind of "combine" all their inputs into one gradfn node in the
        # graph (e.g., BackwardHookFunctionBackward). To keep the parent dependencies
        # exact, we only recurse into the particular gradfn_index.
        current_module = current_struct.module

        self.log_indent_level += 1
        log_args = ['INNER', current_module.__class__.__name__]
        self.log(f'Recursing on {gradfn} with index {gradfn_index}', *log_args)

        if gradfn is None:
            return []

        next_functions = gradfn.next_functions
        if 'BackwardHookFunctionBackward' in str(gradfn) and gradfn_index is not None:
            next_functions = (next_functions[gradfn_index],)

        self.log(f'Next fns: {gradfn.next_functions}  ->  {next_functions}', *log_args)
        metadata = gradfn.metadata

        is_enter_forward = 'input_index' in metadata
        is_exit_forward = 'output_index' in metadata

        if is_enter_forward:
            assert metadata['module'] == current_module
            assert metadata['invocation_id'] == current_struct.invocation_id
            name = f'Input {metadata["input_index"]}'

            self.log(f'Enter forward, in index {metadata["input_index"]}', *log_args)
            upstreams = [UpstreamStructureNode(name, metadata['input_index'])]
        elif is_exit_forward:
            upstream_module = metadata['module']
            invocation_id = metadata['invocation_id']

            self.log(f'Exit forward, out index {metadata["output_index"]}', *log_args)
            self.log(f'Upstream mod: {upstream_module.__class__.__name__}', *log_args)

            upstream_struct = current_struct.get_inner_structure(
                upstream_module, invocation_id
            )

            if upstream_struct is None:
                self.log(f'Creating new upstream structure', *log_args)
                upstream_struct = self._extract_structure(
                    upstream_module, invocation_id
                )
                current_struct.inner_graph.add_node(
                    upstream_struct, structure_id=upstream_struct.structure_id
                )

            upstreams = [
                UpstreamStructureNode(upstream_struct, metadata['output_index'])
            ]
        else:
            if 'upstreams_cache' in gradfn.metadata:
                if gradfn_index in gradfn.metadata['upstreams_cache']:
                    return gradfn.metadata['upstreams_cache'][gradfn_index]

            upstreams = _flatten([
                self._inner_recurse(current_struct, n[0], n[1]) for n in next_functions
            ])


            if 'upstreams_cache' not in gradfn.metadata:
                gradfn.metadata['upstreams_cache'] = {}
            gradfn.metadata['upstreams_cache'][gradfn_index] = upstreams
            self.log(f'Returning {len(upstreams)} intermediate upstreams', *log_args)

        self.log_indent_level -= 1
        return upstreams


def _get_input_gradfns(
        structure: ModuleInvocationStructure
    ) -> tuple[Optional[GradFn], ...]:

    return structure.module_metadata().input_gradfns[structure.invocation_id]

def _get_output_gradfns(
        structure: ModuleInvocationStructure
    ) -> tuple[Optional[GradFn], ...]:

    return structure.module_metadata().output_gradfns[structure.invocation_id]


def is_input_node(node) -> bool:
    if not isinstance(node, str):
        return False
    return bool(re.match(r'Input \d+', node)) or (node=='Input')


def is_output_node(node) -> bool:
    if not isinstance(node, str):
        return False
    return bool(re.match(r'Output \d+', node)) or (node=='Output')


def is_io_node(node) -> bool:
    return is_input_node(node) or is_output_node(node)


# Adapted from: https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
def _flatten(items, seqtypes=(list, tuple)):
    for i, _ in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items