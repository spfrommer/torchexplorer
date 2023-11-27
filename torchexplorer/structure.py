from torch import nn

import re

from typing import Optional, Union
from dataclasses import dataclass

import sys
from loguru import logger

config = {
    'handlers': [
        { 'sink': sys.stderr, 'format': '{message}', 'level': 'DEBUG' },
        { 'sink': 'log.log', 'format': '{message}', 'level': 'DEBUG' },
    ]
}
logger.configure(**config)  # type: ignore
logger.disable("torchexplorer")

log_indent_level = 0
def log(message: str, extract_level: str, class_name: str):
    indent = '| ' * log_indent_level
    logger.opt(colors=True).debug(
        f'{indent}<green>{extract_level}</green>:<blue>{class_name}</blue> {message}'
    )


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
        module: nn.Module, invocation_id: InvocationId=0
    ) -> ModuleInvocationStructure:
    global log_indent_level
    """Module must have had hooks already added and one forward pass."""

    log_indent_level += 1

    log_args = ['OUTER', module.__class__.__name__]
    log('Start extracting structure', *log_args)

    input_n = len(module.torchexplorer_metadata.input_gradfns[invocation_id])
    output_n = len(module.torchexplorer_metadata.output_gradfns[invocation_id])
    structure = ModuleInvocationStructure(module, invocation_id, input_n, output_n)
    structure.upstreams_fetched = False


    downstreams = []
    for i, output_gradfn in enumerate(_get_output_gradfns(structure)):
        if output_gradfn is None:
            continue
        
        node_name = f'Output {i}'
        downstreams.append(DownstreamStructureNode(
            node=node_name, input_index=i, gradfn=output_gradfn
        ))

    i = 0
    while i < len(downstreams):
        downstream = downstreams[i]
        i += 1

        log(f'Processing downstream {downstream}', *log_args)
        
        upstreams = _inner_recurse(structure, downstream.gradfn)

        log(f'Done inner recurse, got {len(upstreams)} upstreams', *log_args)

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
                log(f'Queueing {upstream.node.module.__class__.__name__}', *log_args)
                for j, input_gradfn in enumerate(_get_input_gradfns(upstream.node)):
                    if input_gradfn is None:
                        continue

                    log(f'Queueing upstream {input_gradfn}', *log_args)
                    downstreams.append(DownstreamStructureNode(
                        node=upstream.node, input_index=j, gradfn=input_gradfn
                    ))
                
                upstream.node.upstreams_fetched = True

    log('Done extracting structure', *log_args)
    log_indent_level -= 1
    return structure

def _inner_recurse(
        current_struct: ModuleInvocationStructure,
        gradfn: GradFn,
        gradfn_index: Optional[int]=None
    ) -> list[UpstreamStructureNode]:
    global log_indent_level
    # Some gradfns kind of "combine" all their inputs into one gradfn node in the graph
    # (e.g., BackwardHookFunctionBackward). To keep the parent dependencies exact,
    # we only recurse into the particular gradfn_index.

    log_indent_level += 1

    log_args = ['INNER', current_struct.module.__class__.__name__]
    
    log(f'Recursing on {gradfn} with index {gradfn_index}', *log_args)

    if gradfn is None:
        return []

    next_functions = gradfn.next_functions
    if 'BackwardHookFunctionBackward' in str(gradfn) and gradfn_index is not None:
        next_functions = (next_functions[gradfn_index],)

    log(f'Next functions: {gradfn.next_functions}    ->    {next_functions}', *log_args)
    metadata = gradfn.metadata

    is_enter_forward = 'input_index' in metadata
    is_exit_forward = 'output_index' in metadata

    if is_enter_forward:
        assert metadata['module'] == current_struct.module
        assert metadata['invocation_id'] == current_struct.invocation_id
        name = f'Input {metadata["input_index"]}'

        log(f'Entering forward, input index {metadata["input_index"]}', *log_args)
        log_indent_level -= 1
        return [UpstreamStructureNode(name, metadata['input_index'])]
    
    if is_exit_forward:
        upstream_module = metadata['module']
        invocation_id = metadata['invocation_id']

        log(f'Exiting forward, output index {metadata["output_index"]}', *log_args)
        log(f'Upstream module: {upstream_module.__class__.__name__}', *log_args)

        upstream_struct = current_struct.get_inner_structure(
            upstream_module, invocation_id
        )

        if upstream_struct is None:
            log(f'Creating new upstream structure', *log_args)
            upstream_struct = extract_structure(upstream_module, invocation_id)
            current_struct.inner_graph.add_node(
                upstream_struct,
                memory_id=id(upstream_struct),
                label=upstream_module.__class__.__name__,
                tooltip=str(upstream_module)
            )
        
        log_indent_level -= 1
        return [UpstreamStructureNode(upstream_struct, metadata['output_index'])]

    if 'upstreams_cache' in gradfn.metadata:
        if gradfn_index in gradfn.metadata['upstreams_cache']:
            return gradfn.metadata['upstreams_cache'][gradfn_index]

    all_upstreams = _flatten([
        _inner_recurse(current_struct, next[0], next[1]) for next in next_functions
    ])

    if 'upstreams_cache' not in gradfn.metadata:
        gradfn.metadata['upstreams_cache'] = {}
    
    gradfn.metadata['upstreams_cache'][gradfn_index] = all_upstreams

    log(f'Returning {len(all_upstreams)} intermediate upstreams', *log_args)

    log_indent_level -= 1
    return all_upstreams


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


# Adapted from: https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
def _flatten(items, seqtypes=(list, tuple)):
    for i, _ in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items