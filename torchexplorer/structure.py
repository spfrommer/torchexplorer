from torch import nn

from typing import Optional, Union
from dataclasses import dataclass

from torchexplorer.core import GradFn, InvocationId, ModuleInvocationStructure

StructureNode = Union[ModuleInvocationStructure, str]


@dataclass
class UpstreamStructureNode:
    node: StructureNode
    output_index: int

    def __str__(self):
        node_str = self.node if isinstance(self.node, str) else self.node.str_impl()
        return f'(Upstream {node_str}):{self.output_index}'


@dataclass
class DownstreamStructureNode:
    node: StructureNode
    input_index: int
    gradfn: GradFn

    def __str__(self):
        node_str = self.node if isinstance(self.node, str) else self.node.str_impl()
        return f'(Downstream {node_str}):{self.input_index}'


def extract_structure(
        module: nn.Module, invocation_id: InvocationId=0
    ) -> ModuleInvocationStructure:
    """Module must have had hooks already added and one forward pass."""

    structure = ModuleInvocationStructure(module, invocation_id)
    structure.upstreams_fetched = False


    downstreams = []
    for i, output_gradfn in enumerate(_get_output_gradfns(structure)):
        if output_gradfn is None:
            continue
        downstreams.append(DownstreamStructureNode(
            node='Output', input_index=i, gradfn=output_gradfn
        ))

    i = 0
    while i < len(downstreams):
        downstream = downstreams[i]
        i += 1
        
        upstreams = _inner_recurse(structure, downstream.gradfn)

        for upstream in upstreams:
            structure.inner_graph.add_edge(
                upstream.node,
                downstream.node,
                upstream_output_index=upstream.output_index,
                downstream_input_index=downstream.input_index,
            )

            if upstream.node == 'Input':
                continue

            assert isinstance(upstream.node, ModuleInvocationStructure)
                
            if not upstream.node.upstreams_fetched:
                for j, input_gradfn in enumerate(_get_input_gradfns(upstream.node)):
                    if input_gradfn is None:
                        continue
                    downstreams.append(DownstreamStructureNode(
                        node=upstream.node, input_index=j, gradfn=input_gradfn
                    ))
                
                upstream.node.upstreams_fetched = True

    return structure

def _inner_recurse(
        current_struct: ModuleInvocationStructure, gradfn: GradFn,
    ) -> list[UpstreamStructureNode]:

    if gradfn is None:
        return []

    next_functions = gradfn.next_functions
    metadata = gradfn.metadata

    is_enter_forward = 'input_index' in metadata
    is_exit_forward = 'output_index' in metadata

    if is_enter_forward:
        assert metadata['module'] == current_struct.module
        assert metadata['invocation_id'] == current_struct.invocation_id
        return [UpstreamStructureNode('Input', metadata['input_index'])]
    
    if is_exit_forward:
        upstream_module = metadata['module']
        invocation_id = metadata['invocation_id']

        upstream_struct = current_struct.get_inner_structure(
            upstream_module, invocation_id
        )

        if upstream_struct is None:
            upstream_struct = extract_structure(upstream_module, invocation_id)
            current_struct.inner_graph.add_node(
                upstream_struct,
                memory_id=id(upstream_struct),
                label=upstream_module.__class__.__name__,
                tooltip=str(upstream_module)
            )
        
        return [UpstreamStructureNode(upstream_struct, metadata['output_index'])]

    if 'upstreams_cache' in gradfn.metadata:
        return gradfn.metadata['upstreams_cache']

    all_upstreams = _flatten([
        _inner_recurse(current_struct, parents[0]) for parents in next_functions
    ])

    gradfn.metadata['upstreams_cache'] = all_upstreams

    return all_upstreams


def _get_input_gradfns(
        structure: ModuleInvocationStructure
    ) -> tuple[Optional[GradFn], ...]:

    return structure.module_metadata().input_gradfns[structure.invocation_id]

def _get_output_gradfns(
        structure: ModuleInvocationStructure
    ) -> tuple[Optional[GradFn], ...]:

    return structure.module_metadata().output_gradfns[structure.invocation_id]


# Adapted from: https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
def _flatten(items, seqtypes=(list, tuple)):
    for i, _ in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items