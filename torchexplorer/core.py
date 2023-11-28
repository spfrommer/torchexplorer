from __future__ import annotations

from typing import Callable, Optional, List
import torch
from torch import Tensor
from torch import nn
import networkx as nx

from functools import partial

from dataclasses import dataclass, field
from torchexplorer.histogram import IncrementalHistogram


GradFn = torch.autograd.Function
InvocationId = int
ParamName = str
OTensor = Optional[Tensor]
# For tracking the size of inputs / outputs
AdaptiveSize = Optional[List[Optional[int]]]

@dataclass
class ModuleInvocationHistograms:
    """The histograms associated to a particular InvocationId on a module."""
    input_hists: list[IncrementalHistogram] = field(default_factory=lambda: [])
    output_hists: list[IncrementalHistogram] = field(default_factory=lambda: [])


dict_field = partial(field, default_factory=lambda: {})

@dataclass
class ModuleSharedHistograms:
    """The histograms are shared across all InvocationId on a module."""
    param_hists: dict[ParamName, IncrementalHistogram] = dict_field()
    param_grad_hists: dict[ParamName, IncrementalHistogram] = dict_field()


@dataclass
class ExplorerMetadata:
    """Metadata associated to a module, saved as 'module.torchexplorer_metadata'."""

    # Cleared before every forwards pass
    input_gradfns: dict[InvocationId, tuple[Optional[GradFn], ...]] = dict_field()
    output_gradfns: dict[InvocationId, tuple[Optional[GradFn], ...]] = dict_field()
    forward_invocation_counter = 0
    backward_invocation_counter = 0
    has_tracking_hooks = False

    # Histograms are persisted during trainng
    invocation_hists: dict[InvocationId, ModuleInvocationHistograms] = dict_field()
    invocation_grad_hists: dict[InvocationId, ModuleInvocationHistograms] = dict_field()

    shared_hists: ModuleSharedHistograms = field(
        default_factory=lambda: ModuleSharedHistograms()
    )

    # Input / output sizes are persisted and are dynamically updated
    # As inputs / outputs are processed, the shape is recorded as an int tuple. If a
    # particular dimension has variable size, it is recorded as None. If the number of
    # dimensions is variable, the size overall is just None. The list is for multiple
    # inputs / outputs.
    input_sizes: dict[InvocationId, list[AdaptiveSize]] = dict_field()
    output_sizes: dict[InvocationId, list[AdaptiveSize]] = dict_field()


class ModuleInvocationStructure():
    """The parsed structure of a module invocation.

    There can be multiple of these for a particular module if that module's forward
    method is invoked multiple times on the forwards pass of a parent."""

    def __init__(
            self,
            module: nn.Module,
            invocation_id: InvocationId,
            structure_id: int,
            input_n: int,
            output_n: int
        ):

        self.module = module
        self.invocation_id = invocation_id
        # A unique id for this structure, to enable caching of graphviz calls
        self.structure_id = structure_id

        # Nodes are either 'Input x'/'Output x' strings or ModuleInvocationStructures
        self.inner_graph = nx.DiGraph()

        for i in range(input_n):
            name = f'Input {i}'
            self.inner_graph.add_node(name, memory_id=None, label=name, tooltip={})
        
        for i in range(output_n):
            name = f'Output {i}'
            self.inner_graph.add_node(name, memory_id=None, label=name, tooltip={})

        self.upstreams_fetched = False

        self.graphviz_json_cache: Optional[dict] = None

    def module_metadata(self) -> ExplorerMetadata:
        return self.module.torchexplorer_metadata

    def get_inner_structure(
            self, module: nn.Module, invocation_id: int
        ) -> Optional['ModuleInvocationStructure']:

        return self._inner_filter(
            lambda node: node.module == module and node.invocation_id == invocation_id
        )

    def get_inner_structure_from_memory_id(
            self, memory_id: int
        ) -> Optional['ModuleInvocationStructure']:

        return self._inner_filter(lambda node: id(node) == memory_id)

    def get_inner_structure_from_structure_id(
            self, structure_id: int
        ) -> Optional['ModuleInvocationStructure']:

        return self._inner_filter(lambda node: node.structure_id == structure_id)

    def _inner_filter(self, test_fn: Callable) -> Optional['ModuleInvocationStructure']:
        for node in self.inner_graph.nodes:
            if isinstance(node, ModuleInvocationStructure):
                if test_fn(node):
                    return node

        return None

    # NOTE: Overriding __str__ breaks the graphviz rendering...
    def str_impl(self) -> str:
        return f'{self.module.__class__.__name__}, Invocation {self.invocation_id}'