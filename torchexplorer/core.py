from collections import defaultdict
import itertools
from typing import Callable, Optional, Union
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


class ModuleInvocationStructure():
    """The parsed structure of a module invocation.

    There can be multiple of these for a particular module if that module's forward
    method is invoked multiple times on the forwards pass of a parent."""

    def __init__(self, module: nn.Module, invocation_id: InvocationId):
        self.module = module
        self.invocation_id = invocation_id

        # Inner nodes are either 'Input'/'Output' strings or ModuleInvocationStructures
        self.inner_graph = nx.DiGraph()
        self.inner_graph.add_node(
            'Input', memory_id=None, label='Input', tooltip='Input'
        )
        self.inner_graph.add_node(
            'Output', memory_id=None, label='Output', tooltip='Output'
        )

        self.upstreams_fetched = False

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

    def _inner_filter(self, test_fn: Callable) -> Optional['ModuleInvocationStructure']:
        for node in self.inner_graph.nodes:
            if isinstance(node, ModuleInvocationStructure):
                if test_fn(node):
                    return node

        return None

    # NOTE: Overriding __str__ breaks the graphviz rendering...
    def str_impl(self) -> str:
        return f'{self.module.__class__.__name__}: Invocation {self.invocation_id}'
