import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
from torch import nn

from torchexplorer import StructureWrapper, watch
from torchexplorer.core import (
    ModuleInvocationHistograms, ModuleInvocationStructure
)

import infra


log_all = ['io', 'io_grad', 'params', 'params_grad'] 


def test_basic_mlp_structure():
    model, structure = _basic_mlp_structure()

    assert len(structure.inner_graph.nodes) == len(model) + 2
    assert len(structure.inner_graph.edges) == len(model) + 1

    for node in structure.inner_graph.nodes:
        if not isinstance(node, ModuleInvocationStructure):
            continue

        assert node.module in model
        assert node.invocation_id == 0
        assert node.inner_graph is not None
        assert len(node.inner_graph.nodes) == 2
        assert len(node.inner_graph.edges) == 1


def test_basic_mlp_histograms():
    model, structure = _basic_mlp_structure()
    epochs = 15

    for node in structure.inner_graph.nodes:
        if not isinstance(node, ModuleInvocationStructure):
            continue

        metadata = node.module_metadata()

        assert len(metadata.invocation_hists) == 1
        assert len(metadata.invocation_grad_hists) == 1

        hists: ModuleInvocationHistograms = metadata.invocation_hists[0]
        grad_hists: ModuleInvocationHistograms = metadata.invocation_grad_hists[0]

        assert len(hists.input_hists) == 1
        assert len(hists.output_hists) == 1

        assert len(grad_hists.input_hists) == 1
        assert len(grad_hists.output_hists) == 1

        assert len(hists.input_hists[0].history_bins) == epochs
        assert len(hists.output_hists[0].history_bins) == epochs

        assert len(grad_hists.input_hists[0].history_bins) == epochs
        assert len(grad_hists.output_hists[0].history_bins) == epochs


def _basic_mlp_structure() -> tuple[nn.Module, ModuleInvocationStructure]:
    X = torch.randn(5, 10)
    y = torch.randn(5, 2)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )

    wrapper: StructureWrapper = watch(model, log_freq=1, backend='none')
    infra.run_trial(model, X, y, steps=15)

    structure: ModuleInvocationStructure = wrapper.structure

    return model, structure



def test_repeat_relu_mlp_structure():
    model, structure = _repeat_relu_mlp_structure()

    assert len(structure.inner_graph.nodes) == len(model) + 2
    assert len(structure.inner_graph.edges) == len(model) + 1

    for i, node in enumerate(structure.inner_graph.nodes):
        if not isinstance(node, ModuleInvocationStructure):
            continue

        assert node.module in model
        # Nodes added in reverse order, so the last relu is first after input & output
        assert node.invocation_id == (1 if i == 2 else 0)
        assert node.inner_graph is not None
        assert len(node.inner_graph.nodes) == 2
        assert len(node.inner_graph.edges) == 1


def test_repeat_relu_mlp_histograms():
    model, structure = _repeat_relu_mlp_structure()
    epochs = 15

    for node in structure.inner_graph.nodes:
        if not isinstance(node, ModuleInvocationStructure):
            continue

        metadata = node.module_metadata()

        if isinstance(node.module, nn.ReLU):
            assert len(metadata.invocation_hists) == 2
            assert len(metadata.invocation_grad_hists) == 2
        else:
            assert len(metadata.invocation_hists) == 1
            assert len(metadata.invocation_grad_hists) == 1

        hists: ModuleInvocationHistograms = metadata.invocation_hists[0]
        grad_hists: ModuleInvocationHistograms = metadata.invocation_grad_hists[0]

        assert len(hists.input_hists) == 1
        assert len(hists.output_hists) == 1

        assert len(grad_hists.input_hists) == 1
        assert len(grad_hists.output_hists) == 1

        assert len(hists.input_hists[0].history_bins) == epochs
        assert len(hists.output_hists[0].history_bins) == epochs

        assert len(grad_hists.input_hists[0].history_bins) == epochs
        assert len(grad_hists.output_hists[0].history_bins) == epochs


def _repeat_relu_mlp_structure() -> tuple[nn.Module, ModuleInvocationStructure]:
    X = torch.randn(5, 10)
    y = torch.randn(5, 2)

    relu = nn.ReLU()

    model = nn.Sequential(
        nn.Linear(10, 20),
        relu,
        nn.Linear(20, 2),
        relu
    )

    wrapper: StructureWrapper = watch(model, log_freq=1, backend='none')
    infra.run_trial(model, X, y, steps=15)

    structure: ModuleInvocationStructure = wrapper.structure

    return model, structure


def test_repeat_relu_nested_structure():
    model, structure = _repeat_relu_nested_structure()

    assert len(structure.inner_graph.nodes) == len(model) + 2
    assert len(structure.inner_graph.edges) == len(model) + 1

    for i, node in enumerate(structure.inner_graph.nodes):
        if not isinstance(node, ModuleInvocationStructure):
            continue

        assert node.module in model
        # Nodes added in reverse order, so the last relu is first after input & output
        assert node.invocation_id == (2 if i == 2 else (1 if i == 4 else 0))
        assert node.inner_graph is not None
        
        if i != 5:
            assert len(node.inner_graph.nodes) == 2
            assert len(node.inner_graph.edges) == 1

    submodule_node = list(structure.inner_graph.nodes)[5]
    assert len(submodule_node.inner_graph.nodes) == 4
    assert len(submodule_node.inner_graph.edges) == 3

def _repeat_relu_nested_structure() -> tuple[nn.Module, ModuleInvocationStructure]:
    X = torch.randn(5, 10)
    y = torch.randn(5, 2)

    relu = nn.ReLU()

    submodule = nn.Sequential(
        relu,
        nn.Linear(20, 20)
    )
    model = nn.Sequential(
        nn.Linear(10, 20),
        submodule,
        relu,
        nn.Linear(20, 2),
        relu
    )

    wrapper: StructureWrapper = watch(model, log_freq=1, backend='none')
    infra.run_trial(model, X, y, steps=15)

    structure: ModuleInvocationStructure = wrapper.structure

    return model, structure


def test_inplace_structure():
    model, structure = _inplace_structure()

    assert len(structure.inner_graph.nodes) == len(model) + 2
    assert len(structure.inner_graph.edges) == len(model) + 1

    for i, node in enumerate(structure.inner_graph.nodes):
        if not isinstance(node, ModuleInvocationStructure):
            continue

        assert node.module in model
        assert node.invocation_id == 0
        assert node.inner_graph is not None

        if isinstance(node.module, InplaceModule):
            assert len(node.inner_graph.nodes) == 3
            assert len(node.inner_graph.edges) == 3
        else:
            assert len(node.inner_graph.nodes) == 2
            assert len(node.inner_graph.edges) == 1



class InplaceModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, input):
        x = self.fc1(input)
        x += input
        return x

def _inplace_structure() -> tuple[nn.Module, ModuleInvocationStructure]:
    X = torch.randn(5, 10)
    y = torch.randn(5, 2)

    model = nn.Sequential(
        InplaceModule(),
        nn.Linear(10, 20),
        nn.ReLU(inplace=True),
        nn.Linear(20, 2),
    )

    wrapper: StructureWrapper = watch(
        model,
        log_freq=1,
        ignore_io_grad_classes=[InplaceModule],
        disable_inplace=True,
        backend='none'
    )
    infra.run_trial(model, X, y, steps=15)

    structure: ModuleInvocationStructure = wrapper.structure

    return model, structure



def test_nondiff_structure():
    model, structure = _nondiff_structure()

    assert len(structure.inner_graph.nodes) == 5
    assert len(structure.inner_graph.edges) == 5


def test_nondiff_histograms():
    model, structure = _nondiff_structure()
    epochs = 15

    for node in structure.inner_graph.nodes:
        if isinstance(node, str):
            continue

        metadata = node.module_metadata()

        if isinstance(node.module, NonDiffSubModule):
            assert len(metadata.invocation_hists) == 1
            assert len(metadata.invocation_grad_hists) == 0

            invocation_hists = metadata.invocation_hists[0]
            assert len(invocation_hists.input_hists[0].history_bins) == epochs
            assert len(invocation_hists.output_hists[0].history_bins) == epochs
        elif isinstance(node.module, nn.Linear):
            assert not (node.module.in_features==20 and node.module.out_features==20)

            assert len(metadata.invocation_hists) == 1
            assert len(metadata.invocation_grad_hists) == 1

            invocation_hists = metadata.invocation_hists[0]
            grad_hists = metadata.invocation_grad_hists[0]
            assert len(invocation_hists.input_hists[0].history_bins) == epochs
            assert len(invocation_hists.output_hists[0].history_bins) == epochs

            assert len(grad_hists.input_hists[0].history_bins) == epochs
            assert len(grad_hists.output_hists[0].history_bins) == epochs

class NonDiffSubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 20)

    def forward(self, input_1):
        x = self.fc1(input_1)
        return torch.round(x).long().float()

class NonDiffModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.submodule = NonDiffSubModule()
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        skip = x
        x = self.submodule(x)
        x = self.fc2(x + skip)
        return x

def _nondiff_structure() -> tuple[nn.Module, ModuleInvocationStructure]:
    X = torch.randn(5, 10)
    y = torch.randn(5, 10)

    model = NonDiffModule()

    wrapper: StructureWrapper = watch(model, log_freq=1, backend='none')
    infra.run_trial(model, X, y, steps=15)

    structure: ModuleInvocationStructure = wrapper.structure

    return model, structure