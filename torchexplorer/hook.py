import torch
from torch import nn
from torch import Tensor

from typing import Callable, Optional, Union

from torchexplorer.core import (
    ModuleInvocationHistograms, ExplorerMetadata, ModuleSharedHistograms,
    OTensor, GradFn
)
from torchexplorer.histogram import HistogramParams, IncrementalHistogram


def hook(
        module: nn.Module,
        should_log_callable: Callable,
        log_io=True,
        log_io_grad=True,
        ignore_io_grad_classes: list[type] = [],
        hist_params: HistogramParams = HistogramParams()
    ):

    module.apply(_add_metadata)

    def pre_hook(module: nn.Module, inputs: tuple[OTensor, ...]):
        module.apply(_clear_temporary_metadata)
        # Make sure we are tracking gradients back to inputs
        return _add_dummy(inputs)

    module.register_forward_pre_hook(pre_hook)

    _add_tracking_hooks(module, should_log_callable)

    if log_io:
        _add_io_histogram_hooks(module, hist_params, should_log_callable)
    if log_io_grad:
        _add_io_grad_histogram_hooks(
            module, hist_params, should_log_callable, ignore_io_grad_classes
        )


def push_histogram_histories(
        module: nn.Module,
        hist_params: HistogramParams,
        time: int,
        log_params,
        log_params_grad,
    ):
    """Pushes the histograms of a module and its children to their histories."""
    
    metadata = module.torchexplorer_metadata
    
    shared_hists: ModuleSharedHistograms = metadata.shared_hists
    for name, param in module._parameters.items():
        if log_params and param is not None:
            if name not in shared_hists.param_hists:
                shared_hists.param_hists[name] = IncrementalHistogram(hist_params)

            hist = shared_hists.param_hists[name]
            hist.update(param.detach())
        
        if log_params_grad and param is not None and param.grad is not None:
            if name not in shared_hists.param_grad_hists:
                shared_hists.param_grad_hists[name] = IncrementalHistogram(hist_params)

            hist = shared_hists.param_grad_hists[name]
            hist.update(param.grad.detach())


    def get_hists(h: ModuleInvocationHistograms) -> list[IncrementalHistogram]:
        return h.input_hists + h.output_hists
    
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    all_histograms = (
        flatten([get_hists(h) for h in metadata.invocation_hists.values()]) +
        flatten([get_hists(h) for h in metadata.invocation_grad_hists.values()]) +
        list(metadata.shared_hists.param_hists.values()) + 
        list(metadata.shared_hists.param_grad_hists.values())
    )

    for histogram in all_histograms:
        # For reused modules, don't want to push the histograms twice
        if max(histogram.bin_counts) > 0:
            histogram.push_history(time)

    for child in module.children():
        push_histogram_histories(child, hist_params, time, log_params, log_params_grad)

def _add_dummy(tensors: tuple[OTensor, ...]) -> tuple[OTensor, ...]:
    dummy_tensor = torch.tensor(0.0, requires_grad=True)
    def process_tensor(tensor: OTensor) -> OTensor:
        if tensor is None:
            return None
        return tensor + dummy_tensor if torch.is_floating_point(tensor) else tensor
    return tuple(process_tensor(tensor) for tensor in tensors)


def _add_tracking_hooks(module: nn.Module, should_log_callable: Callable):
    def gradfns(tensors: tuple[OTensor, ...]) -> tuple[Optional[GradFn], ...]:
        def process_tensor(tensor: OTensor):
            if tensor is None or not tensor.requires_grad:
                return None
            return tensor.grad_fn

        return tuple(process_tensor(tensor) for tensor in tensors)

    def pre_hook(module: nn.Module, inputs: tuple[OTensor, ...]):
        if not should_log_callable():
            return

        metadata: ExplorerMetadata = module.torchexplorer_metadata
        metadata.input_gradfns[metadata.forward_invocation_counter] = gradfns(inputs)
        metadata.forward_invocation_counter += 1

        return _add_dummy(inputs)

    def post_hook(
            module: nn.Module,
            inputs: tuple[OTensor, ...],
            outputs: Union[OTensor, tuple[OTensor, ...]]
        ):

        if not should_log_callable():
            return

        outputs_tuple, single_output = _ensure_tuple(outputs)

        metadata: ExplorerMetadata = module.torchexplorer_metadata
        invocation_id = metadata.forward_invocation_counter - 1
        assert invocation_id >= 0
        metadata.output_gradfns[invocation_id] = gradfns(outputs_tuple)

        outputs_tuple = _add_dummy(outputs_tuple)

        for tensors, index_name in [
            (inputs, 'input_index'), (outputs_tuple, 'output_index')
        ]:
            for i, tensor in enumerate(tensors):
                if tensor is None or tensor.grad_fn is None:
                    continue

                assert isinstance(tensor.grad_fn.metadata, dict)

                tensor.grad_fn.metadata['module'] = module
                tensor.grad_fn.metadata[index_name] = i
                tensor.grad_fn.metadata['invocation_id'] = invocation_id

        return outputs_tuple[0] if single_output else outputs_tuple

    def add_hooks(module):
        if not module.torchexplorer_metadata.has_tracking_hooks:
            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)
            module.torchexplorer_metadata.has_tracking_hooks = True
    
    module.apply(add_hooks)

def _add_metadata(module: nn.Module):
    module.torchexplorer_metadata = ExplorerMetadata() # type: ignore

def _clear_temporary_metadata(module: nn.Module):
    module.torchexplorer_metadata.input_gradfns.clear()
    module.torchexplorer_metadata.output_gradfns.clear()
    module.torchexplorer_metadata.forward_invocation_counter = 0
    module.torchexplorer_metadata.backward_invocation_counter = 0

def _add_io_histogram_hooks(
        module: nn.Module, hist_params: HistogramParams, should_log_callable: Callable
    ):

    def post_hook(
            module: nn.Module,
            inputs: tuple[OTensor],
            outputs: Union[OTensor, tuple[OTensor]]
        ):

        if not should_log_callable():
            return

        outputs_tuple, _ = _ensure_tuple(outputs)

        metadata = module.torchexplorer_metadata
        invocation_id = metadata.forward_invocation_counter - 1

        if invocation_id not in metadata.invocation_hists:
            metadata.invocation_hists[invocation_id] = ModuleInvocationHistograms()
        inv_hists = metadata.invocation_hists[invocation_id]

        input_hists, output_hists = inv_hists.input_hists, inv_hists.output_hists
        for tensors, hists in [(inputs, input_hists), (outputs_tuple, output_hists)]:
            for i, tensor in enumerate(tensors):
                if len(hists) <= i:
                    hists.append(IncrementalHistogram(hist_params))
                if tensor is not None:
                    hists[i].update(tensor.detach())

    def hook_module(m: nn.Module):
        m.register_forward_hook(post_hook)
    module.apply(hook_module)

def _add_io_grad_histogram_hooks(
        module: nn.Module,
        hist_params: HistogramParams,
        should_log_callable: Callable,
        ignore_classes: list[type]
    ):

    def backward_hook(
            module: nn.Module,
            grads_input: Union[OTensor, tuple[OTensor, ...]],
            grads_output: Union[OTensor, tuple[OTensor, ...]]
        ) -> None:

        if not should_log_callable():
            return

        grads_input_tuple, _ = _ensure_tuple(grads_input)
        grads_output_tuple, _ = _ensure_tuple(grads_output)

        metadata = module.torchexplorer_metadata
        # When going backwards, assume that the backward hooks are called in reverse
        # order to the forward hooks.
        last_forward_counter = metadata.forward_invocation_counter - 1
        invocation_id = last_forward_counter - metadata.backward_invocation_counter
        metadata.backward_invocation_counter += 1

        if invocation_id not in metadata.invocation_grad_hists:
            metadata.invocation_grad_hists[invocation_id] = ModuleInvocationHistograms()
        inv_hists = metadata.invocation_grad_hists[invocation_id]

        for tensors, hists in (
            [(grads_input_tuple, inv_hists.input_hists),
             (grads_output_tuple, inv_hists.output_hists)]
        ):
            for i, tensor in enumerate(tensors):
                if len(hists) <= i:
                    hists.append(IncrementalHistogram(hist_params))

                if tensor is not None:
                    if len(tensor.shape) == 0:
                        tensor = tensor.reshape(1, 1)
                    tensor = tensor.reshape(tensor.shape[0], -1)
                    norms = torch.norm(tensor.float(), dim=1)
                    hists[i].update(norms.detach())

    if type(module) not in ignore_classes:
        module.register_full_backward_hook(backward_hook)

        for child in module.children():
            _add_io_grad_histogram_hooks(
                child, hist_params, should_log_callable, ignore_classes
            )

def _ensure_tuple(
        x: Union[OTensor, tuple[OTensor, ...]]
    ) -> tuple[tuple[OTensor, ...], bool]:

    is_single = x is None or isinstance(x, Tensor)
    ret_x = (x,) if is_single else x
    return ret_x, is_single # type: ignore 