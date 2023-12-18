from __future__ import annotations


import torch
from torch.nn import Module
from torch import Tensor

from typing import Callable, Optional, Union, Tuple

from torchexplorer.core import (
    SizeTracker, ModuleInvocationHistograms, ExplorerMetadata, ModuleSharedHistograms,
    OTensor, GradFn
)


from torchexplorer.components.histogram import HistogramParams, IncrementalHistogram
from torchexplorer import utils


OTensorOrTuple = Union[OTensor, Tuple[OTensor, ...]]


def push_histogram_histories(
        module: Module,
        hist_params: HistogramParams,
        time: int,
        log_params: bool,
        log_params_grad: bool,
    ):
    """Pushes the histograms of a module and its children to their histories."""
    
    metadata = module.torchexplorer_metadata
    
    shared_hists: ModuleSharedHistograms = metadata.shared_hists
    for name, param in module._parameters.items():
        if log_params and (param is not None):
            if name not in shared_hists.param_hists:
                shared_hists.param_hists[name] = IncrementalHistogram(hist_params)

            hist = shared_hists.param_hists[name]
            hist.update(param.detach())
        if log_params_grad and (param is not None) and (param.grad is not None):
            if name not in shared_hists.param_grad_hists:
                shared_hists.param_grad_hists[name] = IncrementalHistogram(hist_params)

            hist = shared_hists.param_grad_hists[name]
            hist.update(param.grad.detach())


    def all_hists(h: ModuleInvocationHistograms) -> list[IncrementalHistogram]:
        return h.input_hists + h.output_hists
    
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    all_histograms = (
        flatten([all_hists(h) for h in metadata.invocation_hists.values()]) +
        flatten([all_hists(h) for h in metadata.invocation_grad_hists.values()]) +
        list(metadata.shared_hists.param_hists.values()) + 
        list(metadata.shared_hists.param_grad_hists.values())
    )

    for histogram in all_histograms:
        # For reused modules, don't want to push the histograms twice
        if max(histogram.bin_counts) > 0:
            histogram.push_history(time)

    for child in module.children():
        push_histogram_histories(child, hist_params, time, log_params, log_params_grad)


def hook(
        module: Module,
        should_log_callable: Callable,
        log_io=True,
        log_io_grad=True,
        ignore_io_grad_classes: list[type] = [],
        hist_params: HistogramParams = HistogramParams()
    ):

    def _add_metadata(module: Module):
        module.torchexplorer_metadata = ExplorerMetadata() # type: ignore

    def _clear_temporary_metadata_forward(module: Module):
        module.torchexplorer_metadata.input_gradfns.clear()
        module.torchexplorer_metadata.output_gradfns.clear()
        module.torchexplorer_metadata.forward_invocation_counter = 0

    module.apply(_add_metadata)

    @return_if_not_should_log(should_log_callable)
    def pre_hook(module: Module, inputs: tuple[OTensor, ...]):
        module.apply(_clear_temporary_metadata_forward)
        # Make sure we are tracking gradients back to inputs
        return _add_dummy(inputs)

    module.register_forward_pre_hook(pre_hook)

    _add_tracking_hooks(module, should_log_callable)
    _add_size_record_hooks(module, should_log_callable)

    if log_io:
        _add_io_histogram_hooks(module, hist_params, should_log_callable)
    if log_io_grad:
        _add_io_grad_histogram_hooks(
            module, hist_params, should_log_callable, ignore_io_grad_classes
        )
    
    def _clear_temporary_metadata_backward(module: Module):
        module.torchexplorer_metadata.backward_invocation_counter = 0

    @return_if_not_should_log(should_log_callable)
    def backward_hook(module: Module, _, __):
        module.apply(_clear_temporary_metadata_backward)
    module.register_full_backward_hook(backward_hook)

def _add_tracking_hooks(module: Module, should_log_callable: Callable):
    @return_if_not_should_log(should_log_callable)
    def pre_hook(module: Module, inputs: tuple[OTensor, ...]):
        # In case an input is passed directly to a submodule
        # Honestly not really sure why it's necessary but it is
        inputs = _add_dummy(inputs)

        metadata: ExplorerMetadata = module.torchexplorer_metadata
        input_gradfns = _get_next_gradfns(inputs)
        metadata.input_gradfns[metadata.forward_invocation_counter] = input_gradfns
        metadata.forward_invocation_counter += 1

        # Add a graph "spacer" so that we have different gradfns for stopping an outer
        # recursion and starting an inner recursion.
        return _add_dummy(inputs)

    @return_if_not_should_log(should_log_callable)
    def post_hook(module: Module, inputs: tuple[OTensor, ...], outputs: OTensorOrTuple):
        outputs_tuple, single_output = _ensure_tuple(outputs)

        metadata: ExplorerMetadata = module.torchexplorer_metadata
        invocation_id = metadata.forward_invocation_counter - 1
        assert invocation_id >= 0

        # Just in case an input is passed directly through to an output
        outputs_tuple = _add_dummy(outputs_tuple)

        # These gradfns are used to recurse inside the module. So we record them
        # before adding a dummy operation to the graph.
        metadata.output_gradfns[invocation_id] = _get_tensor_gradfns(outputs_tuple)
        
        # Add a graph "spacer" so that we have different gradfns for stopping an outer
        # recursion and starting an inner recursion.
        outputs_tuple = _add_dummy(outputs_tuple)

        for io_gradfns, index_name in [
            (_get_next_gradfns(inputs), 'input_index'),
            (_get_tensor_gradfns(outputs_tuple), 'output_index')
        ]:
            for i, gradfn in utils.enum_not_none(io_gradfns):
                assert isinstance(gradfn.metadata, dict)

                gradfn.metadata['module'] = module
                gradfn.metadata[index_name] = i
                gradfn.metadata['invocation_id'] = invocation_id

        return outputs_tuple[0] if single_output else outputs_tuple

    def add_hooks(module: Module, append_hook_function=True):
        if not module.torchexplorer_metadata.has_tracking_hooks:
            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)
            module.torchexplorer_metadata.has_tracking_hooks = True

            if append_hook_function:
                module.torchexplorer_metadata.hook_registration_functions.append(
                    lambda m: add_hooks(m, append_hook_function=False)
                )
    
    module.apply(add_hooks)

def _add_size_record_hooks(module: Module, should_log_callable: Callable):
    def record_sizes(tensors: tuple[OTensor, ...], tensor_trackers: list[SizeTracker]):
        for i, tensor in utils.enum_not_none(tensors):
            shape = list(tensor.shape)

            if len(tensor_trackers) <= i:
                tensor_trackers.append(SizeTracker(shape, tensor.type()))
            elif tensor_trackers[i].size is not None:
                stored_shape = tensor_trackers[i].size
                assert stored_shape is not None
                if len(shape) != len(stored_shape):
                    tensor_trackers[i].size = None
                
                # TODO: with graphviz caching, this actually doesn't need to run
                # since only the first pass sizes are stored.
                for j in range(len(shape)):
                    if shape[j] != stored_shape[j]:
                        stored_shape[j] = None

    @return_if_not_should_log(should_log_callable)
    def post_hook(module: Module, inputs: tuple[OTensor, ...], outputs: OTensorOrTuple):
        outputs_tuple, _ = _ensure_tuple(outputs)

        metadata: ExplorerMetadata = module.torchexplorer_metadata
        invocation_id = metadata.forward_invocation_counter - 1

        for (tensors, sizes) in [
            (inputs, metadata.input_sizes), (outputs_tuple, metadata.output_sizes)
        ]:
            record_sizes(tensors, sizes.setdefault(invocation_id, []))

    def add_hook(module: Module, append_hook_function=True):
        if not module.torchexplorer_metadata.has_size_record_hooks:
            module.register_forward_hook(post_hook)
            module.torchexplorer_metadata.has_size_record_hooks = True

            if append_hook_function:
                module.torchexplorer_metadata.hook_registration_functions.append(
                    lambda m: add_hook(m, append_hook_function=False)
                )
    module.apply(add_hook)

def _add_io_histogram_hooks(
        module: Module, hist_params: HistogramParams, should_log_callable: Callable
    ):

    @return_if_not_should_log(should_log_callable)
    def post_hook(module: Module, inputs: tuple[OTensor, ...], outputs: OTensorOrTuple):
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

    def add_hook(module: Module, append_hook_function=True):
        if not module.torchexplorer_metadata.has_io_histogram_hooks:
            module.register_forward_hook(post_hook)
            module.torchexplorer_metadata.has_io_histogram_hooks = True

            if append_hook_function:
                module.torchexplorer_metadata.hook_registration_functions.append(
                    lambda m: add_hook(m, append_hook_function=False)
                )
    
    module.apply(add_hook)

def _add_io_grad_histogram_hooks(
        module: Module,
        hist_params: HistogramParams,
        should_log_callable: Callable,
        ignore_classes: list[type]
    ):

    @return_if_not_should_log(should_log_callable)
    def backward_hook(
            module: Module, grads_input: OTensorOrTuple, grads_output: OTensorOrTuple
        ):

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
            while len(hists) < len(tensors):
                hists.append(IncrementalHistogram(hist_params))

            for i, tensor in utils.enum_not_none(tensors):
                if len(tensor.shape) == 0:
                    tensor = tensor.reshape(1, 1)
                tensor = tensor.reshape(tensor.shape[0], -1)
                norms = torch.norm(tensor.float(), dim=1)
                hists[i].update(norms.detach())

    def add_hook(module, append_hook_function=True):
        if not module.torchexplorer_metadata.has_io_grad_histogram_hooks:
            module.register_full_backward_hook(backward_hook)
            module.torchexplorer_metadata.has_io_grad_histogram_hooks = True

            if append_hook_function:
                module.torchexplorer_metadata.hook_registration_functions.append(
                    lambda m: add_hook(m, append_hook_function=False)
                )

    if type(module) not in ignore_classes:
        add_hook(module)

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

def _add_dummy(tensors: tuple[OTensor, ...]) -> tuple[OTensor, ...]:
    def process_tensor(tensor: Tensor) -> OTensor:
        dummy_tensor = torch.tensor(0.0, requires_grad=True)
        return tensor + dummy_tensor if torch.is_floating_point(tensor) else tensor
    return tuple(process_tensor(tensor) for tensor in utils.iter_not_none(tensors))

def _get_tensor_gradfns(tensors: tuple[OTensor, ...]) -> tuple[Optional[GradFn], ...]:
    def process_tensor(tensor: Tensor) -> Optional[GradFn]:
        return tensor.grad_fn if tensor.requires_grad else None # type: ignore

    return tuple(process_tensor(tensor) for tensor in utils.iter_not_none(tensors))

def _get_next_gradfns(tensors: tuple[OTensor, ...]) -> tuple[Optional[GradFn], ...]:
    # Hacky workaround, couldn't figure out import
    # Check if all gradfns are the same 'BackwardHookFunctionBackward'
    backhook_class = 'BackwardHookFunctionBackward'
    if tensors[0] is not None and backhook_class in str(tensors[0].grad_fn):
        # Multiple inputs will share a BackwardHookFunctionBackward gradfn.
        # To tease apart multiple input nodes in the explorer, we need to go
        # one level deeper.
        return tuple(f[0] for f in tensors[0].grad_fn.next_functions) # type: ignore

    return _get_tensor_gradfns(tensors)

def return_if_not_should_log(should_log_callable: Callable):
    def decorator(hook):
        def wrapper(*args, **kwargs):
            if not should_log_callable():
                return
            return hook(*args, **kwargs)
        return wrapper
    return decorator