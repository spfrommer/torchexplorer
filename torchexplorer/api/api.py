from __future__ import annotations

from torch import Tensor, nn
from torch.nn import Module
import wandb

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

from torchexplorer import core
from torchexplorer.core import ExplorerMetadata
from torchexplorer.api.backend import (
    Backend, DummyBackend, StandaloneBackend, WandbBackend
)
from torchexplorer.components.histogram import HistogramParams
from torchexplorer.hook import hook
from torchexplorer.render import layout
from torchexplorer.structure import structure


@dataclass
class StructureWrapper:
    structure: Optional[core.ModuleInvocationStructure] = None


LIGHTNING_EPOCHS = ('epoch', lambda module, step: module.current_epoch)


def setup(ulimit=50000):
    """Bump the open file limit. Must be called before `wandb.init()`.
    
    This is a necessary workaround for long wandb training runs (see `README.md` common
    errors #4).
    
    Args:
        ulimit (int): The new open file limit.
    """
    if wandb.run is not None:
        raise ValueError('Call `torchexplorer.setup()` before `wandb.init()`')

    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit, ulimit))

watch_counter = 0

def watch(
        module: Module,
        log: list[str] = ['io', 'io_grad', 'params', 'params_grad'],
        log_freq: int = 500,
        ignore_io_grad_classes: list[type] = [],
        disable_inplace: bool = False,
        bins: int = 20,
        sample_n: Optional[int] = 100,
        reject_outlier_proportion: float = 0.1,
        time_log: tuple[str, Callable] = ('step', lambda module, step: step),
        delay_log_multi_backward: bool = False,
        backend: Literal['wandb', 'standalone', 'none'] = 'wandb',
        standalone_dir: str = './torchexplorer_standalone',
        standalone_port: int = 5000,
        verbose: bool = True,
    ) -> StructureWrapper:
    """Watch a module and log its structure and histograms to a backend.

    Args:
        module (Module): The module to watch.
        log (list[str]): What to log. Can be a subset of
            ['io', 'io_grad', 'params', 'params_grad'].
        log_freq (int): How many backwards passes to wait between logging.
        ignore_io_grad_classes (list[type]): A list of classes to ignore when logging
            io_grad. This is useful for ignoring classes which do inplace operations,
            which will throw an error.
        disable_inplace (bool): disables the `inplace` attribute for all activations in
            the module.
        bins (int): The number of bins to use for histograms.
        sample_n (Optional[int]): The number of tensor elements to randomly sample for
            histograms. Passing `None` will sample all elements.
        reject_outlier_proportion (float): The proportion of outliners to reject when
            computing histograms, based on distance to the median. `0.0` means reject
            nothing, `1.0` rejects everything. Helps chart stay in a reasonable range.
        time_log: ([tuple[str, Callable]): A tuple of (time_unit, Callable) to use for
            logging. The allable should take in the module and step and return a value
            to log. The time_unit string is just the axis label on the histogram graph.
            If `module` is a pytorch lightning modules, `torchexplorer.LIGHTNING_EPOCHS`
            should work to change the time axis to epochs.
        delay_log_multi_backward (bool): Whether to delay logging until after all
            backward hooks have been called. This is useful if the module argument is
            invoked multiple times in one step before `backward()` is called on the
            loss. 
        backend (Literal['wandb', 'standalone', 'none']): The backend to log to. If
            'wandb', there must be an active wandb run. Otherwise, a standalone web app
            will be created in the `standalone_dir`.
        standalone_dir (str): The directory to create the standalone web app in. Only
            matters if the 'standalone' backend is selected.
        standalone_port (int): The port to run the standalone server on. Only matters if
            the 'standalone' backend is selected.
        verbose (bool): Whether to print out standalone server start message.
    
    Returns:
        StructureWrapper: A wrapper around the module's structure. Access the parsed
            structure with `wrapper.structure`.
    """

    global watch_counter

    if time_log is None:
        time_log = ('step', lambda module, step: step)
    time_name, time_fn = time_log

    if disable_inplace:
        _disable_inplace(module)

    wrapper = StructureWrapper()
    hist_params = HistogramParams(bins, sample_n, reject_outlier_proportion, time_name)

    step_counter = 0
    should_log_callable = lambda: (step_counter % log_freq == 0) and module.training

    backend_handler: Backend
    if backend == 'wandb':
        backend_handler = WandbBackend(watch_counter)
    elif backend == 'standalone':
        backend_handler = StandaloneBackend(standalone_dir, standalone_port, verbose)
    elif backend == 'none':
        backend_handler = DummyBackend()

    hook.hook(
        module,
        should_log_callable,
        log_io='io' in log,
        log_io_grad='io_grad' in log,
        ignore_io_grad_classes=ignore_io_grad_classes,
        hist_params=hist_params
    )


    layout_cache = None
    backward_last_called = False
    def handle_step(module):
        nonlocal step_counter, backend_handler, wrapper, layout_cache
        if should_log_callable():
            time = time_fn(module, step_counter)
            hook.push_histogram_histories(
                module,
                hist_params,
                time=time,
                log_params='params' in log,
                log_params_grad='params_grad' in log
            )
            renderable, layout_cache = layout.layout(wrapper.structure, layout_cache)
            backend_handler.update(renderable)

        step_counter += 1

    def post_forward_hook(module, __, ___):
        nonlocal wrapper, backward_last_called
        if module.training and (wrapper.structure is None):
            wrapper.structure = structure.extract_structure(module)
        
        if delay_log_multi_backward and backward_last_called:
            handle_step(module)
        backward_last_called = False


    def post_backward_hook(_, __, ___):
        # This hook is called after we've backprop'd and called all the other hooks
        
        # We need this fancy scheme in case our module is invoked twice and both 
        # outputs go into the loss. Then we get two backward_hook invocations on one
        # backwards pass, but we only want to push histograms once for this.
        nonlocal backward_last_called

        if not delay_log_multi_backward:
            if backward_last_called:
                raise RuntimeError(
                    'Got two backward() calls in one step! This is only supported if '
                    'delay_log_multi_backward=True is passed to watch().'
                )
            handle_step(module)
        backward_last_called = True

    module.register_forward_hook(post_forward_hook)
    module.register_full_backward_hook(post_backward_hook)

    watch_counter += 1

    return wrapper


def attach(tensor: Tensor, parent: Module, name: str) -> Tensor:
    """Monitor a particular tensor within a `forward` call.

    Example:
        Intended usage::
        
            def forward(self, x):
                y = x + 3
                y = torch.attach(y, self, name='y_monitor')
                z = y * 5
                return z

    Example:
        Note that the return value must be assigned in order for this to have any
        effect. The following will do nothing by itself::

            torch.attach(y, self, name='y_monitor') # WRONG!!!

    Args:
        tensor (Tensor): The tensor to add to the graph.
        parent (Module): The `Module` whose `forward` method we are currently in.
            Usually will be `self`.
        name (str): The name of the resulting node in the interface.

    Returns:
        Tensor: the same tensor passed to `attach`. Must be assigned.
    """
    if not hasattr(parent, 'torchexplorer_attach_modules'):
        parent.torchexplorer_attach_modules = nn.ModuleDict()
        parent.torchexplorer_attach_modules.torchexplorer_metadata = ExplorerMetadata() # type: ignore
    
    metadata = parent.torchexplorer_metadata
    attach_modules = parent.torchexplorer_attach_modules

    if name not in attach_modules:
        dummy = core.DummyAttachModule()
        dummy.torchexplorer_metadata = ExplorerMetadata() # type: ignore
        dummy.attach_name = name # type: ignore
        for hook_registration_function in metadata.hook_registration_functions:
            hook_registration_function(dummy)
        attach_modules[name] = dummy

    return attach_modules[name](tensor)


def _disable_inplace(module: Module):
    def disable(m: Module):
        if hasattr(m, 'inplace'):
            m.inplace = False # type: ignore

    module.apply(disable)