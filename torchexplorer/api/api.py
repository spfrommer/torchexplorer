from __future__ import annotations

from torch import nn
import wandb

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from torchexplorer import core
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
    """Bump the open file limit. Must be called before wandb.init().
    
    This is a necessary workaround for long wandb training runs (see README common
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
        module: nn.Module,
        log: list[str] = ['io', 'io_grad', 'params', 'params_grad'],
        log_freq: int = 500,
        ignore_io_grad_classes: list[type] = [],
        disable_inplace: bool = False,
        bins: int = 20,
        sample_n: Optional[int] = 100,
        reject_outlier_proportion: float = 0.1,
        time_log: tuple[str, Callable] = ('step', lambda module, step: step),
        backend: Literal['wandb', 'standalone', 'none'] = 'wandb',
        standalone_dir: str = './torchexplorer_standalone',
        standalone_port: int = 5000,
        verbose: bool = True,
    ) -> StructureWrapper:
    """Watch a module and log its structure and histograms to a backend.

    Args:
        module (nn.Module): The module to watch.
        log (list[str]): What to log. Can be a subset of
            ['io', 'io_grad', 'params', 'params_grad'].
        log_freq (int): How many backwards passes to wait between logging.
        ignore_io_grad_classes (list[type]): A list of classes to ignore when logging
            io_grad. This is useful for ignoring classes which do inplace operations,
            which will throw an error.
        disable_inplace (bool): disables the 'inplace' attribute for all activations in
            the module.
        bins (int): The number of bins to use for histograms.
        sample_n (Optional[int]): The number of tensor elements to randomly sample for
            histograms. Passing "None" will sample all elements.
        reject_outlier_proportion (float): The proportion of outliners to reject when
            computing histograms, based on distance to the median. 0.0 means reject
            nothing, 1.0 rejects everything. Helps chart stay in a reasonable range.
        time_log: ([tuple[str, Callable]): A tuple of (time_unit, Callable) to use for
            logging. The allable should take in the module and step and return a value
            to log. The time_unit string is just the axis label on the histogram graph.
            If "module" is a pytorch lightning modules, torchexplorer.LIGHTNING_EPOCHS
            should work to change the time axis to epochs.
        backend (Literal['wandb', 'standalone', 'none']): The backend to log to. If
            'wandb', there must be an active wandb run. Otherwise, a standalone web app
            will be created in the standalone_dir.
        standalone_dir (str): The directory to create the standalone web app in. Only
            matters if the 'standalone' backend is selected.
        standalone_port (int): The port to run the standalone server on. Only matters if
            the 'standalone' backend is selected.
        verbose (bool): Whether to print out standalone server start message.
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


    def post_forward_hook(module, __, ___):
        nonlocal step_counter, wrapper
        if module.training and (wrapper.structure is None):
            wrapper.structure = structure.extract_structure(module)


    layout_cache = None
    def post_backward_hook(_, __, ___):
        # This hook is called after we've backprop'd and called all the other hooks
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

    module.register_forward_hook(post_forward_hook)
    module.register_full_backward_hook(post_backward_hook)

    watch_counter += 1

    return wrapper


def _disable_inplace(module: nn.Module):
    def disable(m: nn.Module):
        if hasattr(m, 'inplace'):
            m.inplace = False # type: ignore

    module.apply(disable)