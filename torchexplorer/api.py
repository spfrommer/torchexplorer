from __future__ import annotations

import os
import shutil
import json
import threading
from torch import nn
import wandb
import sys

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from torchexplorer import core, hook, layout, structure
from torchexplorer.histogram import HistogramParams


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
        sample_n: int = 100,
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
        sample_n (int): The number of tensor elements to randomly sample for histograms.
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

    hist_params = HistogramParams(bins, sample_n, reject_outlier_proportion, time_name)
    step_counter = 0
    should_log_callable = lambda: (step_counter % log_freq == 0) and module.training
    wrapper = StructureWrapper()

    if backend == 'standalone':
        if verbose:
            print(f'Starting TorchExplorer at http://localhost:{standalone_port}')
        _standalone_backend_init(standalone_dir, standalone_port)

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


    # Make sure that the backward hook is getting the watch counter from when the
    # specific watch() was called, not the last watch
    watch_counter_copy = watch_counter
    layout_cache = None
    def post_backward_hook(_, __, ___):
        # This hook is called after we've backprop'd and called all the other hooks
        nonlocal step_counter, wrapper, layout_cache
        step_counter += 1

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

            if backend == 'wandb':
                _wandb_backend_update(renderable, watch_counter_copy)
            elif backend == 'standalone':
                _standalone_backend_update(renderable, standalone_dir)


    module.register_forward_hook(post_forward_hook)
    module.register_full_backward_hook(post_backward_hook)

    watch_counter += 1

    return wrapper


def _wandb_backend_update(renderable: layout.ModuleInvocationRenderable, counter: int):
    if wandb.run is None:
        raise ValueError('Call `wandb.init` before `torchexplorer.watch`')

    explorer_table, fields = layout.wandb_table(renderable)

    chart = wandb.plot_table(
        vega_spec_name='spfrom_team/torchexplorer_v2a',
        data_table=explorer_table,
        fields=fields,
        string_fields={}
    )

    wandb.log({f'explorer_chart_{counter}': chart}, commit=False)


def _standalone_backend_init(standalone_dir: str, standalone_port: int):
    source_explorer_dir = os.path.dirname(__file__)
    source_app_path = os.path.join(source_explorer_dir, 'standalone')
    target_app_path = os.path.abspath(standalone_dir)
    source_vega_path = os.path.join(source_explorer_dir, 'vega/vega_dataless.json')
    target_vega_path = os.path.join(target_app_path, 'vega_dataless.json')

    if not os.path.exists(target_app_path):
        shutil.copytree(source_app_path, target_app_path)
        shutil.copyfile(source_vega_path, target_vega_path)

    # Launch flask app
    sys.path.insert(1, target_app_path)
    import app # type: ignore
    app.vega_spec_path = target_vega_path
    threading.Thread(target=lambda: app.app.run(port=standalone_port)).start()


def _standalone_backend_update(
        renderable: layout.ModuleInvocationRenderable, standalone_dir: str,
    ):

    data = layout.serialized_rows(renderable)
    target_app_path = os.path.abspath(standalone_dir)

    data_path = os.path.join(target_app_path, 'data', 'data.json')
    with open(data_path, 'w') as f:
        f.write(json.dumps(data))


def _disable_inplace(module: nn.Module):
    def disable(m: nn.Module):
        if hasattr(m, 'inplace'):
            m.inplace = False # type: ignore

    module.apply(disable)
