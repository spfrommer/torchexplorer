<p align="center">
<img src="./res/logo.svg" width="500">
</p>

<p align="center">
<em> Made by <a href="https://sam.pfrommer.us/">Samuel Pfrommer</a> as part of <a href="https://www2.eecs.berkeley.edu/Faculty/Homepages/sojoudi.html">Somayeh Sojoudi's group</a> at Berkeley. </em>
</p>

<p align="center">
  <img src="./res/usage.gif" alt="animated" />
</p>


Curious about what's happening in your network? TorchExplorer is a simple tool that allows you to interactively inspect the inputs, outputs, parameters, and gradients for each `nn.Module` in your network. It integrates with [weights and biases](https://wandb.ai/site) and can also operate locally as a standalone solution. If your use case fits (see do's/don'ts below), it's very simple to try:

```python
model = ...

torchexplorer.watch(model, backend='wandb') # Or 'standalone'

# Training loop...
```

For full usage examples, see `/tests` and `/examples`. 

### Install
Installing requires one external `graphviz` dependency, which should be available on most package managers.

```bash
sudo apt-get install libgraphviz-dev graphviz
pip install torchexplorer
```

If you want to run the visualization as a standalone app for local training (as opposed to in weights and biases), you should also install `flask`:

```bash
pip install flask
```

## API

The api surface is just one function call, inspired by wandb's [watch](https://docs.wandb.ai/ref/python/watch).

```python
def watch(
    module: nn.Module,
    log: list[str] = ['io', 'io_grad', 'params', 'params_grad'],
    log_freq: int = 1000,
    ignore_io_grad_classes: list[type] = [],
    disable_inplace: bool = False,
    bins: int = 10,
    sample_n: int = 100,
    reject_outlier_proportion: float = 0,
    time_log: tuple[str, callable] = ('step', lambda module, step: step),
    backend: Literal['wandb', 'standalone', 'none'] = 'wandb',
    standalone_dir: str = './torchexplorer_standalone',
    standalone_port: int = 5000
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
        nothing, 1.0 means reject everything
    time_log: ([tuple[str, callable]): A tuple of (time_unit, callable) to use for
        logging. The callable should take in the module and step and return a value
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
"""
```

## Do's, don'ts, and other notes

1. When invoking a module, **don't use the `module.forward(x)` method**. Always call the forwards method as `module(x)`. The former does not call the hooks that `torchexplorer` uses.
2. Only call `.backward()` once in a training step.
3. **Recursive operations are not supported,** and **anything which dynamically changes the module-level control flow over training is not supported**. For instance, something like this isn't permissible:
```python
if x > 0:
    return self.module1(x)
else:
    return self.module2(x)
```
4. **Inplace operations are not supported** and should be corrected or filtered (see "Common errors" below).
5. Keyword tensor arguments to the forwards method are not supported. In other words, only positional arguments will be tracked. Behavior for keyword tensor arguments is untested as of now.
6. Nondifferentiable operations which break the autograd graph are permissible and should not cause a crash. However, the resulting module-level graph will be correspondingly disconnected.

## Common errors

This section includes a nonexhaustive list of errors that I've run into. For something not covered here, feel free to open a GitHub issue.


**1. Inplace operations in the computational graph**
 
```
RuntimeError: Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace...
```

This indicates that an inplace operation is occurring somewhere in the computational graph, which messes with the input/output gradient capturing (`io_grad`) feature. This commonly comes from inplace activations (e.g. `nn.ReLU(inplace=True)`), or residual inplace additions (e.g. `out += identity`). If you don't care about gradients you can just omit `'io_grad'` in `log` argument to the `watch` function. Otherwise, there are two additional tools available. You can use the `disable_inplace` argument to automatically turn off the `inplace` flag on all activations. If this still doesn't cut it, you must figure out what submodules are doing inplace operations and either manually fix them or pass those classes to the `ignore_io_grad_classes` argument. For example, the `BasicBlock` in the torchvision resnet implementation has an inplace residual connection. So we would do the following:


```python
model = torchvision.models.resnet18(pretrained=False)
watch(
    model,
    disable_inplace=True,
    ignore_io_grad_classes=[torchvision.models.resnet.BasicBlock]
)
```


**2. Weights and biases chart glitches**
```
"No data available." in the Custom Chart.
```

This occasionally shows up for me in the weights and biases interface and seems to be a difficult-to-reproduce bug in their custom charts support. Sometimes waiting fixes it. If possible, just restarting training when you notice this.

```
"Something went wrong..." and Google Chrome crashes.
```

It happens occasionally that the wandb website crashes with torchexplorer active. Reloading the page seems to always work.

**3. Graphviz overflow errors**
```
"Trapezoid overflow" error in the graphviz call.
```
This is a [known bug](https://github.com/ellson/MOTHBALLED-graphviz/issues/56) in Graphviz 2.42.2, an ancient version which is still the default on most package managers. If you're getting this error, you can fix it by installing a [newer release](https://gitlab.com/graphviz/graphviz/-/releases).
