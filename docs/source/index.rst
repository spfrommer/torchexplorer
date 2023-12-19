TorchExplorer
=========================================

`TorchExplorer <https://github.com/spfrommer/torchexplorer>`_ is an interfactive neural network visualizer for PyTorch.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

API
---

.. note::
   For wandb training, make sure to call `torchexplorer.setup()` before `wandb.init()`.
   This will configure subprocess open file limits to work around some wandb limitations.

.. autofunction:: torchexplorer.setup

.. autofunction:: torchexplorer.watch

.. autofunction:: torchexplorer.attach

.. autoclass:: torchexplorer.LIGHTNING_EPOCHS

.. autoclass:: torchexplorer.StructureWrapper
