from __future__ import annotations

import os, sys
import click

import wandb
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
from torch import Tensor
from torch import nn
import torchvision
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from torchexplorer import watch, LIGHTNING_EPOCHS

from data.cifar10_datamodule import CIFAR10DataModule
from data.metadata import get_normalize_layer


wandb_init_params = {
    'project': 'torchexplorer_examples_cifar10resnet',
    # 'project': 'torchexplorer_demo',
    'name': 'cifar10resnet',
    'dir': '/tmp/torchexplorer_examples_cifar10resnet',
}

@click.command()
@click.option('--backend', default='wandb')
def main(backend):
    model = TestModule()

    watch(
        model,
        log_freq=1,
        disable_inplace=True,
        ignore_io_grad_classes=[torchvision.models.resnet.BasicBlock],
        time_log=LIGHTNING_EPOCHS,
        backend=backend
    )

    data = CIFAR10DataModule(data_dir='/tmp/cifar10')
    logger = _setup_logging()
    trainer = _setup_trainer(logger)
    trainer.fit(model, data)

    wandb.finish()


def _setup_logging():
    return WandbLogger(
        project=wandb_init_params['project'],
        save_dir=wandb_init_params['dir'],
        name=wandb_init_params['name']
    )


def _setup_trainer(logger):
    return L.Trainer(
        max_epochs=50,
        num_sanity_val_steps=10,
        devices=1 if torch.cuda.is_available() else 0,
        logger=logger,
    )


class TestModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

        self.train_outputs = []
        self.valid_outputs = []

        self.normalize = get_normalize_layer('cifar10')
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 10)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def forward(self, signal: Tensor) -> Tensor:
        x = self.normalize(signal)
        return self.model(x)


    def training_step(self, batch, batch_idx: int) -> dict:
        variables = self._step_variables(batch, batch_idx)
        self.train_outputs.append(variables)
        return variables

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int) -> dict:
        variables = self._step_variables(batch, batch_idx)
        self.valid_outputs.append(variables)
        return variables


    def _step_variables(self, batch, batch_idx: int) -> dict:
        signal, target = batch[0], batch[1]
        pred = self(signal)

        vars = {}
        vars['target'] = target
        vars['pred'] = pred.detach()
        vars['loss'] = self.loss(pred, target)
        vars['acc'] = (pred.argmax(dim=1) == target).float().mean().detach()
        return vars

    def on_validation_epoch_end(self) -> None:
        if len(self.train_outputs) == 0:
            # Sanity val steps are run before training starts
            self.valid_outputs.clear()
            return

        vars = {
            'train_loss': torch.stack([r['loss'] for r in self.train_outputs]).mean(),
            'valid_loss': torch.stack([r['loss'] for r in self.valid_outputs]).mean(),
            'train_acc': torch.stack([r['acc'] for r in self.train_outputs]).mean(),
            'valid_acc': torch.stack([r['acc'] for r in self.valid_outputs]).mean()
        }

        self.log('monitored', vars['valid_loss'])

        # Prefix the metrics so that they are grouped appropriately in wandb
        vars = self._prefix_metrics(vars)

        self.logger.experiment.log(vars)
        self.train_outputs.clear()
        self.valid_outputs.clear()

    def _prefix_metrics(self, metrics: dict) -> dict:
        prefixed = {}
        for (k, v) in metrics.items():
            prefix = '1) '
            prefix += 'Train' if 'train' in k else 'Valid'
            prefix += 'Loss' if 'loss' in k else 'Accuracy'

            prefixed[f'{prefix}/{k}'] = v
        return prefixed


if __name__ == '__main__':
    main()
