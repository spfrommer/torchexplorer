import wandb

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from sklearn import datasets

import click

from torchexplorer.api import watch, LIGHTNING_EPOCHS


wandb_init_params = {
    'project': 'torchexplorer_examples_irismlp',
    'name': 'iris_mlp',
    'dir': '/tmp/torchexplorer_examples_irismlp',
}

@click.command()
@click.option('--backend', default='wandb')
def main(backend):
    model = TestModule()
    watch(model, log_freq=15, time_log=LIGHTNING_EPOCHS, backend=backend)

    data = IrisDataModule()
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
        max_epochs=10,
        num_sanity_val_steps=0,
        devices=1 if torch.cuda.is_available() else 0,
        logger=logger,
    )


class TestModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

        self.train_outputs = []
        self.valid_outputs = []

        self.model = nn.Sequential(
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 3),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def forward(self, signal: Tensor) -> Tensor:
        return self.model(signal)


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
        return vars

    def on_validation_epoch_end(self) -> None:
        if len(self.train_outputs) == 0:
            # Sanity val steps are run before training starts
            self.valid_outputs.clear()
            return

        vars = {
            'train_loss': torch.stack([r['loss'] for r in self.train_outputs]).mean(),
            'valid_loss': torch.stack([r['loss'] for r in self.valid_outputs]).mean() 
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
            prefix += 'Loss'

            prefixed[f'{prefix}/{k}'] = v
        return prefixed


class IrisDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.loader_params = {
            'batch_size': 1,
            'num_workers': 4,
            'shuffle': True,
            'drop_last': True,
        }

    def setup(self, stage: str = None) -> None:
        iris = datasets.load_iris()

        X = torch.tensor(iris.data, dtype=torch.float32)
        y = torch.tensor(iris.target, dtype=torch.long)

        self.train_data, self.val_data, self.test_data = (
            self._split_tensors_into_datasets([X, y])
        )

    def _split_tensors_into_datasets(
            self, tensors: list[Tensor], train_proportion=0.7, val_proportion=0.2
        ) -> tuple[Dataset, Dataset, Dataset]:

        data_n = tensors[0].shape[0]
        train_size = int(train_proportion * data_n)
        val_size = int(val_proportion * data_n)

        train_tensors = [t[:train_size] for t in tensors]
        val_tensors = [t[train_size : train_size + val_size] for t in tensors]
        test_tensors = [t[train_size + val_size :] for t in tensors]

        return (
            TensorDataset(*train_tensors),
            TensorDataset(*val_tensors),
            TensorDataset(*test_tensors),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, **self.loader_params)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, **self.loader_params)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, **self.loader_params)
    

if __name__ == '__main__':
    main()