from __future__ import annotations

import os, sys

import wandb
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torchvision
import torch
from torch import nn

from torchexplorer import watch

import infra
from vqvae import VQVAEModel


wandb_init_params = {
    'project': 'torchexplorer_tests_remote',
    # 'project': 'torchexplorer_demo',
    'dir': '/tmp/torchexplorer_tests_remote',
}
log_all = ['io', 'io_grad', 'params', 'params_grad'] 


def test_repeat_relu_nested():
    X = torch.randn(5, 10)
    y = torch.randn(5, 2)

    relu = nn.ReLU()

    submodule = nn.Sequential(
        relu,
        nn.Linear(20, 20),
        nn.Linear(20, 20),
        relu,
        nn.Linear(20, 20),
        relu,
        nn.Linear(20, 20),
        relu,
        nn.Linear(20, 20),
        relu,
        nn.Linear(20, 20),
        relu,
        nn.Linear(20, 20),
    )
    model = nn.Sequential(
        nn.Linear(10, 20),
        submodule,
        relu,
        nn.Linear(20, 2),
        relu
    )

    wandb.init(**wandb_init_params, name='repeat_relu_nested_test')
    watch(model, log_freq=1, ignore_io_grad_classes=[], backend='wandb')
    infra.run_trial(model, X, y, steps=5)
    wandb.finish()


class RepeatedSubmodule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc1(x)


def test_repeat_submodule():
    X = torch.randn(5, 10)
    y = torch.randn(5, 10)

    repeated = RepeatedSubmodule()

    model = nn.Sequential(
        repeated,
        nn.ReLU(),
        repeated
    )

    wandb.init(**wandb_init_params, name='repeat_submodule_test')
    watch(model, log_freq=1, ignore_io_grad_classes=[], backend='wandb')
    infra.run_trial(model, X, y, steps=5)
    wandb.finish()


def test_resnet():
    X = torch.randn(5, 3, 32, 32)
    y = torch.randn(5, 1000)

    model = torchvision.models.resnet18()

    wandb.init(**wandb_init_params, name='resnet_test')
    watch(
        model,
        log_freq=1,
        ignore_io_grad_classes=[torchvision.models.resnet.BasicBlock],
        disable_inplace=True,
        backend='wandb'
    )
    infra.run_trial(model, X, y, steps=5)
    wandb.finish()


def test_transformer_encoder():
    X = torch.rand(10, 32, 512)
    y = torch.randn(10, 32, 512)

    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    model = nn.TransformerEncoder(encoder_layer, num_layers=6)

    wandb.init(**wandb_init_params, name='transformer_encoder_test')
    watch(model, log_freq=1, backend='wandb')
    infra.run_trial(model, X, y, steps=5)
    wandb.finish()


def test_vqvae():
    X = torch.randn(5, 3, 32, 32)
    y = torch.randn(5, 3, 32, 32)

    model = VQVAEModel(
        num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
        num_embeddings=512, embedding_dim=64, commitment_cost=0.25
    )

    wandb.init(**wandb_init_params, name='vqvae_test')
    watch(model, log_freq=1, backend='wandb', disable_inplace=True)
    infra.run_trial(model, X, y, steps=5, pick_yhat=1)
    wandb.finish()


class MultiInputMultiOutputSubmoduleLongName(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x1, x2):
        y1 = self.fc1(x1)
        y2 = self.fc2(x2)
        return y1, y2


class MultiInputMultiOutputModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.submodule = MultiInputMultiOutputSubmoduleLongName()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        y1 = self.fc1(x)
        y2, y3 = self.submodule(y1, y1 * 5)
        y4 = self.fc2(y2) + self.fc2(y3)
        return y4


def test_mimo():
    X = torch.randn(5, 10)
    y = torch.randn(5, 10)

    model = MultiInputMultiOutputModule()

    wandb.init(**wandb_init_params, name='multi_io_test')
    watch(model, log_freq=1, backend='wandb')
    infra.run_trial(model, X, y, steps=5)
    wandb.finish()