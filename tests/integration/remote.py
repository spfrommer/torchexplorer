from __future__ import annotations

import os, sys

import wandb
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torchvision
import torch
from torch import nn
from torch import optim

from torchexplorer import watch, attach

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
        repeated, nn.ReLU(), repeated
    )

    wandb.init(**wandb_init_params, name='repeat_submodule_test')
    watch(model, log_freq=1, ignore_io_grad_classes=[], backend='wandb')
    infra.run_trial(model, X, y, steps=5)
    wandb.finish()


def test_repeat_submodule_multiforward():
    X = torch.randn(5, 10).to(infra.DEVICE)
    y = torch.randn(5, 10).to(infra.DEVICE)

    repeated = RepeatedSubmodule()

    model = nn.Sequential(
        repeated, nn.ReLU(), repeated
    ).to(infra.DEVICE)

    wandb.init(**wandb_init_params, name='repeat_submodule_multiforward_test')
    watch(
        model, log_freq=1, ignore_io_grad_classes=[], backend='wandb',
        delay_log_multi_backward=True
    )

    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    for _ in range(5):
        yhat1 = model(X)
        yhat2 = model(X + 5)
        yhat = yhat1 * 0.0 + yhat2
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model(X)

    # Expecting two bands on the gradients and two bands on the io values
    wandb.finish()


class EmbeddingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 20)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        return self.linear(self.embedding(x))

def test_embedding():
    X = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    y = torch.randn(2, 5, 20)

    model = EmbeddingModule()

    wandb.init(**wandb_init_params, name='embedding_test')
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
        # TODO: The commented out approach gives correct answers...
        # y_times = y1 * 5
        # y2, y3 = self.submodule(y1, y_times)
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


class AttachModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.submodule = MultiInputMultiOutputSubmoduleLongName()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        y1 = self.fc1(x)
        y1 = attach(y1, self, 'y1')
        y2, y3 = self.submodule(y1, y1 * 5)
        y2 = attach(y2, self, 'y2')
        y3 = attach(y3, self, 'y3')
        y4 = self.fc2(y2) + self.fc2(y3)
        y4 = attach(y4, self, 'y4')
        return y4


def test_attach():
    X = torch.randn(5, 10)
    y = torch.randn(5, 10)

    model = AttachModule()

    wandb.init(**wandb_init_params, name='attach_test')
    watch(model, log_freq=1, backend='wandb')
    infra.run_trial(model, X, y, steps=5)
    wandb.finish()
