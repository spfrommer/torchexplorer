import os, sys

import wandb
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torchvision
import torch
from torch import nn

from torchexplorer import watch

import infra


wandb_init_params = {
    'project': 'torchexplorer_tests_remote',
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

    wandb.init(**wandb_init_params, name='repeat_relu_nested')
    watch(model, log_freq=1, ignore_io_grad_classes=[], backend='wandb')
    infra.run_trial(model, X, y, steps=15)
    wandb.finish()


def test_resnet():
    X = torch.randn(5, 3, 32, 32)
    y = torch.randn(5, 1000)

    model = torchvision.models.resnet18()

    wandb.init(**wandb_init_params, name='resnet')
    watch(
        model,
        log_freq=1,
        ignore_io_grad_classes=[torchvision.models.resnet.BasicBlock],
        disable_inplace=True,
        backend='wandb'
    )
    infra.run_trial(model, X, y, steps=15)
    wandb.finish()



def test_transformer_encoder():
    X = torch.rand(10, 32, 512)
    y = torch.randn(10, 32, 512)

    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    model = nn.TransformerEncoder(encoder_layer, num_layers=6)

    wandb.init(**wandb_init_params, name='transformer')
    watch(model, log_freq=1, backend='wandb')
    infra.run_trial(model, X, y, steps=15)
    wandb.finish()
