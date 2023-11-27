from __future__ import annotations

from typing import Optional
import torch
from torch import Tensor
from torch import nn
from torch import optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_trial(
        model: nn.Module,
        X: Tensor, y: Tensor,
        steps: int = 20,
        pick_yhat: Optional[int] = None
    ):

    model = model.to(DEVICE)
    X = X.to(DEVICE)
    y = y.to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    for _ in range(steps):
        yhat = model(X)
        if pick_yhat is not None:
            yhat = yhat[pick_yhat]
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()