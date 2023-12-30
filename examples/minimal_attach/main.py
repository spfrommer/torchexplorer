import torch
from torch import nn
import torchexplorer

class AttachModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torchexplorer.attach(x, self, 'intermediate')
        return self.fc2(x)

model = AttachModule()
dummy_X = torch.randn(5, 10)

torchexplorer.watch(model, log_freq=1, backend='standalone')
model(dummy_X).sum().backward()

# Your model will be available at http://localhost:8080
