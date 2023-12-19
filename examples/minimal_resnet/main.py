import torch
import torchvision
import torchexplorer

model = torchvision.models.resnet18(pretrained=False)
dummy_X = torch.randn(5, 3, 32, 32)

# Only log input/output and parameter histograms, if you don't want even these set log=[].
torchexplorer.watch(model, log_freq=1, log=['io', 'params'], backend='standalone')

# Do one forwards and backwards pass
model(dummy_X).sum().backward()

# Your model will be available at http://localhost:5000
