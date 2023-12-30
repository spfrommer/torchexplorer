import torch
import torchvision
import torchexplorer

model = torchvision.models.resnet18(pretrained=False)
dummy_X = torch.randn(5, 3, 32, 32)

# Only log input/output and parameter histograms, if you don't want these set log=[].
torchexplorer.watch(model, log_freq=1, log=['io', 'params'], backend='standalone')

# To log also gradients, set log = ['io', 'io_grad', 'params', 'params_grad'] (default).
# This doesn't work with in-place operations (see "Common errors #1" in README.md).
# So we must disable in-place activations, and ignore modules with residual connections.
# Here we're using random data on an untrained model, so gradients aren't very useful.
# residual_class = torchvision.models.resnet.BasicBlock
# torchexplorer.watch(
#     model, log_freq=1, disable_inplace=True,
#     log=['io', 'io_grad', 'params', 'params_grad'],
#     ignore_io_grad_classes=[residual_class], backend='standalone'
# )

# Do one forwards and backwards pass
model(dummy_X).sum().backward()

# Your model will be available at http://localhost:8080