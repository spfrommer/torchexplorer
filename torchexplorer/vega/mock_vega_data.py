"""Copies a mock wandb data string to the clipboard for testing purposes."""

from __future__ import annotations

import json
import torch
from torch import optim
from torch import nn
import pyperclip

from torchexplorer import layout, api


class NonDiffSubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 20)

    def forward(self, input_1):
        x = self.fc1(input_1)
        return torch.round(x).long()


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.submodule = NonDiffSubModule()
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        skip = x
        x = self.submodule(x)
        x = self.fc2(x.float() + skip)
        # x = self.fc2(x.float())
        return x

def main():
    # model = TestModule()
    # time_log = ('epoch', lambda module, step: step // 2)
    # structure_wrapper = api.watch(model, log_freq=1, backend='none', time_log=time_log)
    # X, y = torch.randn(5, 10), torch.randn(5, 10)

    import torchvision
    model = torchvision.models.resnet18()
    inplace_classes = [torchvision.models.resnet.BasicBlock]
    structure_wrapper = api.watch(
        model, log_freq=1, backend='none',
        ignore_io_grad_classes=inplace_classes, disable_inplace=True
    )
    X, y = torch.randn(5, 3, 32, 32), torch.randn(5, 1000)


    # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    # model = nn.TransformerEncoder(encoder_layer, num_layers=6)
    # hook.hook(model)
    # X = torch.rand(10, 32, 512)
    # y = torch.randn(10, 32, 512)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    for step in range(2):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    renderable = layout.layout(structure_wrapper.structure)[0]
    rendered_layout = layout.serialized_rows(renderable)

    def custom_json(d: dict):
        string = json.dumps(d, cls=CompactJSONEncoder, indent=2)
        return string.replace('\n', '\n      ')

    pyperclip.copy(custom_json(rendered_layout))

    print('Copied to clipboard. Paste into the vega editor in the "wandb" data.')

# Adapted from:
# https://gist.github.com/jannismain/e96666ca4f059c3e5bc28abb711b5c92

class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 70
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 2
    """Maximum number of items in container that might be put on single line."""

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs["indent"] = 4
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            return self._encode_list(o)
        if isinstance(o, dict):
            return self._encode_object(o)
        if isinstance(o, float):  # Use scientific notation for floats
            return format(o, "g")
        return json.dumps(
            o,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
            separators=(self.item_separator, self.key_separator),
            default=self.default if hasattr(self, "default") else None,
        )

    def _encode_list(self, o):
        if self._put_on_single_line(o):
            return "[" + ", ".join(self.encode(el) for el in o) + "]"
        self.indentation_level += 1
        output = [self.indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

    def _encode_object(self, o):
        if not o:
            return "{}"
        if self._put_on_single_line(o):
            return (
                "{ "
                + ", ".join(
                    f"{self.encode(k)}: {self.encode(el)}" for k, el in o.items()
                )
                + " }"
            )
        self.indentation_level += 1
        output = [
            f"{self.indent_str}{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()
        ]

        self.indentation_level -= 1
        return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

    def _put_on_single_line(self, o):
        return (
            self._primitives_only(o)
            and len(o) <= self.MAX_ITEMS
            and len(str(o)) - 2 <= self.MAX_WIDTH
        )

    def _primitives_only(self, o: list | tuple | dict):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})"
            )

if __name__ == '__main__':
    main()
