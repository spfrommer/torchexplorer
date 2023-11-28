"""Processes the vega_raw.json copied from the vega editor into a vega spec ready for deployment."""

from __future__ import annotations

import pyperclip
import re


with open('vega_raw.json', 'r') as f:
    vega_source = f.read()

# vega_source = re.sub(r'"name": "wandb"((.|\n)*)]', '"name": "wandb"', vega_source)
vega_source = re.sub(r'"name": "wandb"(.|\n)*?      ]', '"name": "wandb"', vega_source)
# The starting panel ids
# vega_source = vega_source.replace('[21, 6, 12, 12, 12]', '[-1, -1, -1, -1, -1]')
vega_source = re.sub(r'panels_node_id_all",[\n\r\s]+"value":\s*\[.*?\]', 'panels_node_id_all", "value": [-1, -1, -1, -1, -1]', vega_source)

with open('vega_dataless.json', 'w') as f:
    f.write(vega_source)

pyperclip.copy(vega_source)

print('Copied to clipboard. Paste into the wandb editor and make a new version.')
