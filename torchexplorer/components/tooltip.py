from __future__ import annotations
import re
from torch.nn import Module, ModuleList, ModuleDict

from torchexplorer.core import SizeTracker, InvocationId

dash = '–'

class Tooltip:
    """The tooltip that pops up next to a Module."""

    def __init__(self, title: str, keys: list[str], vals: list[str]):
        self.title = title
        self.keys = keys
        self.vals = vals
    
    @classmethod
    def create_io(cls, tracker: SizeTracker) -> 'Tooltip':
        name = tracker.type.split('.')[-1]
        keys, vals = ['size'], [str(tracker.size).replace('None', dash)]
        return Tooltip(name, keys, vals)
    
    @classmethod
    def create_moduleinvocation(
            cls, module: Module, parent_module: Module, invocation_id: InvocationId
        ) -> 'Tooltip':

        name_in_parent = cls._get_name_in_parent(module, parent_module)

        io_shape_keys, io_shape_vals = cls._get_io_shape_keyvals(module, invocation_id)
        extra_repr_keys, extra_repr_vals = cls._get_extra_repr_keyvals(module)

        keys = io_shape_keys + extra_repr_keys
        vals = io_shape_vals + extra_repr_vals

        assert len(keys) == len(vals)

        return Tooltip(name_in_parent, keys, vals)
    
    @classmethod
    def create_attach(cls, module: Module) -> 'Tooltip':
        return cls.create_io(module.torchexplorer_metadata.input_sizes[0][0])
    
    @classmethod
    def _get_name_in_parent(cls, module: Module, parent_module: Module) -> str:
        name_in_parent = ''
        for name, m in parent_module.named_children():
            if m == module:
                name_in_parent = name
                break
                
            if isinstance(m, ModuleList):
                for i, mm in enumerate(m):
                    if mm == module:
                        name_in_parent = f'{name}[{i}]'
                        break
            
            if isinstance(m, ModuleDict):
                for k, mm in m.items():
                    if mm == module:
                        name_in_parent = f'{name}[{k}]'
                        break
        
        return name_in_parent

    @classmethod
    def _get_io_shape_keyvals(
            cls, module: Module, invocation_id: InvocationId
        ) -> tuple[list[str], list[str]]:

        metadata = module.torchexplorer_metadata 

        keys, vals = [], []

        one_input = len(metadata.input_sizes[invocation_id]) == 1
        for i, input_tracker in enumerate(metadata.input_sizes[invocation_id]):
            keys.append('in_size' if one_input else f'in{i}_size')
            vals.append(str(input_tracker.size).replace('None', dash))
        
        one_output = len(metadata.output_sizes[invocation_id]) == 1
        for i, output_tracker in enumerate(metadata.output_sizes[invocation_id]):
            keys.append('out_size' if one_output else f'out{i}_size')
            vals.append(str(output_tracker.size).replace('None', dash))

        return keys, vals
    
    @classmethod
    def _get_extra_repr_keyvals(cls, module: Module) -> tuple[list[str], list[str]]:
        try:
            keys, vals = [], []
            extra_rep = module.extra_repr()
            pairs = re.split(r',\s*(?![^()]*\))(?![^[]]*\])', extra_rep)
            for pair in pairs:
                if pair == '':
                    continue
                k, v = pair.split('=') if ('=' in pair) else (dash, pair)
                keys.append(k.strip())
                vals.append(v.strip())
        except Exception:
            keys, vals = [], []
        
        return keys, vals