from __future__ import annotations
import copy

import html
import json
import string
from typing import Optional, Union
import numpy as np
import networkx as nx
from subprocess import Popen, PIPE

from torchexplorer import utils
from torchexplorer import core
from torchexplorer.components.tooltip import Tooltip

from torchexplorer.core import ModuleInvocationHistograms, ModuleInvocationStructure
from torchexplorer.structure.structure import is_input_node, is_io_node

from torchexplorer.render.structs import (
    EdgeLayout, TooltipLayout, NodeLayout
)


def layout(
        structure: ModuleInvocationStructure, cache: Optional[dict] = None
    ) -> tuple[NodeLayout, dict]:

    name = structure.module.__class__.__name__
    if is_io_node(name):
        raise RuntimeError(f'Invalid module name: {name}')
    layout = NodeLayout(display_name=name)

    if cache is None:
        _layout_into(layout, structure, None)
        cache = {'cached_structure': structure}
    else:
        _layout_into(layout, structure, cache['cached_structure'])

    _process_graph(layout)
    return layout, cache

def _layout_into(
        layout: NodeLayout,
        structure: ModuleInvocationStructure,
        cached_structure: Optional[ModuleInvocationStructure] = None
    ):

    json_data = _get_graphviz_json_with_caching(structure, cached_structure)

    for object in json_data['objects']:
        draw_points = np.array(object['_draw_'][1]['points'])
        draw_xs, draw_ys = draw_points[:, 0], draw_points[:, 1]

        inner_layout = NodeLayout()
        # Replace the attach module label
        inner_layout.display_name = object['label'].replace('<>', ' ᴬ')
        inner_layout.bottom_left_corner = [draw_xs.min(), draw_ys.min()]
        inner_layout.top_right_corner = [draw_xs.max(), draw_ys.max()]

        if is_io_node(object['label']):
            _layout_io_node(inner_layout, structure, object)
        else:
            struct = _layout_moduleinvocation_node(inner_layout, structure, object)
            _layout_into(inner_layout, struct, object['cached_structure'])

        layout.inner_graph_layouts.append(inner_layout)
    
    if 'edges' in json_data:
        for edge in json_data['edges']:
            layout.inner_graph_edges.append(EdgeLayout(
                path_points=edge['_draw_'][-1]['points'],
                arrowhead_points=edge['_hdraw_'][-1]['points'],
                downstream_input_index=int(edge['downstream_input_index']),
                upstream_output_index=int(edge['upstream_output_index']),
            ))

    _translate_inner_layouts(layout)

def _layout_io_node(
        layout: NodeLayout, parent_structure: ModuleInvocationStructure, object: dict
    ) -> None:

    is_input = is_input_node(object['label'])
    parent_metadata = parent_structure.module_metadata()
    parent_invocation_id = parent_structure.invocation_id

    io_index = int(object['name'].split(' ')[-1])
    io_tensor_shape = (
        parent_metadata.input_sizes if is_input else parent_metadata.output_sizes
    )[parent_invocation_id][io_index]

    _add_tooltip(layout, Tooltip.create_io(io_tensor_shape))

    has_io_hists = parent_invocation_id in parent_metadata.invocation_hists
    if has_io_hists:
        hists = parent_metadata.invocation_hists[parent_invocation_id]
        hist = (hists.input_hists if is_input else hists.output_hists)[io_index]

    has_grad_hists = parent_invocation_id in parent_metadata.invocation_grad_hists
    if has_grad_hists:
        grad_hists = parent_metadata.invocation_grad_hists[parent_invocation_id]
        grad_hist = (
            grad_hists.input_hists if is_input else grad_hists.output_hists
        )[io_index]

    layout.invocation_hists = ModuleInvocationHistograms(
        input_hists=[hist] if has_io_hists else [],
        output_hists=[hist] if has_io_hists else []
    )

    layout.invocation_grad_hists = ModuleInvocationHistograms(
        input_hists=[grad_hist] if has_grad_hists else [],
        output_hists=[grad_hist] if has_grad_hists else []
    )

def _layout_moduleinvocation_node(
        layout: NodeLayout, parent_structure: ModuleInvocationStructure, object: dict
    ) -> ModuleInvocationStructure:

    structure_id = int(object['structure_id'])
    object_struct = parent_structure.get_inner_structure_from_id(structure_id)
    assert object_struct is not None

    if isinstance(object_struct.module, core.DummyAttachModule):
        _add_tooltip(layout, Tooltip.create_attach(object_struct.module))
    else:
        _add_tooltip(layout, Tooltip.create_moduleinvocation(
            object_struct.module, parent_structure.module, object_struct.invocation_id
        ))

    metadata = object_struct.module_metadata()

    if object_struct.invocation_id in metadata.invocation_hists:
        layout.invocation_hists = (
            metadata.invocation_hists[object_struct.invocation_id]
        )
    if object_struct.invocation_id in metadata.invocation_grad_hists:
        layout.invocation_grad_hists = (
            metadata.invocation_grad_hists[object_struct.invocation_id]
        )
    layout.shared_hists = metadata.shared_hists

    return object_struct


def _add_tooltip(layout: NodeLayout, tooltip: Tooltip) -> None:
    tooltip_title_size, tooltip_font_size = 14, 11
    def _handle_string(str, truncate=False, title=False):
        font_size = tooltip_title_size if title else tooltip_font_size
        truncate_width = 70
        return _truncate_string_width(str, font_size, truncate_width, truncate)

    for i, key in enumerate(tooltip.keys):
        tooltip.keys[i] = _handle_string(key, True)[0]

    line_widths = [_handle_string(tooltip.title, False, True)[1]]
    for key, val in zip(tooltip.keys, tooltip.vals):
        line_widths.append(_handle_string(f'{key}{val}', False)[1])
    
    tooltip_width = max(line_widths) * 0.95 + 20
    tooltip_lines = 1 + len(tooltip.keys)
    tooltip_height = 20 + (tooltip_font_size + 2) * tooltip_lines

    tooltip_bl = [
        layout.top_right_corner[0] + 20, _center(layout)[1] - tooltip_height / 2
    ]
    tooltip_tr = [tooltip_bl[0] + tooltip_width, tooltip_bl[1] + tooltip_height]

    layout.tooltip = TooltipLayout(
        tooltip, bottom_left_corner=tooltip_bl, top_right_corner=tooltip_tr
    ) 

def _process_graph(layout: NodeLayout):
    layout_id_counter = 0

    def process_graph_layout(
        l: NodeLayout, parent_id: int, parent_stack: list[tuple[str, int]]
    ) -> list[int]:

        nonlocal layout_id_counter
        new_id = layout_id_counter
        layout_id_counter += 1

        assert l.display_name is not None
        new_stack = parent_stack + [(l.display_name, new_id)]

        child_ids = []
        for inner_r in l.inner_graph_layouts:
            child_ids += process_graph_layout(inner_r, new_id, new_stack)
        
        l.id = new_id
        l.parent_id = parent_id
        l.parent_stack = new_stack
        l.child_ids = child_ids

        return [new_id] + child_ids

    process_graph_layout(layout, -1, [])

def _translate_inner_layouts(layout: NodeLayout) -> None:
    """Translate visual components to be centered around the input node."""
    target_input_pos = [0.0, 0.0]  # Based on where vega spec expects input to be

    input_centers = []
    for l in layout.inner_graph_layouts:
        if is_input_node(l.display_name):
            input_centers.append(_center(l))

    center = np.mean(np.array(input_centers), axis=0)
    trans = utils.list_add(target_input_pos, [-center[0], -center[1]])

    def apply_translation(l: Union[NodeLayout, TooltipLayout]):
        l.bottom_left_corner = utils.list_add(l.bottom_left_corner, trans)
        l.top_right_corner = utils.list_add(l.top_right_corner, trans)

    for l in layout.inner_graph_layouts:
        apply_translation(l)
        if l.tooltip is not None:
            apply_translation(l.tooltip)

    for e in layout.inner_graph_edges:
        e.path_points = [utils.list_add(p, trans) for p in e.path_points]
        e.arrowhead_points = [utils.list_add(p, trans) for p in e.arrowhead_points]

def _get_graphviz_json_with_caching(
        structure: ModuleInvocationStructure,
        cached_structure: Optional[ModuleInvocationStructure] = None
    ) -> dict:

    if cached_structure is not None:
        json_data = copy.deepcopy(cached_structure.graphviz_json_cache)
        assert json_data is not None
        for object in json_data['objects']:
            if is_io_node(object['label']):
                continue
            
            old_structure_id = int(object['structure_id'])
            old_struct = cached_structure.get_inner_structure_from_id(old_structure_id)
            assert old_struct is not None
            new_struct = structure.get_inner_structure_from_id(old_structure_id)
            assert new_struct is not None
            object['cached_structure'] = old_struct
    else:
        json_data = _get_graphviz_json(structure)
        structure.graphviz_json_cache = json_data

        for object in json_data['objects']:
            object['cached_structure'] = None
    
    return json_data

def _get_graphviz_json(structure: ModuleInvocationStructure, format='json') -> dict:
    _label_nodes(structure)
    _unconstrain_skip_connections(structure.inner_graph)

    A = nx.nx_agraph.to_agraph(structure.inner_graph)
    A.graph_attr.update(splines='ortho', ratio=1)
    A.node_attr.update(shape='box')
    
    dot_source = A.string()
    p = Popen(['dot', f'-T{format}'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    stdout_data, err = p.communicate(input=dot_source.encode())

    if len(err) > 0:
        raise RuntimeError(
            f'Error in dot subprocess:\n{err.decode()}\n'
            f'Dot source:\n{dot_source}\n'
            f'You can try installing the latest version of graphviz to fix.'
        )

    return json.loads(stdout_data)

def _label_nodes(structure: ModuleInvocationStructure) -> None:
    for node in structure.inner_graph.nodes:
        if is_io_node(node):
            label = node
        elif isinstance(node.module, core.DummyAttachModule):
            label = node.module.attach_name + '<>' # Placeholder for later
        else:
            label = node.module.__class__.__name__

        structure.inner_graph.nodes[node]['label'] = label

    multiple_inputs = structure.inner_graph.has_node('Input 1')
    multiple_outputs = structure.inner_graph.has_node('Output 1')

    if not multiple_inputs:
        new_attributes = {'Input 0': {'label': 'Input', 'tooltip': {}}}
        nx.set_node_attributes(structure.inner_graph, new_attributes)
    
    if not multiple_outputs:
        new_attributes = {'Output 0': {'label': 'Output', 'tooltip': {}}}
        nx.set_node_attributes(structure.inner_graph, new_attributes)

def _unconstrain_skip_connections(graph: nx.DiGraph) -> None:
    """A more aesthetic skip connection layout by unconstraining them in graphviz."""

    for edge in graph.edges:
        def avoid_edge_weight(u, v, _):
            return 1 if (u == edge[0] and v == edge[1]) else 0
        path = nx.shortest_path(graph, edge[0], edge[1], weight=avoid_edge_weight)
        if len(path) > 2:
            graph[edge[0]][edge[1]]['constraint'] = False


def _truncate_string_width(st, font_size, truncate_width, truncate):
    # Adapted from https://stackoverflow.com/questions/16007743/roughly-approximate-the-width-of-a-string-of-text-in-python
    size_to_pixels = (font_size / 12) * 16 * (6 / 1000.0) 
    truncate_width = truncate_width / size_to_pixels
    size = 0 # in milinches
    for i, s in enumerate(st):
        if s in 'lij|\' ': size += 37
        elif s in '![]fI.,:;/\\t': size += 50
        elif s in '`-(){}r"': size += 60
        elif s in '*^zcsJkvxy': size += 85
        elif s in 'aebdhnopqug#$L+<>=?_~FZT' + string.digits: size += 95
        elif s in 'BSPEAKVXY&UwNRCHD': size += 112
        elif s in 'QGOMm%W@–': size += 135
        else: size += 50

        if size >= truncate_width and truncate:
            return (st[:max(0, i - 1)] + '...'), size_to_pixels * (size + 150)

    return st, size_to_pixels * size


def _center(r: Union[NodeLayout, TooltipLayout]) -> list[float]:
    return [
        (r.bottom_left_corner[0] + r.top_right_corner[0]) / 2,
        (r.bottom_left_corner[1] + r.top_right_corner[1]) / 2
    ]
