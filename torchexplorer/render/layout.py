from __future__ import annotations
import copy

import json
import string
from typing import Optional, Union
import numpy as np
import networkx as nx
from subprocess import Popen, PIPE

from torchexplorer import utils
from torchexplorer.components.tooltip import Tooltip

from torchexplorer.core import ModuleInvocationHistograms, ModuleInvocationStructure
from torchexplorer.structure.structure import is_input_node, is_io_node

from torchexplorer.render.structs import (
    EdgeRenderable, TooltipRenderable, ModuleInvocationRenderable
)


def layout(
        structure: ModuleInvocationStructure, cache: Optional[dict] = None
    ) -> tuple[ModuleInvocationRenderable, dict]:

    name = structure.module.__class__.__name__
    if is_io_node(name):
        raise RuntimeError(f'Invalid module name: {name}')
    renderable = ModuleInvocationRenderable(display_name=name)

    if cache is None:
        _layout_into(renderable, structure, None)
        cache = {'cached_structure': structure}
    else:
        _layout_into(renderable, structure, cache['cached_structure'])

    _process_graph(renderable)
    return renderable, cache

def _layout_into(
        renderable: ModuleInvocationRenderable,
        structure: ModuleInvocationStructure,
        cached_structure: Optional[ModuleInvocationStructure] = None
    ):

    json_data = _get_graphviz_json_with_caching(structure, cached_structure)

    for object in json_data['objects']:
        draw_points = np.array(object['_draw_'][1]['points'])
        draw_xs, draw_ys = draw_points[:, 0], draw_points[:, 1]

        inner_renderable = ModuleInvocationRenderable()
        inner_renderable.display_name = object['label']
        inner_renderable.bottom_left_corner = [draw_xs.min(), draw_ys.min()]
        inner_renderable.top_right_corner = [draw_xs.max(), draw_ys.max()]

        if is_io_node(object['label']):
            is_input = is_input_node(object['label'])
            metadata = structure.module_metadata()

            io_index = int(object['name'].split(' ')[-1])
            io_tensor_shape = (
                metadata.input_sizes if is_input else metadata.output_sizes
            )[structure.invocation_id][io_index]

            _add_tooltip(inner_renderable, Tooltip.create_io(
                io_tensor_shape, is_input
            ))

            hists = metadata.invocation_hists[structure.invocation_id]
            grad_hists = metadata.invocation_grad_hists[structure.invocation_id]

            hist = (hists.input_hists if is_input else hists.output_hists)[io_index]
            grad_hist = (
                grad_hists.input_hists if is_input else grad_hists.output_hists
            )[io_index]

            inner_renderable.invocation_hists = ModuleInvocationHistograms(
                input_hists=[hist] if is_input else [],
                output_hists=[hist] if not is_input else []
            )

            inner_renderable.invocation_grad_hists = ModuleInvocationHistograms(
                input_hists=[grad_hist] if is_input else [],
                output_hists=[grad_hist] if not is_input else []
            )
        else:
            structure_id = int(object['structure_id'])
            object_struct = structure.get_inner_structure_from_id(structure_id)
            assert object_struct is not None

            _add_tooltip(inner_renderable, Tooltip.create_moduleinvocation(
                object_struct.module, structure.module, object_struct.invocation_id
            ))

            metadata = object_struct.module_metadata()

            if object_struct.invocation_id in metadata.invocation_hists:
                inner_renderable.invocation_hists = (
                    metadata.invocation_hists[object_struct.invocation_id]
                )
            if object_struct.invocation_id in metadata.invocation_grad_hists:
                inner_renderable.invocation_grad_hists = (
                    metadata.invocation_grad_hists[object_struct.invocation_id]
                )
            inner_renderable.shared_hists = metadata.shared_hists

            _layout_into(inner_renderable, object_struct, object['cached_structure'])

        renderable.inner_graph_renderables.append(inner_renderable)
    
    if 'edges' in json_data:
        for edge in json_data['edges']:
            renderable.inner_graph_edges.append(EdgeRenderable(
                path_points=edge['_draw_'][-1]['points'],
                arrowhead_points=edge['_hdraw_'][-1]['points'],
                downstream_input_index=int(edge['downstream_input_index']),
                upstream_output_index=int(edge['upstream_output_index']),
            ))

    _translate_inner_renderables(renderable)

def _add_tooltip(renderable: ModuleInvocationRenderable, tooltip: Tooltip) -> None:
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
        renderable.top_right_corner[0] + 20, _center(renderable)[1] - tooltip_height / 2
    ]
    tooltip_tr = [tooltip_bl[0] + tooltip_width, tooltip_bl[1] + tooltip_height]

    renderable.tooltip = TooltipRenderable(
        tooltip, bottom_left_corner=tooltip_bl, top_right_corner=tooltip_tr
    ) 

def _process_graph(renderable: ModuleInvocationRenderable):
    renderable_id_counter = 0

    def process_graph_renderable(
        r: ModuleInvocationRenderable,
        parent_id: int,
        parent_stack: list[tuple[str, int]]
    ) -> list[int]:

        nonlocal renderable_id_counter
        new_id = renderable_id_counter
        renderable_id_counter += 1

        assert r.display_name is not None
        new_stack = parent_stack + [(r.display_name, new_id)]

        child_ids = []
        for inner_r in r.inner_graph_renderables:
            child_ids += process_graph_renderable(inner_r, new_id, new_stack)
        
        r.id = new_id
        r.parent_id = parent_id
        r.parent_stack = new_stack
        r.child_ids = child_ids

        return [new_id] + child_ids

    process_graph_renderable(renderable, -1, [])

def _translate_inner_renderables(renderable: ModuleInvocationRenderable) -> None:
    """Translate visual components to be centered around the input node."""
    target_input_pos = [0.0, 0.0]  # Based on where vega spec expects input to be

    input_centers = []
    for r in renderable.inner_graph_renderables:
        if is_input_node(r.display_name):
            input_centers.append(_center(r))

    center = np.mean(np.array(input_centers), axis=0)
    trans = utils.list_add(target_input_pos, [-center[0], -center[1]])

    def apply_translation(r):
        r.bottom_left_corner = utils.list_add(r.bottom_left_corner, trans)
        r.top_right_corner = utils.list_add(r.top_right_corner, trans)

    for r in renderable.inner_graph_renderables:
        apply_translation(r)
        if r.tooltip is not None:
            apply_translation(r.tooltip)

    for e in renderable.inner_graph_edges:
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
    # Graphviz carries through the node attributes from structure.py to the JSON 
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
        structure.inner_graph.nodes[node]['label'] = (
            node if is_io_node(node) else node.module.__class__.__name__
        )

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
        elif s in 'QGOMm%W@': size += 135
        else: size += 50

        if size >= truncate_width and truncate:
            return (st[:max(0, i - 1)] + '...'), size_to_pixels * (size + 150)

    return st, size_to_pixels * size


def _center(r: Union[ModuleInvocationRenderable, TooltipRenderable]) -> list[float]:
    return [
        (r.bottom_left_corner[0] + r.top_right_corner[0]) / 2,
        (r.bottom_left_corner[1] + r.top_right_corner[1]) / 2
    ]
