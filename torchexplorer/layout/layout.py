from __future__ import annotations
import copy

import json
import string
from typing import Any, Optional, Union
import numpy as np
import networkx as nx
from subprocess import Popen, PIPE

from dataclasses import dataclass, field

import wandb
from torchexplorer import utils
from torchexplorer.components.tooltip import Tooltip

from torchexplorer.core import (
    ModuleInvocationHistograms, ModuleInvocationStructure, ModuleSharedHistograms
)
from torchexplorer.structure.structure import is_input_node, is_output_node
from torchexplorer.components.histogram import IncrementalHistogram


@dataclass
class EdgeRenderable:
    path_points: list[list[float]]
    arrowhead_points: list[list[float]]
    downstream_input_index: Optional[int]
    upstream_output_index: Optional[int]


@dataclass
class TooltipRenderable:
    tooltip: Tooltip

    # Coordinates in parent of the renderable this tooltip belongs to
    bottom_left_corner: list[float] = field(default_factory=lambda: [0, 0]) 
    top_right_corner: list[float] = field(default_factory=lambda: [0, 0])


@dataclass
class ModuleInvocationRenderable:
    display_name: Optional[str] = None
    tooltip: Optional[TooltipRenderable] = None

    invocation_hists: Optional[ModuleInvocationHistograms] = None
    invocation_grad_hists: Optional[ModuleInvocationHistograms] = None
    shared_hists: Optional[ModuleSharedHistograms] = None

    # Coordinates in parent renderable
    bottom_left_corner: list[float] = field(default_factory=lambda: [0, 0]) 
    top_right_corner: list[float] = field(default_factory=lambda: [0, 0]) 

    # Inner graph data
    inner_graph_renderables: list['ModuleInvocationRenderable'] = (
        field(default_factory=lambda: [])
    )
    inner_graph_edges: list[EdgeRenderable] = field(default_factory=lambda: [])

    # Data added in the _process_graph function, after everything has been layed out
    # These ids are not related to the structure_id of the ModuleInvocationStructure
    id: Optional[int] = None
    parent_id: Optional[int] = None
    # Parent stack includes current renderable (this goes into the parents view in vega)
    parent_stack: Optional[list[tuple[str, int]]] = None
    child_ids: Optional[list[int]] = None


def layout(
        structure: ModuleInvocationStructure, cache: Optional[dict] = None
    ) -> tuple[ModuleInvocationRenderable, dict]:

    name = structure.module.__class__.__name__
    if is_input_node(name) or is_output_node(name):
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

    if not hasattr(structure, 'inner_graph'):
        return

    _preprocess_io_names(structure)
    json_data = _get_graphviz_json_with_caching(structure, cached_structure)


    for object in json_data['objects']:
        draw_points = np.array(object['_draw_'][1]['points'])
        draw_xs, draw_ys = draw_points[:, 0], draw_points[:, 1]

        inner_renderable = ModuleInvocationRenderable()
        inner_renderable.display_name = object['label']
        inner_renderable.bottom_left_corner = [draw_xs.min(), draw_ys.min()]
        inner_renderable.top_right_corner = [draw_xs.max(), draw_ys.max()]

        if object['tooltip'] != '{}':
            _add_tooltip(Tooltip.from_dict_string(object['tooltip']), inner_renderable)

        if not (is_input_node(object['label']) or is_output_node(object['label'])):
            memory_id = int(object['memory_id'])
            object_struct = structure.get_inner_structure_from_memory_id(memory_id)
            assert object_struct is not None

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

def _add_tooltip(tooltip: Tooltip, renderable: ModuleInvocationRenderable) -> None:
    tooltip_title_size, tooltip_font_size = 14, 11
    def _handle_string(str, truncate=False, title=False):
        font_size = tooltip_title_size if title else tooltip_font_size
        truncate_width = 70
        return _truncate_string_width(str, font_size, truncate_width, truncate)

    node_bottom_left = np.array(renderable.bottom_left_corner)
    node_top_right = np.array(renderable.top_right_corner)
    node_center_y = (node_bottom_left[1] + node_top_right[1]) / 2

    for i, key in enumerate(tooltip.keys):
        tooltip.keys[i] = _handle_string(key, True)[0]


    line_widths = [_handle_string(tooltip.title, False, True)[1]]
    for key, val in zip(tooltip.keys, tooltip.vals):
        line_widths.append(_handle_string(f'{key}{val}', False)[1])
    
    tooltip_width = max(line_widths) * 0.95 + 20
    tooltip_lines = 1 + len(tooltip.keys)
    tooltip_height = 20 + (tooltip_font_size + 2) * tooltip_lines

    tooltip_bottom_left = [node_top_right[0] + 20, node_center_y - tooltip_height / 2]
    tooltip_top_right = [
        tooltip_bottom_left[0] + tooltip_width, tooltip_bottom_left[1] + tooltip_height
    ]

    renderable.tooltip = TooltipRenderable(
        tooltip,
        bottom_left_corner=tooltip_bottom_left,
        top_right_corner=tooltip_top_right
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
    target_input_pos = [0, 0]  # Based on where vega spec expects input to be

    input_centers = []
    for r in renderable.inner_graph_renderables:
        if is_input_node(r.display_name):
            input_centers.append(_center(r))

    center = np.mean(np.array(input_centers), axis=0)
    trans = [target_input_pos[0] - center[0], target_input_pos[1] - center[1]]

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

def _preprocess_io_names(structure: ModuleInvocationStructure) -> None:
    multiple_inputs = structure.inner_graph.has_node('Input 1')
    multiple_outputs = structure.inner_graph.has_node('Output 1')

    if not multiple_inputs:
        new_attributes = {'Input 0': {'label': 'Input', 'tooltip': {}}}
        nx.set_node_attributes(structure.inner_graph, new_attributes)
    
    if not multiple_outputs:
        new_attributes = {'Output 0': {'label': 'Output', 'tooltip': {}}}
        nx.set_node_attributes(structure.inner_graph, new_attributes)

def _get_graphviz_json_with_caching(
        structure: ModuleInvocationStructure,
        cached_structure: Optional[ModuleInvocationStructure] = None
    ) -> dict:

    if cached_structure is not None:
        json_data = copy.deepcopy(cached_structure.graphviz_json_cache)
        assert json_data is not None
        for object in json_data['objects']:
            if (is_input_node(object['label']) or is_output_node(object['label'])):
                continue

            old_mem_id = int(object['memory_id'])
            old_struct = cached_structure.get_inner_structure_from_memory_id(old_mem_id)
            assert old_struct is not None
            new_struct = structure.get_inner_structure_from_structure_id(
                old_struct.structure_id
            )
            assert new_struct is not None
            object['memory_id'] = id(new_struct)
            object['cached_structure'] = old_struct
    else:
        json_data = _get_graphviz_json(structure)
        structure.graphviz_json_cache = json_data

        for object in json_data['objects']:
            object['cached_structure'] = None
    
    return json_data

def _get_graphviz_json(structure: ModuleInvocationStructure, format='json') -> dict:
    # Graphviz carries through the node attributes from structure.py to the JSON 
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

def _unconstrain_skip_connections(graph: nx.DiGraph) -> None:
    """A more aesthetic skip connection layout by unconstraining them in graphviz."""

    for edge in graph.edges:
        def avoid_edge_weight(u, v, d):
            return 1 if (u == edge[0] and v == edge[1]) else 0
        path = nx.shortest_path(graph, edge[0], edge[1], weight=avoid_edge_weight)
        if len(path) > 2:
            graph[edge[0]][edge[1]]['constraint'] = False


def wandb_table(
        renderable: ModuleInvocationRenderable
    ) -> tuple[wandb.Table, dict[str, str]]:

    rows = serialized_rows(renderable)
    fields = {key:key for key in rows[0]}
    keys = fields.keys() 
    data = [[row[key] for key in keys] for row in rows]
    table = wandb.Table(data=data, columns=list(keys))
    return table, fields

def serialized_rows(renderable: ModuleInvocationRenderable) -> list[dict]:
    serialized = _serialize_renderable(renderable)

    all_augmented_keys = []
    for item_type, items in serialized.items():
        for item in items:
            for key in item.keys():
                all_augmented_keys.append(f'{item_type}:{key}')

    rows = []
    for item_type, items in serialized.items():
        for item in items:
            new_row = {key: None for key in all_augmented_keys}
            for key, value in item.items():
                new_row[f'{item_type}:{key}'] = value
            new_row['type'] = item_type
            rows.append(new_row)

    return rows

def _serialize_renderable(renderable: ModuleInvocationRenderable) -> dict:
    nodes, edges = [], []

    def process_renderable(r: ModuleInvocationRenderable):
        new_node = _serialize_node(r)
        new_node['active_on_id'] = r.parent_id
        nodes.append(new_node)

        for edge in r.inner_graph_edges:
            new_edge = _serialize_edge(edge)
            new_edge['active_on_id'] = r.id
            edges.append(new_edge)

        for inner_r in r.inner_graph_renderables:
            process_renderable(inner_r)
    
    process_renderable(renderable)

    return {'nodes': nodes, 'edges': edges}

def _serialize_node(r: ModuleInvocationRenderable) -> dict:
    def tooltip_str(renderable: Optional[TooltipRenderable]) -> str:
        if renderable is None:
            return ''
        bl_corn, tr_corn = renderable.bottom_left_corner, renderable.top_right_corner
        title = renderable.tooltip.title
        keys, vals = renderable.tooltip.keys, renderable.tooltip.vals

        corners_str = _serialize_list(bl_corn + tr_corn)
        return _mid_join(
            [corners_str, title, _serialize_list(keys), _serialize_list(vals)]
        )

    def hist_strs(histogram: IncrementalHistogram) -> list[str]:
        history_bins, history_times = histogram.subsample_history()
        return [
            _serialize_list([_eformat(histogram.min), _eformat(histogram.max)]),
            _serialize_list([histogram.min, histogram.max]),
            histogram.params.time_unit,
            _serialize_list([_eformat(history_times[0]), _eformat(history_times[-1])]),
            _serialize_list(history_times),
            _serialize_lists_nest2(history_bins)
        ]

    def interleave_and_serialize_list(
            raw_hists: Optional[list], grad_hists: Optional[list],
            prefix: str, suffix: str
        ) -> str:

        raw_hists = [] if raw_hists is None else raw_hists
        grad_hists = [] if grad_hists is None else grad_hists

        raw_prefixes = [f'{prefix} {i}' for i in range(len(raw_hists))]
        grad_prefixes = [f'{prefix} {i} ({suffix})' for i in range(len(grad_hists))]

        if len(raw_hists) == len(grad_hists):
            joined_hists = utils.interleave(raw_hists, grad_hists)
            joined_prefixes = utils.interleave(raw_prefixes, grad_prefixes)
        else:
            assert len(raw_hists) == 0 or len(grad_hists) == 0
            joined_hists = raw_hists + grad_hists
            joined_prefixes = raw_prefixes + grad_prefixes

        return _top_join([
            _mid_join([joined_prefixes[i]] + hist_strs(h))
            for i, h in enumerate(joined_hists)
            if len(h.history_bins) > 0
        ])

    def interleave_and_serialize_dict(
            raw_hists: Optional[dict], grad_hists: Optional[dict]
        ) -> str:

        raw_hists = {} if raw_hists is None else raw_hists
        grad_hists = {} if grad_hists is None else grad_hists
        
        grad_hists = {f'{k} (grad)': v for k, v in grad_hists.items()}
        # The hist_dict_str sorts alphabetically which does the interleaving
        joined_hists = {**raw_hists, **grad_hists}

        return _top_join([
            _mid_join([k] + hist_strs(joined_hists[k]))
            for k in sorted(joined_hists.keys())
            if len(joined_hists[k].history_bins) > 0
        ])
    
    def renderable_resolve(attr1: str, attr2: str):
        if getattr(r, attr1) is not None:
            return getattr(getattr(r, attr1), attr2)
        return None


    input_hists = renderable_resolve('invocation_hists', 'input_hists')
    output_hists = renderable_resolve('invocation_hists', 'output_hists')
    input_grad_hists = renderable_resolve('invocation_grad_hists', 'input_hists')
    output_grad_hists = renderable_resolve('invocation_grad_hists', 'output_hists')
    param_hists = renderable_resolve('shared_hists', 'param_hists')
    param_grad_hists = renderable_resolve('shared_hists', 'param_grad_hists')
    

    input_hists_str = interleave_and_serialize_list(
        input_hists, input_grad_hists, 'input', 'grad norm'
    )
    output_hists_str = interleave_and_serialize_list(
        output_hists, output_grad_hists, 'output', 'grad norm'
    )
    param_hists_str = interleave_and_serialize_dict(param_hists, param_grad_hists)

    assert (r.child_ids is not None) and (r.parent_stack is not None)

    new_object = {
        'id': r.id,
        'child_ids': _serialize_list(r.child_ids),
        'parent_stack': _serialize_lists_nest2(r.parent_stack),
        'display_name': r.display_name,
        'tooltip': tooltip_str(r.tooltip),
        'input_histograms': input_hists_str,
        'output_histograms': output_hists_str,
        'param_histograms': param_hists_str,

        'bottom_left_corner_x': r.bottom_left_corner[0],
        'bottom_left_corner_y': r.bottom_left_corner[1],
        'top_right_corner_x': r.top_right_corner[0],
        'top_right_corner_y': r.top_right_corner[1],
    }

    return new_object

def _serialize_edge(edge: EdgeRenderable) -> dict:
    def interpolate_points(points: list[list[float]]):
        # Sometimes lines are very long, which get dissapeared if one end goes off
        # renderer. So we want to interpolate these long edges linearly
        max_dist = 20

        i = 0
        # Go to -2 to ignore the end of path marker
        while i < len(points) - 2:
            p1, p2 = points[i], points[i + 1]
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist > max_dist:
                direction = np.array(p2) - np.array(p1)
                new_point = np.array(p1) + direction * max_dist / dist
                points.insert(i + 1, new_point.tolist())
            i += 1
        
        return points

    def points_str(points: list[list[float]]) -> str:
        return _serialize_lists_nest2(interpolate_points(points))

    # Makes things easier in vega
    end_of_path = [[-10000.0, -10000.0]]
    return {
        'downstream_input_index': edge.downstream_input_index,
        'upstream_output_index': edge.upstream_output_index,
        'path_points': points_str(edge.path_points + end_of_path),
        'arrowhead_points': points_str(edge.arrowhead_points + end_of_path),
    }

def _serialize_list(l: list[Any]) -> str:
    return '::'.join([str(x) for x in l])

def _serialize_lists_nest2(l: list[list[Any]]) -> str:
    return ';'.join([_serialize_list(inner_l) for inner_l in l])

def _mid_join(l: list[str]) -> str:
    return '!!'.join(l)

def _top_join(l: list[str]) -> str:
    return '|'.join(l)

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
    
def _eformat(num, include_pm=True) -> str:
    # Adapted from https://stackoverflow.com/questions/9910972/number-of-digits-in-exponent
    prec, exp_digits = 1, 1

    if abs(num) < 100 and abs(num) > 1.0:
        string = str(num) if isinstance(num, int) else f'{num:.1f}'
    elif abs(num) < 100 and abs(num) > 0.1:
        string = f'{num:.2f}'
    else:
        s = '%.*e' % (prec, num)
        mantissa, exp = s.split('e')
        if include_pm:
            # add 1 to digits as 1 is taken by sign +/-
            string = '%se%+0*d' % (mantissa, exp_digits+1, int(exp))
        else:
            string = '%se%0*d' % (mantissa, exp_digits, int(exp))

    # Make minus signs longer to be more visible
    return string.replace('-', '–') 

def _center(r: Union[ModuleInvocationRenderable, TooltipRenderable]) -> list[float]:
    return [
        (r.bottom_left_corner[0] + r.top_right_corner[0]) / 2,
        (r.bottom_left_corner[1] + r.top_right_corner[1]) / 2
    ]