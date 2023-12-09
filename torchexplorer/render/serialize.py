from __future__ import annotations
from typing import Any, Iterable, Optional, Union
import numpy as np

from torchexplorer import utils
from torchexplorer.components.histogram import IncrementalHistogram

from torchexplorer.render.structs import (
    EdgeRenderable, TooltipRenderable, ModuleInvocationRenderable
)

def serialize_rows(renderable: ModuleInvocationRenderable) -> list[dict]:
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
            raw_hists: Optional[list[IncrementalHistogram]],
            grad_hists: Optional[list[IncrementalHistogram]],
            prefix: str,
            suffix: str
        ) -> str:

        raw_hists = [] if raw_hists is None else raw_hists
        grad_hists = [] if grad_hists is None else grad_hists

        raw_prefixes = [
            _serialize_list([f'{prefix} {i}', '']) for i in range(len(raw_hists))
        ]
        grad_prefixes = [
            _serialize_list([f'{prefix} {i}', suffix]) for i in range(len(grad_hists))
        ]

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
            raw_hists: Optional[dict[str, IncrementalHistogram]],
            grad_hists: Optional[dict[str, IncrementalHistogram]],
            suffix: str
        ) -> str:

        raw_hists = {} if raw_hists is None else raw_hists
        grad_hists = {} if grad_hists is None else grad_hists
        
        grad_hists = {_serialize_list([k, suffix]): v for k, v in grad_hists.items()}
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
    param_hists_str = interleave_and_serialize_dict(
        param_hists, param_grad_hists, 'grad'
    )

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

def _serialize_list(l: Iterable[Any]) -> str:
    return '::'.join([str(x) for x in l])

def _serialize_lists_nest2(l: Iterable[Iterable[Any]]) -> str:
    return ';'.join([_serialize_list(inner_l) for inner_l in l])

def _mid_join(l: list[str]) -> str:
    return '!!'.join(l)

def _top_join(l: list[str]) -> str:
    return '|'.join(l)
    
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