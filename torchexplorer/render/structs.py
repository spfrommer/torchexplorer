from __future__ import annotations
from typing import Optional

from dataclasses import dataclass, field

from torchexplorer.components.tooltip import Tooltip

from torchexplorer.core import (
    ModuleInvocationHistograms, ModuleSharedHistograms
)


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


# Either a specific module invocation or for IO
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