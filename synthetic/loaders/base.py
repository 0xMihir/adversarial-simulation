from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from synthetic.schema import WOMDRoadLineType


@dataclass
class LaneSegmentData:
    """
    Unified lane segment representation produced by both AV2 and WOMD loaders.
    All XY coordinates are in metric (metres) in the source dataset's local frame.
    Z is dropped — pipeline is 2D throughout (§9.7).
    """
    lane_id: str
    centerline_xy: np.ndarray      # (N, 2) float64
    left_boundary_xy: np.ndarray   # (M, 2) float64
    right_boundary_xy: np.ndarray  # (K, 2) float64
    successor_ids: list[str] = field(default_factory=list)
    predecessor_ids: list[str] = field(default_factory=list)
    left_neighbor_id: str | None = None
    right_neighbor_id: str | None = None
    lane_type: str = "surface"     # e.g. "surface", "bike", "bus", "parking"
    left_mark_type: str = "UNKNOWN"   # raw source string
    right_mark_type: str = "UNKNOWN"
    is_intersection: bool = False
    left_womd_type: WOMDRoadLineType = WOMDRoadLineType.TYPE_UNKNOWN
    right_womd_type: WOMDRoadLineType = WOMDRoadLineType.TYPE_UNKNOWN


@runtime_checkable
class ScenarioLoader(Protocol):
    def load_scenario(self, scenario_id: str) -> list[LaneSegmentData]: ...
    def list_scenario_ids(self) -> list[str]: ...
