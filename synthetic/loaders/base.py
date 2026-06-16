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
    centerline_xy: np.ndarray                  # (N, 2) float64
    left_boundary_feature_ids: list[str]       # IDs into MapFeatureData dict; AV2: single element, WOMD: one per BoundarySegment
    right_boundary_feature_ids: list[str]      # IDs into MapFeatureData dict
    successor_ids: list[str] = field(default_factory=list)
    predecessor_ids: list[str] = field(default_factory=list)
    left_neighbor_ids: list[str] = field(default_factory=list)   # AV2: at most one; WOMD: one per LaneNeighbor
    right_neighbor_ids: list[str] = field(default_factory=list)
    lane_type: str = "surface"     # e.g. "surface", "bike", "bus", "parking"
    is_intersection: bool = False


@dataclass
class MapFeatureData:
    """
    A full-length road boundary feature (RoadLine or RoadEdge in WOMD; merged
    boundary in AV2). Scene elements are created one-per-feature, not per-lane-slice.
    """
    feature_id: str
    polyline_xy: np.ndarray        # (N, 2) float64, complete feature polyline
    womd_type: WOMDRoadLineType
    is_road_edge: bool             # True → ROAD_EDGE/SHOULDER_LINE; False → lane marking


@runtime_checkable
class ScenarioLoader(Protocol):
    def load_scenario(self, scenario_id: str) -> tuple[list[LaneSegmentData], dict[str, MapFeatureData]]: ...
    def list_scenario_ids(self) -> list[str]: ...
