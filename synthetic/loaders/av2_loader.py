"""
AV2 motion-forecasting scenario loader.

Directory layout expected:
    av2_root/
        {scenario_id}/
            map/
                log_map_archive_{scenario_id}.json
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from synthetic.schema import WOMDRoadLineType
from .base import LaneSegmentData, MapFeatureData
from av2.map.map_api import ArgoverseStaticMap

AV2_TO_WOMD: dict[str, WOMDRoadLineType] = {
    "SOLID_WHITE": WOMDRoadLineType.TYPE_SOLID_SINGLE_WHITE,
    "SOLID_YELLOW": WOMDRoadLineType.TYPE_SOLID_SINGLE_YELLOW,
    "DASHED_WHITE": WOMDRoadLineType.TYPE_BROKEN_SINGLE_WHITE,
    "DASHED_YELLOW": WOMDRoadLineType.TYPE_BROKEN_SINGLE_YELLOW,
    "DOUBLE_SOLID_YELLOW": WOMDRoadLineType.TYPE_SOLID_DOUBLE_YELLOW,
    "DOUBLE_SOLID_WHITE": WOMDRoadLineType.TYPE_SOLID_DOUBLE_WHITE,
}


def _map_marking(av2_type: str) -> WOMDRoadLineType:
    return AV2_TO_WOMD.get(av2_type, WOMDRoadLineType.TYPE_UNKNOWN)


class AV2ScenarioLoader:
    """
    Loads AV2 motion-forecasting map data via av2.map.map_api.ArgoverseStaticMap.
    Results are cached per-scenario (LRU, max 128 entries).
    """

    def __init__(self, av2_root: Path, max_cache_size: int = 128) -> None:
        self.av2_root = Path(av2_root)
        self._load_cached = lru_cache(maxsize=max_cache_size)(self._load_scenario_inner)

    def list_scenario_ids(self) -> list[str]:
        cache_path = self.av2_root / ".av2_scenario_ids.json"
        if cache_path.exists() and cache_path.stat().st_mtime >= self.av2_root.stat().st_mtime:
            return json.loads(cache_path.read_text())
        ids = [d.name for d in sorted(self.av2_root.iterdir()) if d.is_dir()]
        cache_path.write_text(json.dumps(ids))
        return ids

    def load_scenario(self, scenario_id: str) -> tuple[list[LaneSegmentData], dict[str, MapFeatureData]]:
        return self._load_cached(scenario_id)

    def _load_scenario_inner(self, scenario_id: str) -> tuple[list[LaneSegmentData], dict[str, MapFeatureData]]:
        map_dir = self.av2_root / scenario_id
        avm = ArgoverseStaticMap.from_map_dir(map_dir, build_raster=False)
        lane_segments = avm.get_scenario_lane_segments()

        segments: list[LaneSegmentData] = []
        map_features: dict[str, MapFeatureData] = {}

        for ls in lane_segments:
            # if ls.is_intersection:
            #     continue

            lane_id = str(ls.id)
            left_raw = ls.left_mark_type.value if hasattr(ls.left_mark_type, "value") else str(ls.left_mark_type)
            right_raw = ls.right_mark_type.value if hasattr(ls.right_mark_type, "value") else str(ls.right_mark_type)
            left_xy = np.asarray(ls.left_lane_boundary.xyz, dtype=np.float64)[:, :2]
            right_xy = np.asarray(ls.right_lane_boundary.xyz, dtype=np.float64)[:, :2]

            left_feat_id = f"{lane_id}_left"
            right_feat_id = f"{lane_id}_right"
            left_womd = _map_marking(left_raw)
            right_womd = _map_marking(right_raw)

            has_left_feature = left_xy.shape[0] >= 2# and not (left_raw == "NONE" and ls.is_intersection)
            has_right_feature = right_xy.shape[0] >= 2# and not (right_raw == "NONE" and ls.is_intersection)

            if has_left_feature:
                map_features[left_feat_id] = MapFeatureData(
                    feature_id=left_feat_id,
                    polyline_xy=left_xy,
                    womd_type=left_womd,
                    is_road_edge=left_womd == WOMDRoadLineType.TYPE_UNKNOWN,
                )

            if has_right_feature:
                map_features[right_feat_id] = MapFeatureData(
                    feature_id=right_feat_id,
                    polyline_xy=right_xy,
                    womd_type=right_womd,
                    is_road_edge=right_womd == WOMDRoadLineType.TYPE_UNKNOWN,
                )

            segments.append(LaneSegmentData(
                lane_id=lane_id,
                centerline_xy=np.asarray(avm.get_lane_segment_centerline(ls.id), dtype=np.float64)[:, :2],
                left_boundary_feature_ids=[left_feat_id] if has_left_feature else [],
                right_boundary_feature_ids=[right_feat_id] if has_right_feature else [],
                successor_ids=[str(s) for s in ls.successors],
                predecessor_ids=[str(p) for p in ls.predecessors],
                left_neighbor_ids=[str(ls.left_neighbor_id)] if ls.left_neighbor_id is not None else [],
                right_neighbor_ids=[str(ls.right_neighbor_id)] if ls.right_neighbor_id is not None else [],
                lane_type=ls.lane_type.value if hasattr(ls.lane_type, "value") else str(ls.lane_type),
                is_intersection=False,
            ))
        return segments, map_features

    @staticmethod
    def map_marking_type(av2_type: str) -> WOMDRoadLineType:
        return _map_marking(av2_type)
