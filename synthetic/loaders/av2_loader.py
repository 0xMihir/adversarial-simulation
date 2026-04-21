"""
AV2 motion-forecasting scenario loader.

Directory layout expected:
    av2_root/
        {scenario_id}/
            map/
                log_map_archive_{scenario_id}.json
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from synthetic.schema import WOMDRoadLineType
from .base import LaneSegmentData

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
    Loads AV2 motion-forecasting map data via av2.map.av2_map_api.ArgoverseStaticMap.
    Results are cached per-scenario (LRU, max 128 entries).
    """

    def __init__(self, av2_root: Path, max_cache_size: int = 128) -> None:
        self.av2_root = Path(av2_root)
        self._load_cached = lru_cache(maxsize=max_cache_size)(self._load_scenario_inner)

    def list_scenario_ids(self) -> list[str]:
        return [d.name for d in sorted(self.av2_root.iterdir()) if d.is_dir()]

    def load_scenario(self, scenario_id: str) -> list[LaneSegmentData]:
        return self._load_cached(scenario_id)

    def _load_scenario_inner(self, scenario_id: str) -> list[LaneSegmentData]:
        try:
            from av2.map.av2_map_api import ArgoverseStaticMap
        except ImportError as e:
            raise ImportError("av2 package required: uv add av2") from e

        map_dir = self.av2_root / scenario_id / "map"
        avm = ArgoverseStaticMap.from_map_dir(map_dir, build_raster=False)
        lane_segments = avm.get_scenario_lane_segments()

        result: list[LaneSegmentData] = []
        for ls in lane_segments:
            if ls.is_intersection:
                continue  # surface lanes only (§0.2)

            left_raw = ls.left_mark_type.value if hasattr(ls.left_mark_type, "value") else str(ls.left_mark_type)
            right_raw = ls.right_mark_type.value if hasattr(ls.right_mark_type, "value") else str(ls.right_mark_type)

            result.append(LaneSegmentData(
                lane_id=str(ls.id),
                centerline_xy=np.asarray(ls.centerline, dtype=np.float64)[:, :2],
                left_boundary_xy=np.asarray(ls.left_lane_boundary.xyz, dtype=np.float64)[:, :2],
                right_boundary_xy=np.asarray(ls.right_lane_boundary.xyz, dtype=np.float64)[:, :2],
                successor_ids=[str(s) for s in ls.successors],
                predecessor_ids=[str(p) for p in ls.predecessors],
                left_neighbor_id=str(ls.left_neighbor_id) if ls.left_neighbor_id is not None else None,
                right_neighbor_id=str(ls.right_neighbor_id) if ls.right_neighbor_id is not None else None,
                lane_type=ls.lane_type.value if hasattr(ls.lane_type, "value") else str(ls.lane_type),
                left_mark_type=left_raw,
                right_mark_type=right_raw,
                is_intersection=False,
                left_womd_type=_map_marking(left_raw),
                right_womd_type=_map_marking(right_raw),
            ))
        return result

    @staticmethod
    def map_marking_type(av2_type: str) -> WOMDRoadLineType:
        return _map_marking(av2_type)
