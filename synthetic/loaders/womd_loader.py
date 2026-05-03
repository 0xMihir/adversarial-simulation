"""
WOMD (Waymo Open Motion Dataset) scenario loader.

Reads TFRecord files using a pure-Python TFRecord iterator (no TensorFlow required)
and parses Scenario protobufs compiled from synthetic/loaders/proto/scenario.proto.

To compile the proto stubs, run from the project root:
    bash scripts/compile_protos.sh

Directory layout expected:
    womd_root/
        training/
            *.tfrecord
        (or any flat layout of *.tfrecord files)
"""
from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

from synthetic.schema import WOMDRoadLineType
from .base import LaneSegmentData

WOMD_ROAD_LINE_TO_WOMD: dict[int, WOMDRoadLineType] = {
    0: WOMDRoadLineType.TYPE_UNKNOWN,
    1: WOMDRoadLineType.TYPE_BROKEN_SINGLE_WHITE,
    2: WOMDRoadLineType.TYPE_SOLID_SINGLE_WHITE,
    3: WOMDRoadLineType.TYPE_SOLID_DOUBLE_WHITE,
    4: WOMDRoadLineType.TYPE_BROKEN_SINGLE_YELLOW,
    5: WOMDRoadLineType.TYPE_BROKEN_SINGLE_YELLOW,  # broken double → broken single
    6: WOMDRoadLineType.TYPE_SOLID_SINGLE_YELLOW,
    7: WOMDRoadLineType.TYPE_SOLID_DOUBLE_YELLOW,
    8: WOMDRoadLineType.TYPE_SOLID_DOUBLE_YELLOW,   # passing double → solid double
}

WOMD_LANE_TYPE_TO_STR: dict[int, str] = {
    0: "undefined",
    1: "freeway",
    2: "surface",
    3: "bike",
}


def _map_road_line_type(womd_type_int: int) -> WOMDRoadLineType:
    return WOMD_ROAD_LINE_TO_WOMD.get(womd_type_int, WOMDRoadLineType.TYPE_UNKNOWN)


def _masked_crc32c(data: bytes) -> int:
    try:
        import crc32c
        crc = crc32c.crc32c(data)
    except ImportError:
        # Fallback: use binascii crc32 (not crc32c, but acceptable for dev)
        import binascii
        crc = binascii.crc32(data) & 0xFFFFFFFF
    # Waymo masking: rotate right by 15, add constant
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xFFFFFFFF


def _iter_tfrecords(path: Path) -> Iterator[bytes]:
    """
    Yield raw serialized protobuf bytes from a TFRecord file.
    TFRecord format per record:
        uint64  length
        uint32  masked_crc32c(length as little-endian uint64)
        byte[length]  data
        uint32  masked_crc32c(data)
    """
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                break
            if len(header) < 12:
                raise ValueError(f"Truncated TFRecord header in {path}")
            length = struct.unpack("<Q", header[:8])[0]
            data = f.read(length)
            if len(data) < length:
                raise ValueError(f"Truncated TFRecord data in {path}")
            f.read(4)  # skip data CRC
            yield data


def _polyline_to_xy(points) -> np.ndarray:
    """Convert repeated MapPoint proto messages → (N, 2) float64 array."""
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([[p.x, p.y] for p in points], dtype=np.float64)


class WOMDScenarioLoader:
    """
    Loads WOMD map data from TFRecord files without TensorFlow.
    Results are cached per-scenario (LRU, max 128 entries).

    Usage:
        loader = WOMDScenarioLoader(Path("data/womd/training/"))
        segments = loader.load_scenario("scenario_id_string")
    """

    def __init__(self, womd_root: Path, max_cache_size: int = 128) -> None:
        self.womd_root = Path(womd_root)
        self._scenario_index: dict[str, Path] | None = None
        self._load_cached = lru_cache(maxsize=max_cache_size)(self._load_scenario_inner)

    def _build_index(self) -> dict[str, Path]:
        """Scan all .tfrecord files and map scenario_id → file path."""
        try:
            from synthetic.loaders.proto import scenario_pb2
        except ImportError:
            raise ImportError(
                "Proto stubs not compiled. Run: bash scripts/compile_protos.sh"
            )

        index: dict[str, Path] = {}
        for tfr in sorted(self.womd_root.rglob("*.tfrecord-*")):
            for raw in _iter_tfrecords(tfr):
                sc = scenario_pb2.Scenario()
                sc.ParseFromString(raw)
                index[sc.scenario_id] = tfr
        return index

    def list_scenario_ids(self) -> list[str]:
        if self._scenario_index is None:
            self._scenario_index = self._build_index()
        return list(self._scenario_index.keys())

    def load_scenario(self, scenario_id: str) -> list[LaneSegmentData]:
        return self._load_cached(scenario_id)

    def _load_scenario_inner(self, scenario_id: str) -> list[LaneSegmentData]:
        try:
            from synthetic.loaders.proto import scenario_pb2
        except ImportError:
            raise ImportError(
                "Proto stubs not compiled. Run: bash scripts/compile_protos.sh"
            )

        if self._scenario_index is None:
            self._scenario_index = self._build_index()

        tfr_path = self._scenario_index.get(scenario_id)
        if tfr_path is None:
            raise KeyError(f"Scenario {scenario_id!r} not found in {self.womd_root}")

        # Find the matching Scenario in the TFRecord
        scenario = None
        for raw in _iter_tfrecords(tfr_path):
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(raw)
            if sc.scenario_id == scenario_id:
                scenario = sc
                break
        if scenario is None:
            raise RuntimeError(f"Scenario {scenario_id!r} not found in {tfr_path}")

        # Build feature_id → MapFeature lookup
        feature_map: dict[int, object] = {mf.id: mf for mf in scenario.map_features}

        result: list[LaneSegmentData] = []
        for mf in scenario.map_features:
            if not mf.HasField("lane"):
                continue
            lane = mf.lane

            # Skip intersection-interior lanes (§0.2: surface lanes only in v1)
            lane_type_str = WOMD_LANE_TYPE_TO_STR.get(lane.type, "undefined")
            if lane_type_str not in {"surface", "bike"}:
                continue

            centerline_xy = _polyline_to_xy(lane.polyline)

            # Resolve left boundary: stitch all left_boundary BoundarySegments
            left_xy = self._resolve_boundaries(lane.left_boundaries, feature_map)
            right_xy = self._resolve_boundaries(lane.right_boundaries, feature_map)

            # Infer mark types from first boundary segment type
            left_womd = self._boundary_segment_to_womd(lane.left_boundaries, feature_map)
            right_womd = self._boundary_segment_to_womd(lane.right_boundaries, feature_map)
            left_raw = left_womd.value
            right_raw = right_womd.value

            result.append(LaneSegmentData(
                lane_id=str(mf.id),
                centerline_xy=centerline_xy,
                left_boundary_xy=left_xy,
                right_boundary_xy=right_xy,
                successor_ids=[str(i) for i in lane.exit_lanes],
                predecessor_ids=[str(i) for i in lane.entry_lanes],
                left_neighbor_id=str(lane.left_neighbors[0].feature_id) if lane.left_neighbors else None,
                right_neighbor_id=str(lane.right_neighbors[0].feature_id) if lane.right_neighbors else None,
                lane_type=lane_type_str,
                left_mark_type=left_raw,
                right_mark_type=right_raw,
                is_intersection=False,
                left_womd_type=left_womd,
                right_womd_type=right_womd,
            ))
        return result

    @staticmethod
    def _resolve_boundaries(boundary_segments, feature_map: dict[int, object]) -> np.ndarray:
        """
        Resolve a lane's repeated BoundarySegment list to a stitched (N, 2) polyline.
        Each BoundarySegment references a RoadLine or RoadEdge MapFeature by ID.
        For v1 we concatenate the full referenced feature polyline (no index clipping).
        """
        if not boundary_segments:
            return np.zeros((0, 2), dtype=np.float64)

        parts: list[np.ndarray] = []
        for bs in boundary_segments:
            ref = feature_map.get(bs.boundary_feature_id)
            if ref is None:
                continue
            if ref.HasField("road_line"):
                pts = _polyline_to_xy(ref.road_line.polyline)
            elif ref.HasField("road_edge"):
                pts = _polyline_to_xy(ref.road_edge.polyline)
            else:
                continue
            if pts.shape[0] > 0:
                parts.append(pts)

        if not parts:
            return np.zeros((0, 2), dtype=np.float64)
        return np.concatenate(parts, axis=0)

    @staticmethod
    def _boundary_segment_to_womd(boundary_segments, feature_map: dict[int, object]) -> WOMDRoadLineType:
        """Infer WOMD type from the first resolvable boundary segment."""
        for bs in boundary_segments:
            ref = feature_map.get(bs.boundary_feature_id)
            if ref is None:
                continue
            if ref.HasField("road_line"):
                return _map_road_line_type(ref.road_line.type)
            if ref.HasField("road_edge"):
                return WOMDRoadLineType.TYPE_SOLID_SINGLE_WHITE  # treat road edge as solid white
        return WOMDRoadLineType.TYPE_UNKNOWN
