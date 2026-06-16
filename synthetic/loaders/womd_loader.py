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

import json
import struct
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

from synthetic.schema import WOMDRoadLineType
from .base import LaneSegmentData, MapFeatureData

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


def _iter_tfrecords_with_offsets(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield (byte_offset_of_record_start, data) for each record in a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            offset = f.tell()
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
            yield offset, data


def _read_tfrecord_at(path: Path, offset: int) -> bytes:
    """Read a single TFRecord at a known byte offset."""
    with open(path, "rb") as f:
        f.seek(offset)
        header = f.read(12)
        if len(header) < 12:
            raise ValueError(f"Truncated TFRecord header in {path} at offset {offset}")
        length = struct.unpack("<Q", header[:8])[0]
        data = f.read(length)
        if len(data) < length:
            raise ValueError(f"Truncated TFRecord data in {path} at offset {offset}")
    return data


def _polyline_to_xy(points) -> np.ndarray:
    """Convert repeated MapPoint proto messages → (N, 2) float64 array."""
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([[p.x, p.y] for p in points], dtype=np.float64)


def _unique_ordered(ids: list[str]) -> list[str]:
    """Deduplicate a list while preserving insertion order."""
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class WOMDScenarioLoader:
    """
    Loads WOMD map data from TFRecord files without TensorFlow.
    Results are cached per-scenario (LRU, max 128 entries).

    Usage:
        loader = WOMDScenarioLoader(Path("data/womd/training/"))
        segments, map_features = loader.load_scenario("scenario_id_string")
    """

    def __init__(self, womd_root: Path, max_cache_size: int = 128) -> None:
        self.womd_root = Path(womd_root)
        self._scenario_index: dict[str, tuple[Path, int]] | None = None
        self._load_cached = lru_cache(maxsize=max_cache_size)(self._load_scenario_inner)

    @property
    def _index_cache_path(self) -> Path:
        root = self.womd_root if self.womd_root.is_dir() else self.womd_root.parent
        return root / ".womd_scenario_index.json"

    def _index_is_stale(self, cache_path: Path) -> bool:
        """Return True if any tfrecord is newer than the cache file."""
        cache_mtime = cache_path.stat().st_mtime
        if self.womd_root.is_file():
            return self.womd_root.stat().st_mtime > cache_mtime
        return any(
            tfr.stat().st_mtime > cache_mtime
            for tfr in self.womd_root.rglob("*.tfrecord-*")
        )

    def _load_index_from_cache(self, cache_path: Path) -> dict[str, tuple[Path, int]]:
        raw = json.loads(cache_path.read_text())
        return {sid: (Path(path), offset) for sid, (path, offset) in raw.items()}

    def _save_index_to_cache(self, index: dict[str, tuple[Path, int]], cache_path: Path) -> None:
        serializable = {sid: [str(path), offset] for sid, (path, offset) in index.items()}
        cache_path.write_text(json.dumps(serializable))

    def _build_index(self) -> dict[str, tuple[Path, int]]:
        """Scan all .tfrecord files and map scenario_id → (file path, byte offset)."""
        try:
            from synthetic.loaders.proto import scenario_pb2
        except ImportError:
            raise ImportError(
                "Proto stubs not compiled. Run: bash scripts/compile_protos.sh"
            )

        cache_path = self._index_cache_path
        if cache_path.exists() and not self._index_is_stale(cache_path):
            return self._load_index_from_cache(cache_path)

        index: dict[str, tuple[Path, int]] = {}
        if self.womd_root.is_file() and self.womd_root.suffix.startswith(".tfrecord"):
            for offset, raw in _iter_tfrecords_with_offsets(self.womd_root):
                sc = scenario_pb2.Scenario()
                sc.ParseFromString(raw)
                index[sc.scenario_id] = (self.womd_root, offset)
        else:
            for tfr in sorted(self.womd_root.rglob("*.tfrecord-*")):
                for offset, raw in _iter_tfrecords_with_offsets(tfr):
                    sc = scenario_pb2.Scenario()
                    sc.ParseFromString(raw)
                    index[sc.scenario_id] = (tfr, offset)

        self._save_index_to_cache(index, cache_path)
        return index

    def list_scenario_ids(self) -> list[str]:
        if self._scenario_index is None:
            self._scenario_index = self._build_index()
        return list(self._scenario_index.keys())

    def load_scenario(self, scenario_id: str) -> tuple[list[LaneSegmentData], dict[str, MapFeatureData]]:
        return self._load_cached(scenario_id)

    def _load_scenario_inner(self, scenario_id: str) -> tuple[list[LaneSegmentData], dict[str, MapFeatureData]]:
        try:
            from synthetic.loaders.proto import scenario_pb2
        except ImportError:
            raise ImportError(
                "Proto stubs not compiled. Run: bash scripts/compile_protos.sh"
            )

        if self._scenario_index is None:
            self._scenario_index = self._build_index()

        entry = self._scenario_index.get(scenario_id)
        if entry is None:
            raise KeyError(f"Scenario {scenario_id!r} not found in {self.womd_root}")

        tfr_path, offset = entry
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(_read_tfrecord_at(tfr_path, offset))

        # Build full-polyline MapFeatureData for every RoadLine and RoadEdge
        map_features: dict[str, MapFeatureData] = {}
        for mf in scenario.map_features:
            if mf.HasField("road_line"):
                pts = _polyline_to_xy(mf.road_line.polyline)
                if pts.shape[0] >= 2:
                    feat_id = str(mf.id)
                    map_features[feat_id] = MapFeatureData(
                        feature_id=feat_id,
                        polyline_xy=pts,
                        womd_type=_map_road_line_type(mf.road_line.type),
                        is_road_edge=False,
                    )
            elif mf.HasField("road_edge"):
                pts = _polyline_to_xy(mf.road_edge.polyline)
                if pts.shape[0] >= 2:
                    feat_id = str(mf.id)
                    map_features[feat_id] = MapFeatureData(
                        feature_id=feat_id,
                        polyline_xy=pts,
                        womd_type=WOMDRoadLineType.TYPE_UNKNOWN,
                        is_road_edge=True,
                    )

        segments: list[LaneSegmentData] = []
        for mf in scenario.map_features:
            if not mf.HasField("lane"):
                continue
            lane = mf.lane

            lane_type_str = WOMD_LANE_TYPE_TO_STR.get(lane.type, "undefined")

            centerline_xy = _polyline_to_xy(lane.polyline)

            # BoundarySegment.lane_start_index and lane_end_index index into this lane's
            # centerline polyline (not into the boundary feature's polyline) — we do not
            # use them. We only reference boundary_feature_id to link to the MapFeature pool.
            left_boundary_feature_ids = _unique_ordered(
                [str(bs.boundary_feature_id) for bs in lane.left_boundaries]
            )
            right_boundary_feature_ids = _unique_ordered(
                [str(bs.boundary_feature_id) for bs in lane.right_boundaries]
            )

            left_neighbor_ids = [str(n.feature_id) for n in lane.left_neighbors]
            right_neighbor_ids = [str(n.feature_id) for n in lane.right_neighbors]

            segments.append(LaneSegmentData(
                lane_id=str(mf.id),
                centerline_xy=centerline_xy,
                left_boundary_feature_ids=left_boundary_feature_ids,
                right_boundary_feature_ids=right_boundary_feature_ids,
                successor_ids=[str(i) for i in lane.exit_lanes],
                predecessor_ids=[str(i) for i in lane.entry_lanes],
                left_neighbor_ids=left_neighbor_ids,
                right_neighbor_ids=right_neighbor_ids,
                lane_type=lane_type_str,
                is_intersection=False,
            ))
        return segments, map_features
