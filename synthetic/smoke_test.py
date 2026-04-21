"""
Smoke test for the synthetic data pipeline.
Run from the project root:
    python synthetic/smoke_test.py

Generates 9 samples (3 mock scenarios × stages A/B/C) and validates them.
No AV2/WOMD data required — uses the MockLoader from the test suite.
Target runtime: < 10 seconds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema.scene import ParsedScene
from synthetic.config import CurriculumStage
from synthetic.generator import SyntheticSceneGenerator
from synthetic.loaders.base import LaneSegmentData
from synthetic.schema import SyntheticGroundTruth, WOMDRoadLineType


def _make_segments(n_lanes: int = 4, length: float = 80.0) -> list[LaneSegmentData]:
    segs = []
    for i in range(n_lanes):
        y_left = float(i * 4 + 2)
        y_right = float(i * 4)
        cl = np.array([[x, (y_left + y_right) / 2] for x in np.linspace(0, length, 30)], dtype=np.float64)
        left = np.array([[x, y_left] for x in np.linspace(0, length, 15)], dtype=np.float64)
        right = np.array([[x, y_right] for x in np.linspace(0, length, 15)], dtype=np.float64)
        segs.append(LaneSegmentData(
            lane_id=str(i),
            centerline_xy=cl,
            left_boundary_xy=left,
            right_boundary_xy=right,
            successor_ids=[str(i + 1)] if i < n_lanes - 1 else [],
            predecessor_ids=[str(i - 1)] if i > 0 else [],
            left_neighbor_id=str(i - 1) if i > 0 else None,
            right_neighbor_id=str(i + 1) if i < n_lanes - 1 else None,
            lane_type="surface",
            left_mark_type="SOLID_WHITE",
            right_mark_type="DASHED_WHITE",
            is_intersection=False,
            left_womd_type=WOMDRoadLineType.TYPE_SOLID_SINGLE_WHITE,
            right_womd_type=WOMDRoadLineType.TYPE_BROKEN_SINGLE_WHITE,
        ))
    return segs


class MockLoader:
    def __init__(self, scenario_ids: list[str]) -> None:
        self._ids = scenario_ids
        self.source_dataset = "av2"

    def list_scenario_ids(self) -> list[str]:
        return self._ids

    def load_scenario(self, scenario_id: str) -> list[LaneSegmentData]:
        seed = abs(hash(scenario_id)) % 100
        n_lanes = 3 + seed % 3
        return _make_segments(n_lanes=n_lanes)


def main() -> None:
    scenario_ids = ["scenario_A", "scenario_B", "scenario_C"]
    stages = list(CurriculumStage)

    loader = MockLoader(scenario_ids)
    gen = SyntheticSceneGenerator(loader=loader, seed=42)

    t0 = time.perf_counter()
    results = []

    for scenario_id in scenario_ids:
        for stage in stages:
            scene, gt = gen.generate(scenario_id, stage, index=0)

            # Validate schema
            recovered = ParsedScene.model_validate(scene.model_dump())
            assert recovered.case_id == scene.case_id, "Pydantic round-trip failed"

            # GT element IDs must match scene element IDs
            scene_ids = {e.id for e in scene.elements}
            gt_ids = set(gt.element_classes.keys())
            assert gt_ids == scene_ids, f"GT/scene ID mismatch: extra={gt_ids-scene_ids}, missing={scene_ids-gt_ids}"

            # Index lists must cover all elements
            all_idx = set(scene.roadway_indices) | set(scene.road_marking_indices) | set(scene.other_indices)
            assert all_idx == set(range(len(scene.elements))), "Index lists don't cover all elements"

            results.append({
                "scenario": scenario_id,
                "stage": stage.value,
                "n_elements": len(scene.elements),
                "n_lanes": len(gt.topology),
                "n_topology_edges": sum(
                    len(t.entry_lane_ids) + len(t.exit_lane_ids)
                    for t in gt.topology.values()
                ),
                "n_texts": len(scene.texts),
            })

    elapsed = time.perf_counter() - t0

    print(f"\nSmoke test — {len(results)} samples in {elapsed:.2f}s")
    print(f"{'Scenario':<15} {'Stage':<6} {'Elements':>9} {'Lanes':>7} {'Topo edges':>11} {'Texts':>7}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['scenario']:<15} {r['stage']:<6} {r['n_elements']:>9} "
            f"{r['n_lanes']:>7} {r['n_topology_edges']:>11} {r['n_texts']:>7}"
        )

    assert elapsed < 10.0, f"Smoke test too slow: {elapsed:.1f}s (limit 10s)"
    print(f"\nAll {len(results)} samples valid. ✓")


if __name__ == "__main__":
    main()
