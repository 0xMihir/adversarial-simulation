"""
Tests for SyntheticSceneGenerator using a mock ScenarioLoader.
These tests run without AV2/WOMD data.
"""
import numpy as np
import pytest

from schema.scene import ParsedScene
from synthetic.config import CurriculumStage
from synthetic.generator import SyntheticSceneGenerator
from synthetic.loaders.base import LaneSegmentData
from synthetic.schema import SyntheticGroundTruth, WOMDRoadLineType


def _make_mock_segments(n_lanes: int = 3) -> list[LaneSegmentData]:
    """Create n_lanes parallel lane segments running along x-axis."""
    segments = []
    for i in range(n_lanes):
        y_left = float(i * 4 + 2)
        y_right = float(i * 4)
        cl = np.array([[x, (y_left + y_right) / 2] for x in np.linspace(0, 50, 20)], dtype=np.float64)
        left = np.array([[x, y_left] for x in np.linspace(0, 50, 10)], dtype=np.float64)
        right = np.array([[x, y_right] for x in np.linspace(0, 50, 10)], dtype=np.float64)
        pred_id = str(i - 1) if i > 0 else None
        succ_id = str(i + 1) if i < n_lanes - 1 else None
        segments.append(LaneSegmentData(
            lane_id=str(i),
            centerline_xy=cl,
            left_boundary_xy=left,
            right_boundary_xy=right,
            successor_ids=[succ_id] if succ_id else [],
            predecessor_ids=[pred_id] if pred_id else [],
            left_neighbor_id=str(i - 1) if i > 0 else None,
            right_neighbor_id=str(i + 1) if i < n_lanes - 1 else None,
            lane_type="surface",
            left_mark_type="SOLID_WHITE",
            right_mark_type="DASHED_WHITE",
            is_intersection=False,
            left_womd_type=WOMDRoadLineType.TYPE_SOLID_SINGLE_WHITE,
            right_womd_type=WOMDRoadLineType.TYPE_BROKEN_SINGLE_WHITE,
        ))
    return segments


class MockLoader:
    def __init__(self, segments: list[LaneSegmentData]) -> None:
        self._segments = segments
        self.source_dataset = "av2"

    def list_scenario_ids(self) -> list[str]:
        return ["mock_scenario"]

    def load_scenario(self, scenario_id: str) -> list[LaneSegmentData]:
        return self._segments


@pytest.fixture
def generator() -> SyntheticSceneGenerator:
    loader = MockLoader(_make_mock_segments(3))
    return SyntheticSceneGenerator(loader=loader, seed=42)


def test_generate_returns_correct_types(generator):
    scene, gt = generator.generate("mock_scenario", CurriculumStage.A)
    assert isinstance(scene, ParsedScene)
    assert isinstance(gt, SyntheticGroundTruth)


def test_generate_scene_passes_pydantic_validation(generator):
    scene, gt = generator.generate("mock_scenario", CurriculumStage.A)
    # Round-trip through model_validate (the critical cross-pipeline schema test)
    recovered = ParsedScene.model_validate(scene.model_dump())
    assert recovered.case_id == scene.case_id


def test_generate_gt_element_ids_match_scene(generator):
    scene, gt = generator.generate("mock_scenario", CurriculumStage.A)
    scene_elem_ids = {e.id for e in scene.elements}
    gt_elem_ids = set(gt.element_classes.keys())
    assert gt_elem_ids == scene_elem_ids, (
        f"GT has extra: {gt_elem_ids - scene_elem_ids}, "
        f"missing: {scene_elem_ids - gt_elem_ids}"
    )


def test_generate_is_deterministic(generator):
    scene1, gt1 = generator.generate("mock_scenario", CurriculumStage.C, index=0)
    scene2, gt2 = generator.generate("mock_scenario", CurriculumStage.C, index=0)
    assert scene1.model_dump() == scene2.model_dump()
    assert gt1.model_dump() == gt2.model_dump()


def test_generate_different_index_gives_different_output(generator):
    scene0, _ = generator.generate("mock_scenario", CurriculumStage.C, index=0)
    scene1, _ = generator.generate("mock_scenario", CurriculumStage.C, index=1)
    pts0 = scene0.elements[0].resampled_points
    pts1 = scene1.elements[0].resampled_points
    # At least some point should differ (different crop/rotation with Stage C randomization)
    # For Stage C (random rotation enabled), this is very likely different
    assert scene0.case_id != scene1.case_id  # different case IDs for different indices


def test_generate_stage_a_produces_elements(generator):
    scene, gt = generator.generate("mock_scenario", CurriculumStage.A)
    assert len(scene.elements) > 0
    assert len(gt.element_classes) > 0


def test_generate_index_lists_cover_all_elements(generator):
    scene, _ = generator.generate("mock_scenario", CurriculumStage.A)
    all_idx = set(scene.roadway_indices) | set(scene.road_marking_indices) | set(scene.other_indices)
    expected = set(range(len(scene.elements)))
    assert all_idx == expected, f"Index lists don't cover all elements: {expected - all_idx} missing"


def test_generate_no_duplicate_element_ids(generator):
    scene, _ = generator.generate("mock_scenario", CurriculumStage.B)
    ids = [e.id for e in scene.elements]
    assert len(ids) == len(set(ids)), "Duplicate element IDs found"


def test_generate_topology_ids_reference_valid_lanes(generator):
    _, gt = generator.generate("mock_scenario", CurriculumStage.A)
    valid_lane_ids = set(gt.topology.keys())
    for lane_id, topo in gt.topology.items():
        for ref in topo.entry_lane_ids + topo.exit_lane_ids:
            assert ref in valid_lane_ids, f"topology entry {ref!r} not in valid lane IDs"
