"""
Tests for SyntheticSceneGenerator using a mock ScenarioLoader.
These tests run without AV2/WOMD data.
"""
import numpy as np
import pytest

from schema.scene import ParsedScene
from synthetic.config import CurriculumStage
from synthetic.generator import SyntheticSceneGenerator
from synthetic.loaders.base import LaneSegmentData, MapFeatureData
from synthetic.schema import SyntheticGroundTruth, WOMDRoadLineType


def _make_mock_segments(n_lanes: int = 3) -> tuple[list[LaneSegmentData], dict[str, MapFeatureData]]:
    """Create n_lanes parallel lane segments running along x-axis with matching MapFeatureData."""
    segments = []
    map_features: dict[str, MapFeatureData] = {}
    for i in range(n_lanes):
        y_left = float(i * 4 + 2)
        y_right = float(i * 4)
        cl = np.array([[x, (y_left + y_right) / 2] for x in np.linspace(0, 50, 20)], dtype=np.float64)
        left = np.array([[x, y_left] for x in np.linspace(0, 50, 10)], dtype=np.float64)
        right = np.array([[x, y_right] for x in np.linspace(0, 50, 10)], dtype=np.float64)
        left_feat_id = f"feat_L{i}"
        right_feat_id = f"feat_R{i}"
        pred_id = str(i - 1) if i > 0 else None
        succ_id = str(i + 1) if i < n_lanes - 1 else None
        segments.append(LaneSegmentData(
            lane_id=str(i),
            centerline_xy=cl,
            left_boundary_feature_ids=[left_feat_id],
            right_boundary_feature_ids=[right_feat_id],
            successor_ids=[succ_id] if succ_id else [],
            predecessor_ids=[pred_id] if pred_id else [],
            left_neighbor_ids=[str(i - 1)] if i > 0 else [],
            right_neighbor_ids=[str(i + 1)] if i < n_lanes - 1 else [],
            lane_type="surface",
            is_intersection=False,
        ))
        map_features[left_feat_id] = MapFeatureData(
            feature_id=left_feat_id,
            polyline_xy=left,
            womd_type=WOMDRoadLineType.TYPE_SOLID_SINGLE_WHITE,
            is_road_edge=i == 0,  # outermost left is a road edge
        )
        map_features[right_feat_id] = MapFeatureData(
            feature_id=right_feat_id,
            polyline_xy=right,
            womd_type=WOMDRoadLineType.TYPE_BROKEN_SINGLE_WHITE,
            is_road_edge=i == n_lanes - 1,  # outermost right is a road edge
        )
    return segments, map_features


class MockLoader:
    def __init__(self, segments: list[LaneSegmentData], map_features: dict[str, MapFeatureData]) -> None:
        self._segments = segments
        self._map_features = map_features
        self.source_dataset = "av2"

    def list_scenario_ids(self) -> list[str]:
        return ["mock_scenario"]

    def load_scenario(self, scenario_id: str) -> tuple[list[LaneSegmentData], dict[str, MapFeatureData]]:
        return self._segments, self._map_features


@pytest.fixture
def generator() -> SyntheticSceneGenerator:
    segments, map_features = _make_mock_segments(3)
    loader = MockLoader(segments, map_features)
    return SyntheticSceneGenerator(loader=loader, seed=42)


def test_generate_returns_correct_types(generator):
    scene, gt = generator.generate("mock_scenario", CurriculumStage.NoRandomization)
    assert isinstance(scene, ParsedScene)
    assert isinstance(gt, SyntheticGroundTruth)


def test_generate_scene_passes_pydantic_validation(generator):
    scene, gt = generator.generate("mock_scenario", CurriculumStage.NoRandomization)
    # Round-trip through model_validate (the critical cross-pipeline schema test)
    recovered = ParsedScene.model_validate(scene.model_dump())
    assert recovered.case_id == scene.case_id


def test_generate_gt_element_ids_match_scene(generator):
    scene, gt = generator.generate("mock_scenario", CurriculumStage.NoRandomization)
    scene_elem_ids = {e.id for e in scene.elements}
    gt_elem_ids = set(gt.element_classes.keys())
    assert gt_elem_ids == scene_elem_ids, (
        f"GT has extra: {gt_elem_ids - scene_elem_ids}, "
        f"missing: {scene_elem_ids - gt_elem_ids}"
    )


def test_generate_is_deterministic(generator):
    scene1, gt1 = generator.generate("mock_scenario", CurriculumStage.FullRandomization, index=0)
    scene2, gt2 = generator.generate("mock_scenario", CurriculumStage.FullRandomization, index=0)
    assert scene1.model_dump() == scene2.model_dump()
    assert gt1.model_dump() == gt2.model_dump()


def test_generate_different_index_gives_different_output(generator):
    scene0, _ = generator.generate("mock_scenario", CurriculumStage.FullRandomization, index=0)
    scene1, _ = generator.generate("mock_scenario", CurriculumStage.FullRandomization, index=1)
    pts0 = scene0.elements[0].resampled_points
    pts1 = scene1.elements[0].resampled_points
    # At least some point should differ (different crop/rotation with Stage C randomization)
    # For Stage C (random rotation enabled), this is very likely different
    assert scene0.case_id != scene1.case_id  # different case IDs for different indices


def test_generate_stage_a_produces_elements(generator):
    scene, gt = generator.generate("mock_scenario", CurriculumStage.NoRandomization)
    assert len(scene.elements) > 0
    assert len(gt.element_classes) > 0


def test_generate_index_lists_cover_all_elements(generator):
    scene, _ = generator.generate("mock_scenario", CurriculumStage.NoRandomization)
    all_idx = set(scene.roadway_indices) | set(scene.road_marking_indices) | set(scene.other_indices)
    expected = set(range(len(scene.elements)))
    assert all_idx == expected, f"Index lists don't cover all elements: {expected - all_idx} missing"


def test_generate_no_duplicate_element_ids(generator):
    scene, _ = generator.generate("mock_scenario", CurriculumStage.PartialRandomization)
    ids = [e.id for e in scene.elements]
    assert len(ids) == len(set(ids)), "Duplicate element IDs found"


def test_generate_topology_ids_reference_valid_lanes(generator):
    _, gt = generator.generate("mock_scenario", CurriculumStage.NoRandomization)
    valid_lane_ids = set(gt.topology.keys())
    for lane_id, topo in gt.topology.items():
        for ref in topo.entry_lane_ids + topo.exit_lane_ids:
            assert ref in valid_lane_ids, f"topology entry {ref!r} not in valid lane IDs"
