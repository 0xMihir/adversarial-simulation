from synthetic.schema import (
    ElementClass,
    SyntheticGroundTruth,
    WOMDRoadLineType,
)
from schema.scene import Point2D
from schema.annotation import ElementStatus  # check cross-import works


_VALID_GT = dict(
    scene_id="test_0",
    source_scenario_id="abc",
    source_dataset="av2",
    curriculum_stage="C",
    element_classes={"e1": "ROAD_EDGE", "e2": "LANE_MARKING_SOLID"},
    lane_centerlines={"lane1": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}]},
    topology={
        "lane1": {
            "entry_lane_ids": [],
            "exit_lane_ids": [],
            "left_neighbor_id": None,
            "right_neighbor_id": None,
        }
    },
    boundary_assignments={
        "lane1": {
            "left_boundary_element_id": "e1",
            "right_boundary_element_id": "e2",
            "left_mark_type": "TYPE_UNKNOWN",
            "right_mark_type": "TYPE_SOLID_SINGLE_WHITE",
        }
    },
    marking_types={"e2": "TYPE_SOLID_SINGLE_WHITE"},
    endpoint_adjacency={"e1": ["e2"], "e2": ["e1"]},
    inverse_transform={
        "values": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "original_centroid_x": 0.0,
        "original_centroid_y": 0.0,
        "scale_factor": 100.0,
        "pca_rotation_rad": 0.0,
    },
)


def test_synthetic_ground_truth_validates():
    gt = SyntheticGroundTruth.model_validate(_VALID_GT)
    assert gt.scene_id == "test_0"
    assert gt.curriculum_stage == "C"


def test_element_class_enum_values():
    expected = {
        "ROAD_EDGE", "LANE_MARKING_SOLID", "LANE_MARKING_DASHED", "CROSSWALK_STRIPE",
        "STOP_LINE", "SHOULDER_LINE", "SIGN_SYMBOL", "ARROW_SYMBOL",
        "ANNOTATION_TEXT", "ANNOTATION_GEOMETRY", "VEHICLE_OUTLINE", "IMPACT_MARKER", "OTHER",
    }
    actual = {e.value for e in ElementClass}
    assert actual == expected


def test_womd_road_line_type_has_unknown():
    assert WOMDRoadLineType.TYPE_UNKNOWN in WOMDRoadLineType


def test_schema_imports_from_schema_package():
    # Verify the re-export chain works
    assert ElementStatus.AUTO.value == "auto"
    p = Point2D(x=1.0, y=2.0)
    assert p.x == 1.0
