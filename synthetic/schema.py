from enum import Enum
from pydantic import BaseModel
from schema.scene import Point2D


class ElementClass(str, Enum):
    ROAD_EDGE = "ROAD_EDGE"
    LANE_MARKING_SOLID = "LANE_MARKING_SOLID"
    LANE_MARKING_DASHED = "LANE_MARKING_DASHED"
    CROSSWALK_STRIPE = "CROSSWALK_STRIPE"
    STOP_LINE = "STOP_LINE"
    SHOULDER_LINE = "SHOULDER_LINE"
    SIGN_SYMBOL = "SIGN_SYMBOL"
    ARROW_SYMBOL = "ARROW_SYMBOL"
    ANNOTATION_TEXT = "ANNOTATION_TEXT"
    ANNOTATION_GEOMETRY = "ANNOTATION_GEOMETRY"
    VEHICLE_OUTLINE = "VEHICLE_OUTLINE"
    IMPACT_MARKER = "IMPACT_MARKER"
    OTHER = "OTHER"


class WOMDRoadLineType(str, Enum):
    TYPE_SOLID_SINGLE_WHITE = "TYPE_SOLID_SINGLE_WHITE"
    TYPE_SOLID_SINGLE_YELLOW = "TYPE_SOLID_SINGLE_YELLOW"
    TYPE_BROKEN_SINGLE_WHITE = "TYPE_BROKEN_SINGLE_WHITE"
    TYPE_BROKEN_SINGLE_YELLOW = "TYPE_BROKEN_SINGLE_YELLOW"
    TYPE_SOLID_DOUBLE_YELLOW = "TYPE_SOLID_DOUBLE_YELLOW"
    TYPE_SOLID_DOUBLE_WHITE = "TYPE_SOLID_DOUBLE_WHITE"
    TYPE_UNKNOWN = "TYPE_UNKNOWN"


class LaneTopology(BaseModel):
    entry_lane_ids: list[str]    # predecessor lane IDs
    exit_lane_ids: list[str]     # successor lane IDs
    left_neighbor_id: str | None
    right_neighbor_id: str | None


class BoundaryAssignment(BaseModel):
    left_boundary_element_id: str | None
    right_boundary_element_id: str | None
    left_mark_type: WOMDRoadLineType
    right_mark_type: WOMDRoadLineType


class InverseTransform(BaseModel):
    """3×3 row-major affine that maps normalized → original metric coords."""
    values: list[list[float]]     # 3×3
    original_centroid_x: float
    original_centroid_y: float
    scale_factor: float           # longest hull diagonal in original coords
    pca_rotation_rad: float       # angle applied during normalization; 0.0 if PCA disabled


class SyntheticGroundTruth(BaseModel):
    scene_id: str
    source_scenario_id: str
    source_dataset: str           # "av2" | "womd"
    curriculum_stage: str         # CurriculumStage value

    element_classes: dict[str, str]              # element_id → ElementClass value
    lane_centerlines: dict[str, list[Point2D]]   # lane_id → centerline pts (normalized coords)
    topology: dict[str, LaneTopology]            # lane_id → LaneTopology
    boundary_assignments: dict[str, BoundaryAssignment]  # lane_id → BoundaryAssignment
    marking_types: dict[str, str]                # element_id → WOMDRoadLineType value
    endpoint_adjacency: dict[str, list[str]]     # element_id → element_ids sharing an endpoint

    inverse_transform: InverseTransform
