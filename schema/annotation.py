from enum import Enum
from pydantic import BaseModel
from .scene import ParsedScene, Point2D


class ElementStatus(str, Enum):
    AUTO = "auto"           # Machine-generated, not reviewed
    CONFIRMED = "confirmed" # Human verified as correct
    CORRECTED = "corrected" # Human modified
    REJECTED = "rejected"   # Excluded from training


class LaneAnnotation(BaseModel):
    id: str
    polyline: list[Point2D]
    raw_control_points: list[Point2D] = []
    lane_type: str = "driving"        # driving | bike | bus | shoulder | parking
    left_boundary_type: str = "unknown"   # road_edge | solid_white | broken_white |
    right_boundary_type: str = "unknown"  # solid_yellow | broken_yellow | double_yellow |
                                          # double_white | virtual | unknown
    entry_lanes: list[str] = []       # Predecessor lane IDs
    exit_lanes: list[str] = []        # Successor lane IDs
    speed_limit_mph: float | None = None
    status: ElementStatus = ElementStatus.AUTO
    source_element_ids: list[str] = []
    notes: str | None = None


class VehicleWaypoint(BaseModel):
    position: Point2D
    heading: float        # Radians
    timestamp_index: int  # Ordinal position in sequence (0 = first)
    phase: str = "pre_crash"  # pre_crash | collision | post_crash | final_rest
    lane_id: str | None = None
    speed_estimate: float | None = None


class VehicleAnnotation(BaseModel):
    id: str
    vehicle_type: str = "car"  # car | truck | motorcycle | pedestrian | bicycle | animal
    waypoints: list[VehicleWaypoint]
    status: ElementStatus = ElementStatus.AUTO


class LaneConnection(BaseModel):
    id: str
    from_lane_id: str
    to_lane_id: str
    from_end: str = "end"    # "start" | "end" — which endpoint of from_lane
    to_end: str = "start"    # "start" | "end" — which endpoint of to_lane
    connection_type: str = "through"  # through | left_turn | right_turn | u_turn | merge
    control_points: list[Point2D] | None = None  # Optional Bezier through intersection
    status: ElementStatus = ElementStatus.AUTO


class CorrectionRecord(BaseModel):
    timestamp: str
    element_type: str   # lane | vehicle | connection
    element_id: str
    action: str         # confirmed | corrected | rejected | created | deleted
    previous_value: dict | None = None


class CaseAnnotation(BaseModel):
    case_id: str
    far_filename: str
    scene: ParsedScene
    auto_centerlines: list[LaneAnnotation]  # Immutable original auto output
    lanes: list[LaneAnnotation]             # Working copy (auto + human edits)
    vehicles: list[VehicleAnnotation]
    lane_connections: list[LaneConnection]
    corrections: list[CorrectionRecord] = []
    created_at: str
    updated_at: str
    annotator: str | None = None
    workflow_status: str = "not_started"  # not_started | in_progress | reviewed
    auto_confidence: float | None = None
    hidden_roadway_ids: list[str] = []


class CaseAnnotationUpdate(BaseModel):
    """Partial update payload for PUT /api/annotations/{id}"""
    lanes: list[LaneAnnotation] | None = None
    vehicles: list[VehicleAnnotation] | None = None
    lane_connections: list[LaneConnection] | None = None
    workflow_status: str | None = None
    annotator: str | None = None
    corrections: list[CorrectionRecord] | None = None
    hidden_roadway_ids: list[str] | None = None


class CaseSummary(BaseModel):
    """Lightweight summary returned by GET /api/cases"""
    id: str
    filename: str
    annotated: bool
    workflow_status: str
    updated_at: str | None
    lane_count: int
    vehicle_count: int
    auto_confidence: float | None
