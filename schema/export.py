from pydantic import BaseModel
from .annotation import LaneAnnotation, VehicleAnnotation, LaneConnection


class TrainingSample(BaseModel):
    case_id: str
    lanes: list[dict]       # WOMD-aligned lane dicts
    vehicles: list[dict]    # Vehicle waypoint sequences
    connections: list[dict] # Lane topology


class GraphNode(BaseModel):
    id: str
    x: float
    y: float
    lane_id: str
    endpoint: str  # "start" | "end"


class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: str  # within_lane | through | left_turn | right_turn | u_turn | merge
    length: float | None = None


class SceneGraph(BaseModel):
    case_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
