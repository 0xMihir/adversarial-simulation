"""
GET /api/export/jsonl               — stream all annotations as JSONL
GET /api/export/graph/{case_id}     — per-case GNN graph structure
"""
import json
import math
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from schema.annotation import ElementStatus
from schema.export import GraphEdge, GraphNode, SceneGraph
from ..services import annotations as ann_service

router = APIRouter()


def _polyline_length(polyline: list) -> float:
    total = 0.0
    for i in range(1, len(polyline)):
        dx = polyline[i].x - polyline[i - 1].x
        dy = polyline[i].y - polyline[i - 1].y
        total += math.sqrt(dx * dx + dy * dy)
    return total


@router.get("/jsonl")
def export_jsonl(
    status: str | None = Query(None, description="Filter by workflow_status"),
    min_confidence: float | None = Query(None, description="Minimum auto_confidence"),
):
    """
    Stream all annotations as newline-delimited JSON.
    Each line is a training sample (confirmed + corrected elements only).
    """
    def generate():
        for case_id in ann_service.list_annotated_ids():
            ann = ann_service.read_annotation(case_id)
            if ann is None:
                continue
            if status and ann.workflow_status != status:
                continue
            if min_confidence and (ann.auto_confidence or 0.0) < min_confidence:
                continue

            confirmed_statuses = {ElementStatus.CONFIRMED, ElementStatus.CORRECTED}

            sample = {
                "case_id": ann.case_id,
                "lanes": [
                    {
                        "id": l.id,
                        "polyline": [{"x": p.x, "y": p.y} for p in l.polyline],
                        "lane_type": l.lane_type,
                        "left_boundary": l.left_boundary_type,
                        "right_boundary": l.right_boundary_type,
                        "entry_lanes": l.entry_lanes,
                        "exit_lanes": l.exit_lanes,
                        "speed_limit_mph": l.speed_limit_mph,
                        "status": l.status,
                    }
                    for l in ann.lanes
                    if l.status in confirmed_statuses
                ],
                "vehicles": [
                    {
                        "id": v.id,
                        "type": v.vehicle_type,
                        "waypoints": [
                            {
                                "x": w.position.x,
                                "y": w.position.y,
                                "heading": w.heading,
                                "phase": w.phase,
                                "timestamp_index": w.timestamp_index,
                            }
                            for w in v.waypoints
                        ],
                    }
                    for v in ann.vehicles
                    if v.status in confirmed_statuses
                ],
                "connections": [
                    {
                        "from": c.from_lane_id,
                        "to": c.to_lane_id,
                        "type": c.connection_type,
                    }
                    for c in ann.lane_connections
                    if c.status in confirmed_statuses
                ],
            }
            yield json.dumps(sample) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/graph/{case_id}", response_model=SceneGraph)
def export_graph(case_id: str):
    """
    Return pre-computed lane graph for GNN training (GATv2Conv / PyTorch Geometric).
    Nodes = lane endpoints; edges = within-lane + connections.
    """
    from fastapi import HTTPException
    ann = ann_service.read_annotation(case_id)
    if ann is None:
        raise HTTPException(status_code=404, detail=f"Annotation not found: {case_id}")

    lanes = [l for l in ann.lanes if l.status != ElementStatus.REJECTED]

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for lane in lanes:
        if len(lane.polyline) < 2:
            continue
        start_pt = lane.polyline[0]
        end_pt = lane.polyline[-1]
        nodes.append(GraphNode(
            id=f"{lane.id}_start",
            x=start_pt.x,
            y=start_pt.y,
            lane_id=lane.id,
            endpoint="start",
        ))
        nodes.append(GraphNode(
            id=f"{lane.id}_end",
            x=end_pt.x,
            y=end_pt.y,
            lane_id=lane.id,
            endpoint="end",
        ))
        edges.append(GraphEdge(
            source=f"{lane.id}_start",
            target=f"{lane.id}_end",
            edge_type="within_lane",
            length=_polyline_length(lane.polyline),
        ))

    for conn in ann.lane_connections:
        if conn.status == ElementStatus.REJECTED:
            continue
        edges.append(GraphEdge(
            source=f"{conn.from_lane_id}_end",
            target=f"{conn.to_lane_id}_start",
            edge_type=conn.connection_type,
        ))

    return SceneGraph(case_id=case_id, nodes=nodes, edges=edges)
