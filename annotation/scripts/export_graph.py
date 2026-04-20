#!/usr/bin/env python3
"""
Export lane graph structures for GNN training (GATv2Conv / PyTorch Geometric).

Nodes: lane endpoints (start + end of each polyline)
Edges: within-lane + lane connections

Usage:
  python export_graph.py                     # print JSON to stdout
  python export_graph.py --output graphs/    # write .json per case
  python export_graph.py --pt graphs/        # write .pt tensors (requires torch)
"""
import argparse
import json
import math
import sys
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).parents[1] / "annotations"


def polyline_length(polyline: list[dict]) -> float:
    total = 0.0
    for i in range(1, len(polyline)):
        dx = polyline[i]["x"] - polyline[i - 1]["x"]
        dy = polyline[i]["y"] - polyline[i - 1]["y"]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def to_graph(ann: dict) -> dict:
    lanes = [l for l in ann.get("lanes", []) if l["status"] != "rejected"]

    nodes = []
    edges = []

    for lane in lanes:
        poly = lane["polyline"]
        if len(poly) < 2:
            continue
        nodes.append({
            "id": f"{lane['id']}_start",
            "x": poly[0]["x"],
            "y": poly[0]["y"],
            "lane_id": lane["id"],
            "endpoint": "start",
        })
        nodes.append({
            "id": f"{lane['id']}_end",
            "x": poly[-1]["x"],
            "y": poly[-1]["y"],
            "lane_id": lane["id"],
            "endpoint": "end",
        })
        edges.append({
            "source": f"{lane['id']}_start",
            "target": f"{lane['id']}_end",
            "type": "within_lane",
            "length": polyline_length(poly),
        })

    for conn in ann.get("lane_connections", []):
        if conn["status"] == "rejected":
            continue
        edges.append({
            "source": f"{conn['from_lane_id']}_end",
            "target": f"{conn['to_lane_id']}_start",
            "type": conn["connection_type"],
            "length": None,
        })

    return {"case_id": ann["case_id"], "nodes": nodes, "edges": edges}


def to_pt_tensors(graph: dict):
    """Convert graph to PyTorch Geometric Data object."""
    import torch
    from torch_geometric.data import Data  # type: ignore[import]

    node_ids = {n["id"]: i for i, n in enumerate(graph["nodes"])}
    x = torch.tensor(
        [[n["x"], n["y"]] for n in graph["nodes"]], dtype=torch.float
    )

    edge_index_list = []
    for e in graph["edges"]:
        s = node_ids.get(e["source"])
        t = node_ids.get(e["target"])
        if s is not None and t is not None:
            edge_index_list.append([s, t])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, case_id=graph["case_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Directory to write JSON graphs")
    parser.add_argument("--pt", help="Directory to write .pt tensors")
    parser.add_argument("--status", default="reviewed", help="Filter workflow_status")
    args = parser.parse_args()

    graphs = []
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        ann = json.loads(path.read_text())
        if args.status and ann.get("workflow_status") != args.status:
            continue
        g = to_graph(ann)
        graphs.append(g)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for g in graphs:
            (out_dir / f"{g['case_id']}.json").write_text(json.dumps(g, indent=2))
        print(f"Wrote {len(graphs)} graph JSON files to {args.output}", file=sys.stderr)

    elif args.pt:
        out_dir = Path(args.pt)
        out_dir.mkdir(parents=True, exist_ok=True)
        import torch
        for g in graphs:
            data = to_pt_tensors(g)
            torch.save(data, out_dir / f"{g['case_id']}.pt")
        print(f"Wrote {len(graphs)} .pt files to {args.pt}", file=sys.stderr)

    else:
        for g in graphs:
            print(json.dumps(g))


if __name__ == "__main__":
    main()
