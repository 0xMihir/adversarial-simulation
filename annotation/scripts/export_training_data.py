#!/usr/bin/env python3
"""
Export reviewed annotations to training formats.

Usage:
  python export_training_data.py                 # all reviewed → stdout JSONL
  python export_training_data.py --all           # include auto + unreviewed
  python export_training_data.py -o out.jsonl    # write to file
"""
import argparse
import json
import sys
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).parents[1] / "annotations"


def load_annotations(status_filter: str | None = "reviewed"):
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        ann = json.loads(path.read_text())
        if status_filter and ann.get("workflow_status") != status_filter:
            continue
        yield ann


def to_training_sample(ann: dict) -> dict:
    confirmed = {"confirmed", "corrected"}

    lanes = [
        {
            "id": l["id"],
            "polyline": l["polyline"],
            "lane_type": l["lane_type"],
            "left_boundary": l["left_boundary_type"],
            "right_boundary": l["right_boundary_type"],
            "entry_lanes": l["entry_lanes"],
            "exit_lanes": l["exit_lanes"],
            "speed_limit_mph": l["speed_limit_mph"],
        }
        for l in ann.get("lanes", [])
        if l["status"] in confirmed
    ]

    vehicles = [
        {
            "id": v["id"],
            "type": v["vehicle_type"],
            "waypoints": [
                {
                    "x": w["position"]["x"],
                    "y": w["position"]["y"],
                    "heading": w["heading"],
                    "phase": w["phase"],
                    "timestamp_index": w["timestamp_index"],
                }
                for w in v["waypoints"]
            ],
        }
        for v in ann.get("vehicles", [])
        if v["status"] in confirmed
    ]

    connections = [
        {
            "from": c["from_lane_id"],
            "to": c["to_lane_id"],
            "type": c["connection_type"],
        }
        for c in ann.get("lane_connections", [])
        if c["status"] in confirmed
    ]

    return {
        "case_id": ann["case_id"],
        "lanes": lanes,
        "vehicles": vehicles,
        "connections": connections,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--all", action="store_true", help="Include unreviewed cases")
    parser.add_argument(
        "--status", default="reviewed", help="Filter by workflow_status"
    )
    args = parser.parse_args()

    status = None if args.all else args.status
    out = open(args.output, "w") if args.output else sys.stdout

    count = 0
    for ann in load_annotations(status_filter=status):
        sample = to_training_sample(ann)
        if sample["lanes"] or sample["vehicles"]:
            out.write(json.dumps(sample) + "\n")
            count += 1

    if args.output:
        out.close()
        print(f"Exported {count} samples to {args.output}", file=sys.stderr)
    else:
        print(f"# Exported {count} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
