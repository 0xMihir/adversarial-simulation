"""
Batch classification script — pre-classify all symbol names across a directory
of .far files and save results to a pickle cache.

The server loads this pickle on startup so no inference happens during processing.

Usage:
    uv run python -m annotation.scripts.batch_classify \\
        --data-dir data/nhtsa-ciss/data/output \\
        --output annotation/annotations/clf_cache.pkl \\
        --merge
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from lxml import etree
from transformers import pipeline as hf_pipeline

_MODEL_ID = "sileod/deberta-v3-large-tasksource-nli"

CLASSIFICATION_LABELS = [
    "vehicle (car, suv, truck, pickup, bus)",
    "traffic light / signal",
    "road marking / lane line",
    "direction arrow (north, south, east, west)",
    "turn direction",
    "pedestrian",
    "background / decoration",
]
LABEL_VEHICLE = CLASSIFICATION_LABELS[0]


_PARSER = etree.XMLParser(huge_tree=True)


def extract_symbol_names(far_path: Path) -> set[str]:
    """Extract all unique symbol item names from a .far file using lxml xpath."""
    try:
        tree = etree.parse(str(far_path), _PARSER)
        names = tree.xpath('//item[@type="symbol"]/@nam')
        return {n for n in names if n and n.strip()}
    except Exception as e:
        print(f"  [warn] Failed to parse {far_path.name}: {e}")
        return set()


def classify_names(clf, names: list[str], existing: dict) -> dict:
    """Run batched MNLI on uncached names and return merged cache dict."""
    uncached = [n for n in names if n not in existing]
    if not uncached:
        print(f"  All {len(names)} names already cached, nothing to do.")
        return existing

    print(f"  Classifying {len(uncached)} new names ({len(names) - len(uncached)} cached)...")
    texts = [n.lower().replace("_", " ") for n in uncached]
    results = clf(
        texts,
        candidate_labels=CLASSIFICATION_LABELS,
        multi_label=True,
        hypothesis_template="This item is a {}.",
    )

    cache = dict(existing)
    for name, out in zip(uncached, results):
        scores = out["scores"]
        max_idx = int(np.argmax(scores))
        predicted_class = out["labels"][max_idx]
        predicted_prob = float(scores[max_idx])
        cache[name] = {
            "is_vehicle": predicted_class == LABEL_VEHICLE and predicted_prob > 0.7,
            "predicted_class": predicted_class,
            "predicted_probability": predicted_prob,
        }

    return cache


def main():
    parser = argparse.ArgumentParser(description="Batch classify symbol names from .far files")
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory to search for .far files (searched recursively)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotation/annotations/clf_cache.pkl"),
        help="Output pickle file path (default: annotation/annotations/clf_cache.pkl)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing pickle instead of overwriting",
    )
    args = parser.parse_args()

    # --- Collect symbol names ---
    far_files = sorted(args.data_dir.rglob("*.far"))
    if not far_files:
        print(f"No .far files found in {args.data_dir}")
        return

    print(f"Found {len(far_files)} .far files in {args.data_dir}")
    all_names: set[str] = set()
    for f in far_files:
        names = extract_symbol_names(f)
        all_names.update(names)
    print(f"Unique symbol names across all files: {len(all_names)}")
    print(all_names)
    # --- Load existing cache ---
    existing: dict = {}
    if args.merge and args.output.exists():
        with open(args.output, "rb") as f:
            existing = pickle.load(f)
        print(f"Loaded {len(existing)} existing entries from {args.output}")

    # --- Load model ---
    print(f"Loading model {_MODEL_ID} ...")
    clf = hf_pipeline(
        "zero-shot-classification",
        model=_MODEL_ID,
        local_files_only=True,
    )

    # --- Classify ---
    cache = classify_names(clf, sorted(all_names), existing)

    # --- Save ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved {len(cache)} entries to {args.output}")

    vehicle_count = sum(1 for v in cache.values() if v["is_vehicle"])
    print(f"  {vehicle_count} classified as vehicle, {len(cache) - vehicle_count} as non-vehicle")


if __name__ == "__main__":
    main()
