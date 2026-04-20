"""
Batch texture extraction — extract all sign textures from a directory of .far
files and save them to a single flat pickle cache.

The server loads this pickle on startup so sign textures from other files in
the corpus can be resolved even when a FAR file lacks its own <textures> block.

Usage:
    uv run python -m annotation.scripts.batch_textures \\
        --data-dir data/nhtsa-ciss/data/output \\
        --output annotation/annotations/tex_cache.pkl \\
        --merge
"""

import argparse
import pickle
from pathlib import Path

from lxml import etree

_PARSER = etree.XMLParser(huge_tree=True)


def extract_textures(far_path: Path) -> dict[str, str]:
    """Return {normalized_key: base64_data} for one FAR file."""
    try:
        tree = etree.parse(str(far_path), _PARSER)
    except Exception as e:
        print(f"  [warn] Failed to parse {far_path.name}: {e}")
        return {}

    textures = {}
    tex_block = tree.find(".//textures")
    if tex_block is None:
        return textures

    for tex in tex_block.findall("tex"):
        key = tex.get("key", "").lower().replace("\\", "/")
        if not key:
            continue
        file_elem = tex.find("file")
        if file_elem is not None and file_elem.get("data"):
            textures[key] = file_elem.get("data")

    return textures


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract sign textures from .far files into a flat pickle cache"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory to search for .far files (searched recursively)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotation/annotations/tex_cache.pkl"),
        help="Output pickle file path (default: annotation/annotations/tex_cache.pkl)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing pickle instead of overwriting",
    )
    args = parser.parse_args()

    far_files = sorted(args.data_dir.rglob("*.far"))
    if not far_files:
        print(f"No .far files found in {args.data_dir}")
        return

    print(f"Found {len(far_files)} .far files in {args.data_dir}")

    existing: dict = {}
    if args.merge and args.output.exists():
        with open(args.output, "rb") as f:
            existing = pickle.load(f)
        print(f"Loaded {len(existing)} existing texture entries from {args.output}")

    cache: dict[str, str] = dict(existing)
    files_with_textures = 0

    for far_path in far_files:
        textures = extract_textures(far_path)
        if textures:
            files_with_textures += 1
            new = sum(1 for k in textures if k not in cache)
            cache.update(textures)
            print(f"  {far_path.name}: {len(textures)} textures ({new} new)")

    print(f"\n{files_with_textures} / {len(far_files)} files had textures")
    print(f"Total unique texture keys: {len(cache)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
