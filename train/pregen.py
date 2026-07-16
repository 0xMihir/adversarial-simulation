"""
Materialize SyntheticCISSDataset to SampleArrays shards (see synthetic/pregen_dataset.py
for the format and the why). CPU-only — run it on Anvil's standard partition, not the
GPU allocation:

    # whole training split on one 128-core wholenode (~2h):
    sbatch -p wholenode -A cis251316 -N 1 -n 128 -t 04:00:00 --wrap \
      "cd /anvil/projects/x-cis251316/mpatil/adv-sim-train && \
       uv run python train/pregen.py --split training \
         --out /anvil/scratch/x-mpatil3/ciss-pregen --workers 126"

    # or spread across an array job (task i of N takes every N-th shard):
    #   --num-tasks 4 --task-id $SLURM_ARRAY_TASK_ID

Shards are written atomically (idx.npz last) so a killed job resumes with --resume.
Generation args (--samples-per-scenario, --seed, --stage) must match what train.py
would have used; they're echoed into the manifest for the record.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic.config import CurriculumStage
from synthetic.dataset import SyntheticCISSDataset
from synthetic.loaders import WOMDScenarioLoader
from synthetic.pregen_dataset import FORMAT_VERSION, MANIFEST_NAME, shard_paths


def write_shard(out: Path, shard: int, items: list) -> None:
    """items: list of (SampleArrays, inv_transform, prof) in slot order."""
    samples = [it[0] for it in items]
    invs = np.stack([it[1] for it in items]).astype(np.float64)

    def offsets(lengths: list[int]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    feat = np.concatenate([s.feat for s in samples], axis=0).astype(np.float32)
    prim = np.concatenate([s.primitive_id for s in samples]).astype(np.int32)
    labels = np.concatenate([s.labels for s in samples]).astype(np.int16)
    tokens = np.concatenate([s.lane_tokens for s in samples]).astype(np.int32)

    p = shard_paths(out, shard)
    np.save(p["feat"], feat)
    np.save(p["prim"], prim)
    np.save(p["labels"], labels)
    np.save(p["tokens"], tokens)
    # idx last: its existence marks the shard complete (resume checks it)
    np.savez(
        p["idx"],
        line_off=offsets([s.feat.shape[0] for s in samples]),
        label_off=offsets([s.labels.shape[0] for s in samples]),
        token_off=offsets([s.lane_tokens.shape[0] for s in samples]),
        inv=invs,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="output root; split subdir is created")
    ap.add_argument("--split", choices=["training", "validation"], required=True)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data" / "waymo_motion")
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-scenarios", type=int, default=None)
    ap.add_argument("--samples-per-scenario", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", default="NoRandomization", choices=[s.name for s in CurriculumStage])
    ap.add_argument("--num-tasks", type=int, default=1, help="SLURM array width")
    ap.add_argument("--task-id", type=int, default=0, help="this task's index in [0, num-tasks)")
    ap.add_argument("--resume", action="store_true", help="skip shards whose idx.npz exists")
    args = ap.parse_args()

    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)

    ds = SyntheticCISSDataset(
        WOMDScenarioLoader(args.data_root / args.split),
        CurriculumStage[args.stage],
        samples_per_scenario=args.samples_per_scenario,
        base_seed=args.seed,
        max_scenarios=args.max_scenarios,
        transform_in_worker=True,
    )
    n = len(ds)
    n_shards = (n + args.shard_size - 1) // args.shard_size

    # Manifest first (idempotent across array tasks — same content from each).
    manifest = {
        "format_version": FORMAT_VERSION,
        "n_samples": n,
        "shard_size": args.shard_size,
        "split": args.split,
        "stage": args.stage,
        "samples_per_scenario": args.samples_per_scenario,
        "seed": args.seed,
        "max_scenarios": args.max_scenarios,
        "feat_dim": 7,
    }
    manifest_path = out / MANIFEST_NAME
    if manifest_path.exists():
        with open(manifest_path) as f:
            prior = json.load(f)
        if prior != manifest:
            raise SystemExit(
                f"{manifest_path} exists with different config — refusing to mix. "
                f"Prior: {prior}"
            )
    else:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    my_shards = [s for s in range(n_shards) if s % args.num_tasks == args.task_id]
    if args.resume:
        my_shards = [s for s in my_shards if not shard_paths(out, s)["idx"].exists()]
    print(f"[task {args.task_id}/{args.num_tasks}] {len(my_shards)} shards to write "
          f"(of {n_shards} total, {n} samples)")

    # Workers ship small batches; shards are accumulated and written in the main
    # process. (A one-batch-per-shard batch_sampler puts a whole ~2GB pickled shard
    # in each worker with 2x that in flight per prefetch_factor — OOMs the node.)
    shard_sizes = [
        min((s + 1) * args.shard_size, n) - s * args.shard_size for s in my_shards
    ]
    flat_indices: list[int] = []
    for s in my_shards:
        flat_indices.extend(range(s * args.shard_size, min((s + 1) * args.shard_size, n)))

    loader = DataLoader(
        Subset(ds, flat_indices),
        batch_size=128,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda b: b,  # keep the raw item list; write_shard does the packing
        prefetch_factor=4 if args.workers > 0 else None,
    )

    t_start = time.time()
    buf: list = []
    k = 0  # position in my_shards
    for items in loader:
        buf.extend(items)
        while k < len(my_shards) and len(buf) >= shard_sizes[k]:
            chunk, buf = buf[: shard_sizes[k]], buf[shard_sizes[k]:]
            write_shard(out, my_shards[k], chunk)
            k += 1
            rate = k / (time.time() - t_start)
            print(
                f"shard {my_shards[k-1]:05d} written ({k}/{len(my_shards)}, "
                f"{rate:.2f} shards/s, eta {(len(my_shards) - k) / max(rate, 1e-9) / 60:.0f} min)",
                flush=True,
            )
    assert k == len(my_shards) and not buf, (k, len(my_shards), len(buf))
    print(f"[task {args.task_id}] done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
