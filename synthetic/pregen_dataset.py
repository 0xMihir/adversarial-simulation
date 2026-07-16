"""
PregenCISSDataset — train from pre-generated SampleArrays shards.

The on-the-fly synthetic pipeline costs ~0.39 CPU-s per item (dominated by
scene_to_arrays), which is ~50 CPU-s per batch of 128 — more than Anvil's
24-cores-per-H100 budget can produce at the GPU's ~0.75s/step consumption rate.
Since training runs CurriculumStage.NoRandomization the samples are a fixed set,
so train/pregen.py materializes them once (on cheap CPU nodes) and this dataset
replays them at pure-memcpy cost (~1ms/item).

Shard format (out_dir/<split>/):
    manifest.json                     n_samples, shard_size, config echo
    shard_XXXXX.feat.npy    (sumN, 7) float32   VecFormer line features
    shard_XXXXX.prim.npy    (sumN,)   int32     per-line primitive index
    shard_XXXXX.labels.npy  (sumP,)   int16     per-primitive class labels
    shard_XXXXX.tokens.npy  (sumT,)   int32     SeqGrowGraph target tokens
    shard_XXXXX.idx.npz               line/label/token offsets (K+1,) int64
                                      + inv (K, 3, 3) float64
                                      written LAST -> marks the shard complete

coord is not stored: it is reconstructed as (feat[:, 3:5], z=0), see
scene_to_point.scene_to_lines. Sample `idx` lives at shard idx // shard_size,
slot idx % shard_size — pregen writes contiguous fixed-size shards to keep this
mapping trivial.

Items match SyntheticCISSDataset(transform_in_worker=True) exactly:
(SampleArrays, inv_transform (3,3) float64, prof|None) — so
SyntheticCISSDataset.collate_fn is reused unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter, time as wall_time
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from models.data_transform.scene_to_point import SampleArrays

MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1


def shard_paths(root: Path, shard: int) -> dict[str, Path]:
    stem = f"shard_{shard:05d}"
    return {
        "feat": root / f"{stem}.feat.npy",
        "prim": root / f"{stem}.prim.npy",
        "labels": root / f"{stem}.labels.npy",
        "tokens": root / f"{stem}.tokens.npy",
        "idx": root / f"{stem}.idx.npz",
    }


class PregenCISSDataset(Dataset):
    """Read-only Dataset over shards written by train/pregen.py."""

    def __init__(self, root: str | Path, profile: bool = False) -> None:
        self.root = Path(root)
        self.profile = profile

        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found — run train/pregen.py for this split first"
            )
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{manifest_path}: format_version "
                f"{self.manifest.get('format_version')} != {FORMAT_VERSION}"
            )
        self.n_samples: int = self.manifest["n_samples"]
        self.shard_size: int = self.manifest["shard_size"]
        n_shards = (self.n_samples + self.shard_size - 1) // self.shard_size
        missing = [
            s for s in range(n_shards) if not shard_paths(self.root, s)["idx"].exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{self.root}: {len(missing)}/{n_shards} shards incomplete "
                f"(first missing: {missing[0]}) — pregen didn't finish; rerun with --resume"
            )
        # Lazy per-process mmap cache: shard -> dict of open arrays. Populated on
        # first touch in each DataLoader worker (mmaps don't survive fork usefully).
        self._open: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return self.n_samples

    def _shard(self, shard: int) -> dict[str, Any]:
        got = self._open.get(shard)
        if got is None:
            p = shard_paths(self.root, shard)
            idx = np.load(p["idx"])
            got = {
                "feat": np.load(p["feat"], mmap_mode="r"),
                "prim": np.load(p["prim"], mmap_mode="r"),
                "labels": np.load(p["labels"], mmap_mode="r"),
                "tokens": np.load(p["tokens"], mmap_mode="r"),
                # offsets + inv are small; materialize them
                "line_off": idx["line_off"],
                "label_off": idx["label_off"],
                "token_off": idx["token_off"],
                "inv": idx["inv"],
            }
            self._open[shard] = got
        return got

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        t0 = perf_counter() if self.profile else 0.0
        shard, slot = divmod(idx, self.shard_size)
        s = self._shard(shard)

        a, b = s["line_off"][slot], s["line_off"][slot + 1]
        la, lb = s["label_off"][slot], s["label_off"][slot + 1]
        ta, tb = s["token_off"][slot], s["token_off"][slot + 1]

        feat = np.ascontiguousarray(s["feat"][a:b])
        coord = np.concatenate(
            [feat[:, 3:5], np.zeros((feat.shape[0], 1), np.float32)], axis=1
        )
        sample = SampleArrays(
            feat=feat,
            coord=coord,
            primitive_id=s["prim"][a:b].astype(np.int64),
            labels=s["labels"][la:lb].astype(np.int64),
            lane_tokens=s["tokens"][ta:tb].astype(np.int64),
        )
        inv = np.ascontiguousarray(s["inv"][slot])

        if not self.profile:
            return (sample, inv, None)
        dt = perf_counter() - t0
        # Same keys train.py's worker_prof aggregation expects from the synthetic
        # pipeline; generation phases are all zero here by construction.
        prof = {"gen_load": dt, "getitem_total": dt, "produced_at": wall_time()}
        return (sample, inv, prof)
