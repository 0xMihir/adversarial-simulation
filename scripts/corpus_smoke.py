"""
Corpus-wide parse smoke test for the CISS ingest pipeline.

Parses every case under --data-dir through the production path
(annotation.backend.services.faro_parser.parse_scene), classification in
cache-only mode (no MNLI model load), and writes a per-case JSON report:
parse ok/error bucket, element counts, vehicle count, extent diagonal.

Usage:
    uv run python scripts/corpus_smoke.py --data-dir data/nhtsa-ciss/output \
        --jobs 16 --out scratch/smoke_baseline.json
    uv run python scripts/corpus_smoke.py ... --baseline scratch/smoke_baseline.json

Thresholds are reported, not asserted: this is an outlier/regression miner.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FT_TO_M = 0.3048

_CLS_CACHE = {}


class StubClf:
    """
    Cache-miss fallback mirroring the HF zero-shot pipeline's return shape
    (dict for a str input, list of dicts for a list input). Every uncached
    name is deterministically classified as background so smoke runs never
    load deberta. Misses are counted so the report shows cache coverage.
    """

    LABEL_BACKGROUND = "background / decoration"

    def __init__(self):
        self.misses = 0

    def _one(self, text, candidate_labels):
        self.misses += 1
        labels = [self.LABEL_BACKGROUND] + [
            l for l in candidate_labels if l != self.LABEL_BACKGROUND
        ]
        scores = [0.99] + [0.0] * (len(labels) - 1)
        return {"sequence": text, "labels": labels, "scores": scores}

    def __call__(self, inputs, candidate_labels, **kwargs):
        if isinstance(inputs, str):
            return self._one(inputs, candidate_labels)
        return [self._one(t, candidate_labels) for t in inputs]


def _init_worker(cls_cache_path):
    global _CLS_CACHE
    if cls_cache_path and Path(cls_cache_path).exists():
        with open(cls_cache_path) as f:
            _CLS_CACHE = json.load(f)


def _extent_diag_m(scene) -> float | None:
    import numpy as np

    xs, ys = [], []
    for el in scene.elements:
        for arr in (el.resampled_xy, el.control_xy):
            if arr is not None and len(arr):
                a = np.asarray(arr, dtype=float)
                xs.append(a[:, 0])
                ys.append(a[:, 1])
    if not xs:
        return None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return float(np.hypot(x.max() - x.min(), y.max() - y.min()) * FT_TO_M)


def _smoke_one(case_dir: str) -> dict:
    case_path = Path(case_dir)
    case_id = case_path.name
    scene_path = None
    for ext in ("far", "blz"):
        p = case_path / f"scene.{ext}"
        if p.exists():
            scene_path = p
            break
    if scene_path is None:
        return {"case_id": case_id, "fmt": None, "status": "no_scene_file"}

    fmt = scene_path.suffix.lstrip(".")
    rec: dict[str, str | float | None] = {"case_id": case_id, "fmt": fmt}

    if fmt == "blz":
        # .blz reader lands in Phase 3; until then just count them.
        rec["status"] = "blz_unsupported"
        return rec

    from annotation.backend.services.faro_parser import parse_scene

    clf = StubClf()
    t0 = time.perf_counter()
    try:
        # Readers print per-scene progress; silence for corpus runs.
        with contextlib.redirect_stdout(io.StringIO()):
            scene = parse_scene(
                str(scene_path), case_id, clf_pipeline=clf, cls_cache=dict(_CLS_CACHE)
            )
    except Exception as e:
        rec["status"] = "error"
        rec["error_type"] = type(e).__name__
        rec["error_msg"] = str(e)[:200]
        tb = traceback.extract_tb(e.__traceback__)
        site = next(
            (f for f in reversed(tb) if "adversarial-simulation" in f.filename), None
        )
        if site:
            rec["error_site"] = f"{Path(site.filename).name}:{site.lineno}"
        return rec

    rec["status"] = "ok"
    rec["secs"] = round(time.perf_counter() - t0, 4)
    rec["n_elements"] = len(scene.elements)
    rec["n_roadway"] = len(scene.roadway_indices)
    rec["n_road_markings"] = len(scene.road_marking_indices)
    rec["n_other"] = len(scene.other_indices)
    rec["n_texts"] = len(scene.texts)
    rec["n_images"] = len(scene.images)
    rec["n_vehicles"] = len(scene.vehicles)
    rec["has_scalebar"] = scene.scale_bar is not None
    rec["clf_cache_misses"] = clf.misses
    diag = _extent_diag_m(scene)
    rec["extent_diag_m"] = round(diag, 2) if diag is not None else None
    return rec


def summarize(cases: dict) -> dict:
    import numpy as np

    by_status = Counter(r["status"] for r in cases.values())
    by_fmt = Counter(r.get("fmt") or "none" for r in cases.values())
    ok = [r for r in cases.values() if r["status"] == "ok"]
    errors = Counter(
        f"{r.get('error_type')}@{r.get('error_site', '?')}"
        for r in cases.values()
        if r["status"] == "error"
    )
    diags = np.array([r["extent_diag_m"] for r in ok if r.get("extent_diag_m")])
    summary = {
        "by_status": dict(by_status),
        "by_fmt": dict(by_fmt),
        "error_buckets": dict(errors.most_common(20)),
        "n_ok": len(ok),
        "vehicles_zero": sum(1 for r in ok if r["n_vehicles"] == 0),
        "cache_miss_cases": sum(1 for r in ok if r.get("clf_cache_misses", 0) > 0),
    }
    if len(diags):
        summary["extent_diag_m"] = {
            "median": round(float(np.median(diags)), 1),
            "p5": round(float(np.percentile(diags, 5)), 1),
            "p95": round(float(np.percentile(diags, 95)), 1),
            "outliers_lt10m": int((diags < 10).sum()),
            "outliers_gt2km": int((diags > 2000).sum()),
        }
    return summary


def diff_baseline(cases: dict, baseline: dict) -> dict:
    base_cases = baseline["cases"]
    regressions, fixes, vehicle_changes, bucket_changes = [], [], [], []
    for cid, rec in cases.items():
        old = base_cases.get(cid)
        if old is None:
            continue
        if old["status"] == "ok" and rec["status"] == "error":
            regressions.append(f"{cid}: {rec.get('error_type')}@{rec.get('error_site')}")
        elif old["status"] == "error" and rec["status"] == "ok":
            fixes.append(cid)
        elif old["status"] == "ok" and rec["status"] == "ok":
            if old["n_vehicles"] != rec["n_vehicles"]:
                vehicle_changes.append(f"{cid}: {old['n_vehicles']} -> {rec['n_vehicles']}")
            for k in ("n_roadway", "n_road_markings", "n_other", "n_elements"):
                if old.get(k) != rec.get(k):
                    bucket_changes.append(f"{cid}.{k}: {old.get(k)} -> {rec.get(k)}")
                    break
    return {
        "regressions": regressions,
        "n_regressions": len(regressions),
        "fixes_sample": fixes[:50],
        "n_fixes": len(fixes),
        "vehicle_changes_sample": vehicle_changes[:50],
        "n_vehicle_changes": len(vehicle_changes),
        "bucket_changes_sample": bucket_changes[:50],
        "n_bucket_changes": len(bucket_changes),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/nhtsa-ciss/output"))
    ap.add_argument("--cls-cache", default=str(REPO_ROOT / "notebooks/cls_cache.json"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--out", default=str(REPO_ROOT / "scratch/smoke_report.json"))
    ap.add_argument("--baseline", default=None, help="previous report to diff against")
    ap.add_argument("--limit", type=int, default=None, help="only first N cases")
    args = ap.parse_args()

    case_dirs = sorted(
        str(p) for p in Path(args.data_dir).iterdir() if p.is_dir()
    )
    if args.limit:
        case_dirs = case_dirs[: args.limit]
    print(f"Smoking {len(case_dirs)} cases with {args.jobs} workers...")

    t0 = time.time()
    cases = {}
    with Pool(args.jobs, initializer=_init_worker, initargs=(args.cls_cache,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_smoke_one, case_dirs, chunksize=32)):
            cases[rec["case_id"]] = rec
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(case_dirs)} ({time.time() - t0:.0f}s)")

    report = {
        "meta": {
            "data_dir": args.data_dir,
            "n_cases": len(cases),
            "wall_secs": round(time.time() - t0, 1),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": summarize(cases),
        "cases": cases,
    }

    if args.baseline:
        with open(args.baseline) as f:
            report["baseline_diff"] = diff_baseline(cases, json.load(f))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report["summary"], indent=2))
    if "baseline_diff" in report:
        d = report["baseline_diff"]
        print(
            f"vs baseline: {d['n_regressions']} regressions, {d['n_fixes']} fixes, "
            f"{d['n_vehicle_changes']} vehicle-count changes, "
            f"{d['n_bucket_changes']} bucket changes"
        )
        for r in d["regressions"][:20]:
            print("  REGRESSION", r)
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
