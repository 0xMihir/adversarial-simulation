"""
Singleton classifier service — loads the MNLI pipeline once at startup
and persists cls_cache to disk across restarts.

Usage (in FastAPI lifespan):
    from .services import clf_service
    clf_service.init()

Usage (in routes):
    from .services.clf_service import get_pipeline, get_cache, flush
"""

import atexit
import pickle
from pathlib import Path

_CACHE_PATH = Path(__file__).parents[2] / "annotations" / "clf_cache.pkl"
_MODEL_ID = "sileod/deberta-v3-large-tasksource-nli"

_pipeline = None
_cache: dict = {}
_cache_path_used: Path = _CACHE_PATH


def init(cache_path: Path = _CACHE_PATH) -> None:
    global _pipeline, _cache, _cache_path_used
    from transformers import pipeline as hf_pipeline

    _cache_path_used = cache_path

    print(f"[clf_service] Loading model {_MODEL_ID} ...")
    _pipeline = hf_pipeline(
        "zero-shot-classification",
        model=_MODEL_ID,
        local_files_only=True,
    )

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            _cache = pickle.load(f)
        print(f"[clf_service] Loaded {len(_cache)} cached names from {cache_path}")
    else:
        _cache = {}
        print(f"[clf_service] No cache file found at {cache_path}, starting fresh")

    atexit.register(_flush_to, cache_path)


def _flush_to(cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(_cache, f)
    print(f"[clf_service] Saved {len(_cache)} names to {cache_path}")


def flush() -> None:
    """Persist cache to disk immediately. Call after each process_case."""
    _flush_to(_cache_path_used)


def get_pipeline():
    return _pipeline


def get_cache() -> dict:
    return _cache
