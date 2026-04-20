"""
Singleton texture cache — loads the prebuilt sign texture pickle once at startup.

Run annotation/scripts/batch_textures.sh to generate the pickle.

Usage (in FastAPI lifespan):
    from .services import texture_service
    texture_service.init()

Usage (in routes):
    from .services.texture_service import get_cache
"""

import pickle
from pathlib import Path

_CACHE_PATH = Path(__file__).parents[2] / "annotations" / "tex_cache.pkl"

_cache: dict = {}


def init(cache_path: Path = _CACHE_PATH) -> None:
    global _cache
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            _cache = pickle.load(f)
        print(f"[texture_service] Loaded {len(_cache)} texture entries from {cache_path}")
    else:
        _cache = {}
        print(f"[texture_service] No texture cache at {cache_path}, signs will parse live")


def get_cache() -> dict:
    return _cache
