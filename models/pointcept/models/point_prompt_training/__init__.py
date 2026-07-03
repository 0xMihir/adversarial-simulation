# PDNorm is required by PointTransformerV3; import it eagerly.
from .prompt_driven_normalization import PDNorm

# The full PPT variants depend on pointcept.models.losses, which is not vendored in this
# subset. Import them best-effort so PDNorm (and thus PTv3) still loads without them.
import warnings as _warnings

for _m in (
    "point_prompt_training_v1m1_language_guided",
    "point_prompt_training_v1m2_decoupled",
    "point_prompt_training_v1m3_neo",
):
    try:
        _mod = __import__(f"{__name__}.{_m}", fromlist=["*"])
    except Exception as _exc:  # noqa: BLE001 - optional, keep PDNorm working
        _warnings.warn(
            f"{__name__}: skipped optional module '{_m}' ({_exc})", stacklevel=2
        )
        continue
    _names = getattr(_mod, "__all__", None) or [
        n for n in vars(_mod) if not n.startswith("_")
    ]
    for _n in _names:
        globals()[_n] = getattr(_mod, _n)
