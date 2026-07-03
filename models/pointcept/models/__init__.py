from .builder import build_model
from .modules import PointModule, PointModel

# Some vendored backbones depend on optional CUDA extensions (e.g. pointops,
# torch_cluster) or heavy libs that are not installed in every environment. Import them
# best-effort so that a missing optional dependency for an unused backbone does not break
# importing the backbones we do use (e.g. PointTransformerV3). PTv3 has no such optional
# deps and is imported directly (not guarded) so failures there surface loudly.
import importlib as _importlib
import warnings as _warnings


def _try_import(_module: str) -> None:
    try:
        _mod = _importlib.import_module(_module, __name__)
    except Exception as _exc:  # noqa: BLE001 - optional backbone, keep others working
        _warnings.warn(
            f"pointcept.models: skipped optional module '{_module}' ({_exc})",
            stacklevel=2,
        )
        return
    # Re-export public names, mirroring `from .module import *`
    _names = getattr(_mod, "__all__", None)
    if _names is None:
        _names = [n for n in vars(_mod) if not n.startswith("_")]
    for _n in _names:
        globals()[_n] = getattr(_mod, _n)


# Required, no optional deps — import eagerly so problems are loud.
from .point_transformer_v3 import *  # noqa: E402,F401,F403

# Optional heads/backbones that are actually vendored in this subset. Others from the
# upstream Pointcept zoo (sparse_unet, point_transformer[_v2], spvcnn, octformer, ...)
# are not present here, so we don't list them. These three are imported best-effort
# because they pull optional deps not vendored here (`losses`, `pointops`).
for _m in (".default", ".point_prompt_training", ".sonata"):
    _try_import(_m)
