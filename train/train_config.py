"""
Config knobs for train/train.py. Precedence: defaults < env < CLI.

Every knob is declared ONCE in TrainConfig below; the argparse flags and the env
lookups are derived from the dataclass fields, so adding a knob is a one-line
change. The env var for a knob is always its field name uppercased
(batch_size -> BATCH_SIZE); the CLI flag is always kebab-case (--batch-size).

Run-specific knobs are env-overridable so an A/B pair launches from one committed
file; the CLI overrides env for one-off runs:

    CKPT_DIR=ck/a GRAD_CLIP_NORM=none RUN_NAME=a  torchrun ... train/train.py
    CKPT_DIR=ck/b GRAD_CLIP_NORM=1.0  RUN_NAME=b  torchrun ... train/train.py
    uv run python3 train/train.py --grad-clip-norm 1.0 --no-profile

Resuming a crashed run — restore the training state AND keep the same wandb chart
(the run id is the tail of `readlink wandb/latest-run`):

    uv run python3 train/train.py
        --resume-from checkpoints/model_step15000.pt
        --wandb-id pcm9t81t --wandb-resume must
"""

# NOTE: do NOT add `from __future__ import annotations` to this file. It turns the dataclass
# field types into strings, which silently breaks _resolve()'s union handling (it inspects
# live `X | None` types, not their names). This is a comment rather than part of the
# docstring above because argparse prints that docstring as --help.

import argparse
import os
import types
import typing
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path


def knob(default, help: str = ""):
    """Declare a config knob: its default plus the help shown by --help."""
    return field(default=default, metadata={"help": help})


# --- value parsers (shared by the env and CLI paths, so both accept the same
# --- spellings and the same sentinels) -----------------------------------------


def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"expected a boolean like 1/0/true/false, got {s!r}")


def _parse_path(s: str) -> Path:
    return Path(s).expanduser()


def _opt(inner):
    """X | None parser: "none" or "" means None."""

    def parse(s: str):
        v = s.strip()
        return None if v.lower() in ("none", "") else inner(v)

    return parse


def _parse_clip(s: str) -> float | None:
    """grad_clip_norm's sentinel: none/""/0 all mean "clipping off"."""
    v = s.strip().lower()
    return None if v in ("none", "", "0") else float(v)


_SCALAR = {int: int, float: float, str: str, bool: _parse_bool, Path: _parse_path}

# wandb.init(resume=...) accepts exactly these; see the wandb_resume knob. Typed as a
# Literal so the checker can see that __post_init__'s validation makes the value safe to
# hand straight to wandb.init(resume=...), which is annotated with the same set.
WandbResume = typing.Literal["must", "allow", "never", "auto"]
WANDB_RESUME_MODES = typing.get_args(WandbResume)


@dataclass(frozen=True)
class TrainConfig:
    # --- data ---
    data_root: Path = knob(
        Path("~/data/waymo_motion").expanduser(), "WOMD scenario root"
    )
    # Root of train/pregen.py output. The on-the-fly synthetic pipeline costs ~50 CPU-s per
    # batch vs the H100's ~0.75s/step — more generation than the 24-cores-per-GPU allocation
    # can sustain (GPU duty-cycled at ~40-50%). Pre-generated shards replay the identical
    # NoRandomization samples at memcpy cost.
    pregen_dir: Path | None = knob(
        None, "pregen shard root; unset = generate scenes on the fly"
    )
    # Per-GPU batch size (DDP replicates this per rank; effective batch = batch_size * world_size).
    batch_size: int = knob(128, "per-GPU batch size")
    num_workers: int = knob(16, "DataLoader workers per rank")
    prefetch_factor: int = knob(16, "batches prefetched per worker")

    # --- optimization ---
    lr: float = knob(3e-4, "peak learning rate")
    weight_decay: float = knob(0.05, "AdamW weight decay")
    epochs: int = knob(50, "training epochs")
    grad_accum: int = knob(1, "micro-steps per optimizer update")
    wsd_warmup_ratio: float = knob(0.05, "WSD warmup fraction of total steps")
    wsd_stable_ratio: float = knob(0.3, "WSD stable fraction of total steps")
    wsd_min_decay_ratio: float = knob(0.1, "WSD min LR ratio at end of decay")
    # Weight of the SeqGrowGraph lane-graph next-token loss in the joint objective
    # L = cls_CE + lambda_seq * seq_CE.
    lambda_seq: float = knob(1.0, "SeqGrowGraph next-token loss weight")
    # Gradient clipping. None disables it (original behavior that produced the spikes). Set
    # to a float (e.g. 1.0) to clip grad L2-norm before optimizer.step(). We log the
    # pre-clip grad norm regardless so a spike in grad_norm can be correlated with the loss
    # spikes even when clipping is off.
    grad_clip_norm: float | None = knob(
        10.0, "grad-norm clip; none/0 = off (reproduce spikes), e.g. 1.0 = fix"
    )

    # --- checkpointing ---
    ckpt_dir: Path = knob(Path("checkpoints"), "where checkpoints go (A/B in separate dirs)")
    ckpt_every: int = knob(1000, "steps between checkpoints; 0 = off")
    # Path to a checkpoint saved by save_checkpoint() (model+optim+sched+step), or None to
    # start fresh. When set, training continues from the saved _global_step with the
    # optimizer moments and WSD scheduler position restored, so the LR schedule and Adam
    # state are exactly what they'd be in an uninterrupted run — required to faithfully
    # reproduce step-dependent dynamics like the post-15k loss spikes.
    resume_from: Path | None = knob(None, "checkpoint to resume from; unset = fresh start")

    # --- logging / profiling ---
    use_wandb: bool = knob(True, "log to wandb (rank 0 only regardless)")
    run_name: str | None = knob(None, "wandb run label (tell an A/B pair apart)")
    # Resuming a crashed run: pair these with resume_from so the restored optimizer/scheduler
    # keep logging to the SAME wandb run instead of opening a disconnected new chart.
    # wandb_id is the run id to continue (the tail of wandb/latest-run, e.g. "pcm9t81t");
    # wandb_resume is passed straight to wandb.init(resume=...):
    #   must   = the run must exist (what you want when resuming a known crash — a typo in
    #            the id fails loudly instead of silently starting a fresh run)
    #   allow  = resume if it exists, else start it
    #   never  = error if it exists
    #   auto   = let wandb resume the previous run from this dir if it crashed
    # Unset (None) = always start a new run. Since _global_step is restored from the
    # checkpoint, wandb continues the existing curves at the right x — no gap, no rewind.
    wandb_id: str | None = knob(None, "wandb run id to resume (pairs with --resume-from)")
    wandb_resume: WandbResume | None = knob(
        None, "wandb resume mode: must/allow/never/auto; unset = new run"
    )
    log_every: int = knob(20, "steps between wandb scalar logs")
    # Log per-stage wall-clock (seconds) to wandb to find Python bottlenecks.
    profile: bool = knob(True, "log per-stage wall-clock to wandb")
    # Forces a cuda.synchronize() after every GPU stage so forward/backward timings reflect
    # real compute instead of async kernel-launch time. This SERIALIZES the GPU (blocks the
    # CPU from queuing ahead), inflates idle, and slows real training — it is a measurement
    # tool, not a training setting. Now that the data loader is fixed (data_s ~0 in steady
    # state), keep it OFF for real runs; flip on only to re-attribute per-stage GPU time.
    # Note per-stage forward/backward numbers are meaningless (async) while off.
    profile_sync: bool = knob(False, "cuda.synchronize() each stage (measurement only)")
    profile_log_every: int = knob(20, "steps between per-step profile logs")

    def __post_init__(self) -> None:
        assert self.batch_size > 0, f"batch_size={self.batch_size} must be > 0"
        assert self.grad_accum > 0, f"grad_accum={self.grad_accum} must be > 0"
        assert self.epochs > 0, f"epochs={self.epochs} must be > 0"
        assert self.num_workers >= 0, f"num_workers={self.num_workers} must be >= 0"
        assert self.prefetch_factor > 0, (
            f"prefetch_factor={self.prefetch_factor} must be > 0"
        )
        assert self.grad_clip_norm is None or self.grad_clip_norm > 0, (
            f"grad_clip_norm={self.grad_clip_norm} must be > 0 or None"
        )
        assert self.wsd_warmup_ratio + self.wsd_stable_ratio <= 1.0, (
            f"wsd_warmup_ratio + wsd_stable_ratio = "
            f"{self.wsd_warmup_ratio + self.wsd_stable_ratio} > 1.0 leaves no decay phase"
        )
        assert self.wandb_resume in (None, *WANDB_RESUME_MODES), (
            f"wandb_resume={self.wandb_resume!r} must be one of "
            f"{'/'.join(WANDB_RESUME_MODES)}, or unset"
        )
        # "auto" resumes from the local wandb dir and needs no id; every other mode targets a
        # specific run, so an id is required — without one wandb would quietly open a NEW run,
        # which is exactly the disconnected-chart outcome resuming is meant to avoid.
        assert not (
            self.wandb_resume not in (None, "auto") and self.wandb_id is None
        ), f"wandb_resume={self.wandb_resume!r} needs wandb_id (the run id to resume)"
        # An id with no mode means wandb.init(id=...) with resume unset: it errors if the run
        # exists and silently creates it otherwise. Neither is what "resume this run" means.
        assert not (self.wandb_id is not None and self.wandb_resume is None), (
            "wandb_id is set but wandb_resume is not — pass --wandb-resume must to continue "
            "that run (or drop --wandb-id to start a new one)"
        )


# Env vars that used to work and no longer do. Silently ignoring GRAD_CLIP would leave
# clipping OFF while the launch command says it's on — a training regression that looks
# nothing like a config bug. Fail loudly instead.
_RETIRED_ENV = {"GRAD_CLIP": "GRAD_CLIP_NORM"}


def _resolve(f) -> tuple[typing.Callable, bool, tuple | None]:
    """(parser, is_bool, choices) for a dataclass field, unwrapping X | None unions."""
    t = f.type
    if isinstance(t, str):  # only if someone adds `from __future__ import annotations`
        t = typing.get_type_hints(TrainConfig)[f.name]
    # Two spellings reach here: `int | None` builds a types.UnionType, but a Literal alias
    # (`WandbResume | None`) builds a typing.Union. Accept both.
    optional = typing.get_origin(t) in (types.UnionType, typing.Union)
    if optional:
        inner = [a for a in typing.get_args(t) if a is not type(None)]
        assert len(inner) == 1, f"{f.name}: only X | None unions are supported"
        t = inner[0]
    # A Literal enumerates its own valid values — hand them to argparse as choices.
    if typing.get_origin(t) is typing.Literal:
        choices = typing.get_args(t)
        return (_opt(str) if optional else str), False, choices
    # grad_clip_norm's "0" means off, not 0.0 — it needs the sentinel parser.
    if f.name == "grad_clip_norm":
        return _parse_clip, False, None
    if optional:
        return _opt(_SCALAR[t]), False, None
    return _SCALAR[t], t is bool, None


def _env_name(f) -> str:
    return f.name.upper()


def _reject_retired_env(environ) -> None:
    for old, new in _RETIRED_ENV.items():
        if old in environ:
            raise SystemExit(
                f"{old} is no longer read (renamed to {new}); update the launch command. "
                f"Leaving it set would silently disable gradient clipping."
            )


def build_config(argv=None, environ=None) -> TrainConfig:
    """Resolve knobs with defaults < env < CLI.

    Every argparse default is None — the "not specified" sentinel. Without it we couldn't
    tell "user passed the default" from "user passed nothing", and env could never win.
    """
    environ = os.environ if environ is None else environ
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for f in fields(TrainConfig):
        parse, is_bool, choices = _resolve(f)
        help_ = (
            f"{f.metadata.get('help', '')} "
            f"[env: {_env_name(f)}] (default: {f.default})"
        )
        flag = "--" + f.name.replace("_", "-")
        if is_bool:
            ap.add_argument(
                flag,
                dest=f.name,
                default=None,
                action=argparse.BooleanOptionalAction,
                help=help_,
            )
        else:
            ap.add_argument(
                flag,
                dest=f.name,
                default=None,
                type=parse,
                choices=choices,
                metavar=f.name.upper(),
                help=help_,
            )
    args = ap.parse_args(argv)

    _reject_retired_env(environ)

    vals = {}
    for f in fields(TrainConfig):
        parse, _, _ = _resolve(f)
        cli = getattr(args, f.name)
        if cli is not None:
            vals[f.name] = cli
            continue
        raw = environ.get(_env_name(f))
        if raw is not None:
            try:
                vals[f.name] = parse(raw)
            except ValueError as e:
                raise SystemExit(f"bad env {_env_name(f)}={raw!r}: {e}")
    return TrainConfig(**vals)


def _jsonable(v):
    """wandb stores config as JSON; Path and Enum don't survive the round trip."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, Enum):
        return v.name
    return v


def wandb_config(cfg: TrainConfig, **extra) -> dict:
    """Every knob plus derived extras (world_size, device, model constants), JSON-safe.

    Deriving this from the dataclass means a new knob reaches wandb automatically — there
    is no hand-maintained key list to drift out of sync with the config.
    """
    d = {k: _jsonable(v) for k, v in asdict(cfg).items()}
    d.update({k: _jsonable(v) for k, v in extra.items()})
    return d


def pretty(cfg: TrainConfig, **extra) -> str:
    """Aligned table of every knob, printed at startup on rank 0."""
    rows = [(k, _jsonable(v)) for k, v in asdict(cfg).items()]
    rows += [(k, _jsonable(v)) for k, v in sorted(extra.items())]
    w = max(len(k) for k, _ in rows)
    body = "\n".join(f"  {k:<{w}}  {v}" for k, v in rows)
    return f"--- train config {'-' * 46}\n{body}\n{'-' * 62}"
