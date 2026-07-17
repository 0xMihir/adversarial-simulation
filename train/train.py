"""
Joint training on synthetic CISS-like scenes generated from WOMD scenarios:
PTv3 backbone -> per-primitive pooling -> {ElementClass classifier, SeqGrowGraph
autoregressive lane-graph head}. Loss = cls_CE + lambda_seq * weighted seq_CE.

Every knob lives in train_config.py and is overridable by env var or CLI flag
(`--help` lists them all); see that file's docstring for A/B and resume examples.
"""

import os
import time
import json
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb

from synthetic.config import CurriculumStage
from synthetic.dataset import SyntheticCISSDataset
from synthetic.loaders import WOMDScenarioLoader
from models.model import Model
from models.primitive_decoder import NUM_ELEMENT_CLASSES
from models.seq_grow_graph.vocab import (
    CTRL_BASE,
    N_COORD_BINS,
    NODE_INDEX_BASE,
    TOK_PAD,
    TOK_TO,
    VOCAB_SIZE,
)
from transformers.optimization import get_wsd_schedule

from train_config import TrainConfig, build_config, pretty, wandb_config

# --- config ---
# Every knob lives in train_config.py (precedence: defaults < env < CLI). CFG is assigned
# once, first thing in main(); the helpers below read it instead of taking it as a param.
CFG: TrainConfig | None = None

# --- runtime state (NOT config: facts about this process, resolved during startup) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Overridden under DDP: True only on global rank 0 (gates wandb/prints/checkpointing so
# they don't happen world_size times over).
IS_MAIN = True
# Rank suffix for heartbeat/crash-dump filenames so concurrent DDP ranks don't clobber
# each other's diagnostics; stays 0 (no visible change) for single-GPU runs.
_RANK = 0
# CFG.use_wandb is the user's intent; this is that intent AND'd with the rank gate, so only
# rank 0 logs. Derived once in main() — kept separate so the frozen config keeps recording
# what was asked for, not what rank 0-ness reduced it to.
WANDB_ON = False

# Monotonic step counter for a continuous wandb x-axis across epochs.
_global_step = 0


# --- spconv crash diagnostics -------------------------------------------------
# The intermittent `merge_sort: cudaErrorIllegalAddress` (see scratch/spconv_debug/
# README.md + memory spconv-merge-sort-crash) has never reproduced in isolation, so the
# next production crash is our best evidence. These helpers make that crash maximally
# informative: dump the exact batch that was in flight, and leave a heartbeat so a silent
# SIGKILL (earlyoom at low RAM) is distinguishable from a real CUDA fault.

# earlyoom silently SIGKILLs python at low RAM and it looks IDENTICAL to a spconv crash in
# wandb. psutil is optional; degrade gracefully if it isn't installed.
try:
    import psutil as _psutil
except ImportError:
    _psutil = None


def _host_ram_available_gb() -> float | None:
    if _psutil is None:
        return None
    return _psutil.virtual_memory().available / 1e9


def _write_heartbeat(step: int) -> None:
    """Rewrite ckpt_dir/heartbeat_rank{N}.json each step. If the process VANISHES (no
    python traceback), the last heartbeat's step+timestamp shows it was killed externally
    (earlyoom / OOM), not a raised spconv error."""
    try:
        (CFG.ckpt_dir / f"heartbeat_rank{_RANK}.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "ts": time.time(),
                    "cuda_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                    "cuda_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "host_ram_avail_gb": _host_ram_available_gb(),
                }
            )
        )
    except Exception:
        pass  # heartbeat is best-effort; never let it break training


def _is_spconv_crash(err: BaseException) -> bool:
    s = str(err).lower()
    return any(k in s for k in ("cudaerror", "merge_sort", "illegal", "synchronize"))


def _cpu_batch_snapshot(data_dict: dict, labels) -> dict:
    """Take a CPU copy of the batch BEFORE the forward. Critical: once a
    cudaErrorIllegalAddress sticks, every subsequent CUDA op (incl. a .cpu() device->host
    copy) fails — so we can't move GPU tensors to host AFTER the crash. We snapshot the
    already-on-CPU loader tensors up front and only save them if the step then faults.
    Cheap: these are references/shares to tensors that already exist on the host."""
    snap = {}
    for k in ("coord", "feat", "batch", "primitive_id", "nprim_per_sample", "lane_tokens"):
        v = data_dict.get(k)
        if torch.is_tensor(v):
            snap[k] = v.detach().to("cpu", copy=False)
    if torch.is_tensor(labels):
        snap["labels"] = labels.detach().to("cpu", copy=False)
    snap["grid_size"] = data_dict.get("grid_size")
    return snap


def _dump_crash_batch(step: int, cpu_snap: dict, err: BaseException) -> Path | None:
    """Save the pre-forward CPU snapshot of the batch that crashed, so it can be replayed.
    Touches NO GPU tensors (the context is poisoned). Best-effort — a failure here must not
    mask the original error."""
    try:
        coord = cpu_snap.get("coord")
        dump = {
            "step": step,
            "err": str(err),
            "traceback": traceback.format_exc(),
            "n_points": int(coord.shape[0]) if torch.is_tensor(coord) else None,
            "grid_size": cpu_snap.get("grid_size"),
            "host_ram_avail_gb": _host_ram_available_gb(),
        }
        for k in (
            "coord",
            "feat",
            "batch",
            "primitive_id",
            "nprim_per_sample",
            "lane_tokens",
            "labels",
        ):
            if torch.is_tensor(cpu_snap.get(k)):
                dump[k] = cpu_snap[k]
        path = CFG.ckpt_dir / f"crash_dump_step{step}_rank{_RANK}.pt"
        torch.save(dump, path)
        return path
    except Exception:
        traceback.print_exc()
        return None


def _to_device(data_dict, labels):
    dd = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in data_dict.items()}
    return dd, labels.to(DEVICE)


def _sync():
    # startswith, not ==: under DDP, DEVICE is "cuda:0"/"cuda:1", so an equality check made
    # profile_sync a silent no-op on every multi-GPU run.
    if CFG.profile_sync and DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def save_checkpoint(path, model, optimizer, scheduler, global_step, epoch):
    """Full training state so a run can resume bit-for-bit on the schedule/optimizer.

    The bare model.state_dict() checkpoints from earlier runs are not enough to resume
    faithfully: AdamW's moment buffers and the WSD scheduler's step counter both drive the
    dynamics, and at step 15k the LR is at its stable peak (1e-3), not the 1e-4 a fresh
    scheduler would start from. Restoring all four keeps the resumed run on the same
    trajectory as an uninterrupted one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    """Restore state saved by save_checkpoint. Returns (global_step, epoch) to resume from.

    Tolerates a legacy checkpoint that is a bare model state_dict (the model_step*.pt files
    from the crashed run): those restore weights only and resume bookkeeping from 0, which
    will NOT reproduce the schedule — a warning is printed so it's not mistaken for a
    faithful resume.
    """
    ckpt = torch.load(path, map_location=DEVICE)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print(
            f"WARNING: {path} is a legacy model-only checkpoint. Loading weights but "
            "resuming step/optimizer/scheduler from scratch — LR schedule will NOT match "
            "an uninterrupted run, so spike dynamics may differ."
        )
        model.load_state_dict(ckpt)
        return 0, 0
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt.get("global_step", 0)
    epoch = ckpt.get("epoch", 0)
    print(f"Resumed from {path} at global_step={step}, epoch={epoch}")
    return step, epoch


def run_epoch(
    model: nn.Module | DDP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch=0,
):
    """One pass over `loader`. Train if optimizer given, else eval. Returns
    (mean_loss, accuracy).

    When PROFILE is on, accumulates per-stage wall-clock and logs the mean per-batch
    time (seconds) for each stage to wandb under prof/<split>/*:
      - data:      waiting on the DataLoader (dataset __getitem__ + collate in workers);
                   scene->arrays now happens in the workers, so this is mostly the (now
                   small) batch unpickle
      - transform: fetching the pre-assembled data_dict/labels from the batch (near-zero
                   now that the workers build the arrays; kept for continuity)
      - to_device: host -> GPU transfer
      - forward:   backbone + pooling + decoder + loss
      - backward:  loss.backward() (train only)
      - step:      optimizer.step() + zero_grad (train only)

    The `data` stage above only measures how long the main process *blocked* on the
    loader; it can't say why. When PROFILE is on, the workers also return an in-process
    breakdown of __getitem__ (see SyntheticCISSDataset.collate_fn), surfaced under
    prof_step/worker/* so you can see which phase of scene generation dominates:
      - getitem_total: full per-item worker cost (mean/max over the batch)
      - gen_load:      loader.load_scenario (proto parse; LRU miss on cold scenarios)
      - gen_synth:     element synthesis + randomization + normalization
      - gen_gt:        ground-truth extraction (endpoint adjacency + pydantic build)
    Plus a starvation signal, prof_step/starved_ratio = data / (per-batch worker cost),
    where per-batch worker cost = sum(getitem_total) / num_workers. Ratio >> 1 means the
    batch was already prefetched and you're GPU-bound; ratio ~1 means the workers can't
    refill the prefetch buffer fast enough (the sawtooth spikes).
    """
    global _global_step
    is_train = optimizer is not None
    model.train(is_train)
    split = "train" if is_train else "val"

    total_loss = 0.0
    total_correct = 0
    total_prims = 0
    total_cls_loss = 0.0
    total_seq_loss = 0.0
    total_seq_correct = 0
    total_seq_tokens = 0
    n_batches = 0

    # Bound up-front so the post-loop cleanup (del + empty_cache) is always safe, even if
    # the loader yields nothing or every batch is skipped as empty.
    batch = data_dict = labels = logits = loss = None
    cls_loss = seq_loss = seq_logits = seq_tgt = seq_pred = seq_mask = None

    stages = ("data", "transform", "to_device", "forward", "backward", "step")
    prof = {k: 0.0 for k in stages}  # epoch accumulators
    accum_count = 0  # micro-steps done in the current grad_accum window

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        if is_train:
            optimizer.zero_grad()
        t_prev = time.perf_counter()  # marks end of previous iter / start of data wait
        for step, batch in enumerate(loader):
            step_prof = {k: 0.0 for k in stages}  # this batch's per-stage times

            t = time.perf_counter()
            recv_wall = (
                time.time()
            )  # wall-clock at receipt, comparable to worker stamps
            step_prof["data"] = t - t_prev

            worker_prof = batch.get("worker_prof")  # None when profile off

            # The dataset transforms scenes -> PTv3 arrays in its workers and its collate_fn
            # assembles the batch, so the data_dict/labels arrive ready. (scene_to_lines no
            # longer runs on the main process — that was ~0.18s/step of the old `transform`.)
            data_dict, labels = batch["data_dict"], batch["labels"]
            if labels.numel() == 0:  # skip empty scenes
                t_prev = time.perf_counter()
                continue
            t, t0 = time.perf_counter(), t
            step_prof["transform"] = t - t0

            # Snapshot the batch on the HOST before it goes to GPU. If spconv faults below,
            # the CUDA context is poisoned and we can't copy device->host anymore — so the
            # dump must come from this pre-forward CPU copy, not the on-device tensors.
            crash_snap = _cpu_batch_snapshot(data_dict, labels)

            data_dict, labels = _to_device(data_dict, labels)
            _sync()
            t, t0 = time.perf_counter(), t
            step_prof["to_device"] = t - t0

            # Guard the spconv-consuming forward+backward: on the intermittent
            # merge_sort/illegal-address CUDA fault, dump the exact in-flight batch (for a
            # replayable repro) before re-raising. crash_step is the wandb x-axis step this
            # iteration would log as (_global_step is incremented only after this block).
            crash_step = _global_step + 1 if is_train else _global_step
            try:
                # (n_prim, num_classes), (B, T-1, VOCAB_SIZE) — the model teacher-forces
                # the SeqGrowGraph head on lane_tokens[:, :-1] internally.
                logits, seq_logits = model(data_dict)
                cls_loss = F.cross_entropy(logits, labels)
                seq_tgt = data_dict["lane_tokens"][:, 1:]
                raw_module = model.module if hasattr(model, "module") else model
                seq_loss = F.cross_entropy(
                    seq_logits.reshape(-1, VOCAB_SIZE),
                    seq_tgt.reshape(-1),
                    weight=raw_module.seq_head.token_weights,
                    ignore_index=TOK_PAD,
                )
                loss = cls_loss + CFG.lambda_seq * seq_loss
                _sync()
                t, t0 = time.perf_counter(), t
                step_prof["forward"] = t - t0

                grad_norm = None
                at_boundary = True
                if is_train:
                    accum_count += 1
                    at_boundary = accum_count % CFG.grad_accum == 0
                    # DDP all-reduces grads on every backward; skip it on intermediate
                    # micro-steps and only sync on the accumulation boundary.
                    use_nosync = hasattr(model, "no_sync") and not at_boundary
                    sync_ctx = model.no_sync() if use_nosync else nullcontext()
                    with sync_ctx:
                        (loss / CFG.grad_accum).backward()  # scale so grads average the big batch
                    _sync()
                    t, t0 = time.perf_counter(), t
                    step_prof["backward"] = t - t0
            except RuntimeError as e:
                if _is_spconv_crash(e):
                    path = _dump_crash_batch(crash_step, crash_snap, e)
                    ram = _host_ram_available_gb()
                    ram_s = f"{ram:.1f}GB" if ram is not None else "n/a"
                    print(
                        f"[CRASH] spconv fault at step {crash_step}: {e}\n"
                        f"[CRASH] cuda_alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB "
                        f"host_ram_avail={ram_s}",
                        flush=True,
                    )
                    if path is not None:
                        print(f"[CRASH] dumped failing batch -> {path}", flush=True)
                raise
            if is_train and at_boundary:
                # clip_grad_norm_ returns the pre-clip total norm; capture it either way so
                # a grad_norm blowup can be lined up against the loss spikes even when
                # clipping is off (max_norm=inf is a no-op that still measures the norm).
                clip_to = (
                    CFG.grad_clip_norm if CFG.grad_clip_norm is not None else float("inf")
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_to
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                _sync()
                t, t0 = time.perf_counter(), t
                step_prof["step"] = t - t0

            for k in stages:
                prof[k] += step_prof[k]

            total_loss += float(loss)
            total_correct += int((logits.argmax(dim=-1) == labels).sum())
            total_prims += int(labels.numel())
            total_cls_loss += float(cls_loss)
            total_seq_loss += float(seq_loss)
            with torch.no_grad():
                seq_pred = seq_logits.argmax(dim=-1)  # (B, T-1)
                seq_mask = seq_tgt != TOK_PAD
                total_seq_correct += int((seq_pred[seq_mask] == seq_tgt[seq_mask]).sum())
                total_seq_tokens += int(seq_mask.sum())
            n_batches += 1

            if is_train:
                _global_step += 1
                # Rewrite the heartbeat every step. If the process later VANISHES with no
                # python traceback, this file's step+ts proves it was killed externally
                # (earlyoom/OOM) rather than by the spconv RuntimeError (which dumps a
                # crash_dump_step*.pt instead). Disambiguates the two identical-looking
                # deaths in wandb. See scratch/spconv_debug/README.md.
                _write_heartbeat(_global_step)
                if WANDB_ON and step % CFG.log_every == 0:
                    log = {
                        "train/step_loss": float(loss),
                        "train/cls_loss": float(cls_loss),
                        "train/seq_loss": float(seq_loss),
                    }
                    # SeqGrowGraph token accuracy split by target token type, plus mean
                    # target length (drops toward MAX_SEQ_LEN when truncation bites).
                    with torch.no_grad():
                        for name, lo, hi in (
                            ("coord", 0, N_COORD_BINS),
                            ("index", NODE_INDEX_BASE, CTRL_BASE),
                            ("ctrl", CTRL_BASE, TOK_TO),
                            ("special", TOK_TO, TOK_PAD),
                        ):
                            m = seq_mask & (seq_tgt >= lo) & (seq_tgt < hi)
                            if int(m.sum()) > 0:
                                log[f"train/seq_acc_{name}"] = float(
                                    (seq_pred[m] == seq_tgt[m]).float().mean()
                                )
                        log["train/seq_len_mean"] = float(
                            seq_mask.sum(dim=1).float().mean()
                        )
                    if grad_norm is not None:
                        log["train/grad_norm"] = float(grad_norm)
                    # host RAM available: a downward drift toward 0 preceding a "crash" is
                    # the earlyoom signature, not the spconv fault.
                    ram = _host_ram_available_gb()
                    if ram is not None:
                        log["sys/host_ram_avail_gb"] = ram
                    log["sys/cuda_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
                    wandb.log(log, step=_global_step)

                if IS_MAIN and CFG.ckpt_every > 0 and _global_step % CFG.ckpt_every == 0:
                    ckpt_path = CFG.ckpt_dir / f"model_step{_global_step}.pt"
                    save_model = model.module if hasattr(model, "module") else model
                    save_checkpoint(
                        ckpt_path, save_model, optimizer, scheduler, _global_step, epoch
                    )
                    print(f"Saved checkpoint to {ckpt_path}")

            # High-frequency per-step stage timings (train only; the interesting signal
            # is data-load fluctuation while the GPU is fed).
            if CFG.profile and is_train and step % CFG.profile_log_every == 0:
                step_secs = sum(step_prof.values())
                msg = {f"prof_step/{k}_s": v for k, v in step_prof.items()}
                msg["prof_step/total_s"] = step_secs
                msg["prof_step/n_lines"] = int(data_dict["feat"].shape[0])

                # In-worker __getitem__ breakdown + starvation signal. Explains *why*
                # prof_step/data_s spikes: which generation phase is slow, and whether the
                # buffer was drained (workers can't keep up) or full (GPU-bound).
                if worker_prof is not None:
                    # produced_at_* are raw epoch timestamps (huge, not durations) — turn
                    # them into lags below; don't log them directly.
                    _raw_stamps = {"produced_at_oldest", "produced_at_newest"}
                    for k, v in worker_prof.items():
                        if k in _raw_stamps:
                            continue
                        # n_items is a count, not seconds; everything else is wall-clock.
                        suffix = "" if k == "n_items" else "_s"
                        msg[f"prof_step/worker/{k}{suffix}"] = v
                    # Wall-clock the batch's workers spent generating, if perfectly spread
                    # across NUM_WORKERS. Compared against how long we actually blocked.
                    per_batch_worker_s = worker_prof["getitem_total_sum"] / max(
                        CFG.num_workers, 1
                    )
                    msg["prof_step/worker/per_batch_s"] = per_batch_worker_s
                    msg["prof_step/starved_ratio"] = step_prof["data"] / max(
                        per_batch_worker_s, 1e-9
                    )

                    # IPC handoff lag: how long a finished item sat between the worker
                    # stamping produced_at and the main process receiving the batch
                    # (pickle + queue transit + unpickle). This is the time the `data`
                    # block spends *outside* generation. If the 30s spikes live here, the
                    # fix is the worker->main transfer (payload size / IPC), not prefetch.
                    #   oldest: the item that finished first and waited longest
                    #   newest: the last item to finish; batch can't dispatch before it
                    oldest = worker_prof.get("produced_at_oldest")
                    newest = worker_prof.get("produced_at_newest")
                    if oldest is not None:
                        msg["prof_step/ipc/recv_lag_oldest_s"] = recv_wall - oldest
                    if newest is not None:
                        msg["prof_step/ipc/recv_lag_newest_s"] = recv_wall - newest

                    # Pickled payload size of the fat pydantic scene+GT objects crossing
                    # the worker->main IPC queue. Unpickling this in the main process (~2s
                    # per 48MB batch, measured) happens inside next(loader) and is charged
                    # to data_s — the true bottleneck. batch_bytes/sec of unpickle predicts
                    # the recv_lag plateau depth. build_point_dict discards these objects
                    # into small numpy arrays immediately, so the payload is pure waste.
                    payload = worker_prof.get("pickle_bytes_sum")
                    if payload is not None:
                        msg["prof_step/ipc/payload_mb"] = payload / 1e6

                # Main-process cost NOT spent waiting on the loader: everything measured
                # except `data`. Confirmed cheap (~0.6s), so the bottleneck is the batch
                # unpickle hidden inside `data` (next(loader)), not main compute.
                main_busy = step_secs - step_prof["data"]
                msg["prof_step/main_busy_s"] = main_busy
                if grad_norm is not None:
                    msg["prof_step/grad_norm"] = float(grad_norm)

                if WANDB_ON:
                    wandb.log(msg, step=_global_step)

            t_prev = time.perf_counter()

    # Drop the last step's activations/graph, then hand the caching allocator's freed
    # blocks back to the driver before the caller spins up the next split. On the GB10's
    # unified memory, holding the training peak (the final graph is still referenced by
    # these locals) while validation's DataLoader prefetch (NUM_WORKERS * PREFETCH_FACTOR
    # batches) ramps up is what tipped the run into OOM right after the checkpoint save.
    batch = data_dict = labels = logits = loss = None
    cls_loss = seq_loss = seq_logits = seq_tgt = seq_pred = seq_mask = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    mean_loss = total_loss / max(n_batches, 1)
    acc = total_correct / max(total_prims, 1)
    mean_seq_loss = total_seq_loss / max(n_batches, 1)
    seq_acc = total_seq_correct / max(total_seq_tokens, 1)

    if CFG.profile:
        nb = max(n_batches, 1)
        per_batch = {f"prof/{split}/{k}_s": v / nb for k, v in prof.items()}
        per_batch[f"prof/{split}/total_s"] = sum(prof.values()) / nb
        if WANDB_ON:
            wandb.log({"epoch": epoch, **per_batch}, step=_global_step)
        if IS_MAIN:
            print(
                f"  [prof/{split}] "
                + " ".join(
                    f"{k.split('/')[-1]}={v*1e3:.1f}ms" for k, v in per_batch.items()
                )
            )

    return mean_loss, acc, mean_seq_loss, seq_acc


def main():
    global CFG, DEVICE, WANDB_ON, IS_MAIN, _RANK
    # Parse before dist.init_process_group: a bad flag then exits every rank cleanly instead
    # of hanging the survivors in the NCCL rendezvous. torchrun forwards argv to all ranks,
    # so each parses the same thing.
    CFG = build_config()

    # torchrun sets LOCAL_RANK on every launched process; its absence means single-GPU.
    ddp = "LOCAL_RANK" in os.environ
    local_rank = 0
    world_size = 1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        _RANK = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        DEVICE = f"cuda:{local_rank}"
        IS_MAIN = _RANK == 0
        if IS_MAIN:
            print(f"[ddp] world_size={world_size} device={DEVICE}")
    WANDB_ON = CFG.use_wandb and IS_MAIN

    # Echo every knob once, on rank 0, so a run's log says exactly what it ran with.
    if IS_MAIN:
        print(pretty(CFG, device=DEVICE, world_size=world_size), flush=True)

    # Ensure the checkpoint dir exists up front so the per-step heartbeat and any
    # pre-first-checkpoint crash dump have somewhere to write (a crash can happen before
    # step ckpt_every, when save_checkpoint would otherwise be the first to mkdir it).
    CFG.ckpt_dir.mkdir(parents=True, exist_ok=True)

    if CFG.pregen_dir:
        from synthetic.pregen_dataset import PregenCISSDataset

        train_set = PregenCISSDataset(CFG.pregen_dir / "training", profile=CFG.profile)
        val_set = PregenCISSDataset(CFG.pregen_dir / "validation", profile=CFG.profile)
    else:
        train_loader_src = WOMDScenarioLoader(CFG.data_root / "training")
        val_loader_src = WOMDScenarioLoader(CFG.data_root / "validation")

        train_set = SyntheticCISSDataset(
            train_loader_src, CurriculumStage.NoRandomization, profile=CFG.profile
        )
        val_set = SyntheticCISSDataset(
            val_loader_src, CurriculumStage.NoRandomization, profile=CFG.profile
        )

    # Keep workers alive between epochs so scene generators aren't rebuilt, and prefetch
    # several batches ahead so the GPU never waits on data.
    persistent = CFG.num_workers > 0
    prefetch = CFG.prefetch_factor if CFG.num_workers > 0 else None
    train_sampler = DistributedSampler(train_set, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if ddp else None
    train_loader = DataLoader(
        train_set,
        batch_size=CFG.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=CFG.num_workers,
        collate_fn=SyntheticCISSDataset.collate_fn,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=CFG.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=CFG.num_workers,
        collate_fn=SyntheticCISSDataset.collate_fn,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    model = Model().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, fused=True
    )
    # One optimizer update per grad_accum micro-batches.
    total_steps = CFG.epochs * (len(train_loader) // CFG.grad_accum)
    warmup_steps = int(total_steps * CFG.wsd_warmup_ratio)
    stable_steps = int(total_steps * CFG.wsd_stable_ratio)
    decay_steps = total_steps - warmup_steps - stable_steps

    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_decay_steps=decay_steps,
        num_training_steps=total_steps,
        min_lr_ratio=CFG.wsd_min_decay_ratio,
    )

    global _global_step
    start_epoch = 0
    if CFG.resume_from is not None:
        # Load into the raw (pre-DDP-wrap) model: DDP's constructor broadcasts rank 0's
        # weights to every rank, so this alone is enough to resume consistently under DDP.
        _global_step, start_epoch = load_checkpoint(
            CFG.resume_from, model, optimizer, scheduler
        )

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    if WANDB_ON:
        # id+resume continue an existing run's curves (see the wandb_id knob); both None =
        # a fresh run. Since _global_step was restored above, wandb picks the x-axis back up
        # where the crash left off rather than replaying from 0.
        wandb.init(
            project="adversarial-simulation",
            name=CFG.run_name,
            id=CFG.wandb_id,
            resume=CFG.wandb_resume,
            # Derived from the dataclass, so a new knob reaches wandb automatically. The
            # extras are facts about this run that aren't knobs.
            config=wandb_config(
                CFG,
                world_size=world_size,
                device=DEVICE,
                num_classes=NUM_ELEMENT_CLASSES,
                seq_vocab_size=VOCAB_SIZE,
            ),
        )

    for epoch in range(start_epoch, CFG.epochs):
        if ddp:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc, train_seq_loss, train_seq_acc = run_epoch(
            model, train_loader, optimizer, scheduler, epoch
        )
        val_loss, val_acc, val_seq_loss, val_seq_acc = run_epoch(
            model, val_loader, epoch=epoch
        )

        if IS_MAIN:
            print(
                f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
                f"seq_loss {train_seq_loss:.4f} seq_acc {train_seq_acc:.3f} "
                f"| val loss {val_loss:.4f} acc {val_acc:.3f} "
                f"seq_loss {val_seq_loss:.4f} seq_acc {val_seq_acc:.3f}"
            )
        if WANDB_ON:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "train/epoch_seq_loss": train_seq_loss,
                    "train/epoch_seq_acc": train_seq_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/seq_loss": val_seq_loss,
                    "val/seq_acc": val_seq_acc,
                },
                step=_global_step,
            )

    if WANDB_ON:
        wandb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
