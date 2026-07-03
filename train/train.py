"""
Train the primitive classifier: PTv3 backbone -> per-primitive pooling -> ElementClass
head, on synthetic CISS-like scenes generated from WOMD scenarios.
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb

from synthetic.config import CurriculumStage
from synthetic.dataset import SyntheticCISSDataset
from synthetic.loaders import WOMDScenarioLoader
from models.model import Model
from models.primitive_decoder import NUM_ELEMENT_CLASSES

from schedules import wsd_cosine_decay_scheduler

# --- config ---
DATA_ROOT = Path("~/data/waymo_motion").expanduser()
BATCH_SIZE = 128
NUM_WORKERS = 16
PREFETCH_FACTOR = 16  # batches prefetched per worker
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY = 20
USE_WANDB = True

# Profiling: log per-stage wall-clock (seconds) to wandb to find Python bottlenecks.
PROFILE = True
# PROFILE_SYNC forces a cuda.synchronize() after every GPU stage so forward/backward
# timings reflect real compute instead of async kernel-launch time. This SERIALIZES the
# GPU (blocks the CPU from queuing ahead), inflates idle, and slows real training — it is
# a measurement tool, not a training setting. Now that the data loader is fixed (data_s
# ~0 in steady state), keep it OFF for real runs; flip on only to re-attribute per-stage
# GPU time. Note per-stage forward/backward numbers are meaningless (async) while off.
PROFILE_SYNC = False
PROFILE_LOG_EVERY = 20

# Monotonic step counter for a continuous wandb x-axis across epochs.
_global_step = 0


def _to_device(data_dict, labels):
    dd = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in data_dict.items()}
    return dd, labels.to(DEVICE)


def _sync():
    if PROFILE_SYNC and DEVICE == "cuda":
        torch.cuda.synchronize()


def run_epoch(
    model,
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
    where per-batch worker cost = sum(getitem_total) / NUM_WORKERS. Ratio >> 1 means the
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
    n_batches = 0

    stages = ("data", "transform", "to_device", "forward", "backward", "step")
    prof = {k: 0.0 for k in stages}  # epoch accumulators

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        t_prev = time.perf_counter()  # marks end of previous iter / start of data wait
        for step, batch in enumerate(loader):
            step_prof = {k: 0.0 for k in stages}  # this batch's per-stage times

            t = time.perf_counter()
            recv_wall = (
                time.time()
            )  # wall-clock at receipt, comparable to worker stamps
            step_prof["data"] = t - t_prev

            worker_prof = batch.get("worker_prof")  # None when PROFILE off

            # The dataset transforms scenes -> PTv3 arrays in its workers and its collate_fn
            # assembles the batch, so the data_dict/labels arrive ready. (scene_to_lines no
            # longer runs on the main process — that was ~0.18s/step of the old `transform`.)
            data_dict, labels = batch["data_dict"], batch["labels"]
            if labels.numel() == 0:  # skip empty scenes
                t_prev = time.perf_counter()
                continue
            t, t0 = time.perf_counter(), t
            step_prof["transform"] = t - t0

            data_dict, labels = _to_device(data_dict, labels)
            _sync()
            t, t0 = time.perf_counter(), t
            step_prof["to_device"] = t - t0

            logits = model(data_dict)  # (n_prim, num_classes)
            loss = F.cross_entropy(logits, labels)
            _sync()
            t, t0 = time.perf_counter(), t
            step_prof["forward"] = t - t0

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                _sync()
                t, t0 = time.perf_counter(), t
                step_prof["backward"] = t - t0

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                _sync()
                t, t0 = time.perf_counter(), t
                step_prof["step"] = t - t0

            for k in stages:
                prof[k] += step_prof[k]

            total_loss += float(loss)
            total_correct += int((logits.argmax(dim=-1) == labels).sum())
            total_prims += int(labels.numel())
            n_batches += 1

            if is_train:
                _global_step += 1
                if USE_WANDB and step % LOG_EVERY == 0:
                    wandb.log({"train/step_loss": float(loss)}, step=_global_step)

            # High-frequency per-step stage timings (train only; the interesting signal
            # is data-load fluctuation while the GPU is fed).
            if PROFILE and is_train and step % PROFILE_LOG_EVERY == 0:
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
                        NUM_WORKERS, 1
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

                if USE_WANDB:
                    wandb.log(msg, step=_global_step)

            t_prev = time.perf_counter()

    mean_loss = total_loss / max(n_batches, 1)
    acc = total_correct / max(total_prims, 1)

    if PROFILE:
        nb = max(n_batches, 1)
        per_batch = {f"prof/{split}/{k}_s": v / nb for k, v in prof.items()}
        per_batch[f"prof/{split}/total_s"] = sum(prof.values()) / nb
        if USE_WANDB:
            wandb.log({"epoch": epoch, **per_batch}, step=_global_step)
        print(
            f"  [prof/{split}] "
            + " ".join(
                f"{k.split('/')[-1]}={v*1e3:.1f}ms" for k, v in per_batch.items()
            )
        )

    return mean_loss, acc


def main():
    train_loader_src = WOMDScenarioLoader(DATA_ROOT / "training")
    val_loader_src = WOMDScenarioLoader(DATA_ROOT / "validation")

    train_set = SyntheticCISSDataset(
        train_loader_src, CurriculumStage.NoRandomization, profile=PROFILE
    )
    val_set = SyntheticCISSDataset(
        val_loader_src, CurriculumStage.NoRandomization, profile=PROFILE
    )

    # Keep workers alive between epochs so scene generators aren't rebuilt, and prefetch
    # several batches ahead so the GPU never waits on data.
    persistent = NUM_WORKERS > 0
    prefetch = PREFETCH_FACTOR if NUM_WORKERS > 0 else None
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=SyntheticCISSDataset.collate_fn,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=SyntheticCISSDataset.collate_fn,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    model = Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = wsd_cosine_decay_scheduler(
        optimizer, warmup_steps=100, total_steps=EPOCHS * len(train_loader)
    )

    if USE_WANDB:
        wandb.init(
            project="adversarial-simulation",
            config={
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "epochs": EPOCHS,
                "num_classes": NUM_ELEMENT_CLASSES,
            },
        )

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, scheduler, epoch
        )
        val_loss, val_acc = run_epoch(model, val_loader, epoch=epoch)

        print(
            f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.3f}"
        )
        if USE_WANDB:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                },
                step=_global_step,
            )

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
