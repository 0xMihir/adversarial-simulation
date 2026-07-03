"""
SyntheticCISSDataset — PyTorch Dataset wrapping SyntheticSceneGenerator.

By default (transform_in_worker=True) each item is a (SampleArrays, inverse_transform)
pair: the scene has already been turned into small PTv3 numpy arrays *in the worker*, so
only ~tens of KB cross the IPC boundary instead of the ~1.5MB pydantic ParsedScene +
SyntheticGroundTruth. Profiling proved that pydantic pickle/unpickle across the worker
queue (~2s per 48MB batch, unpickled serially in the main process, charged to data_s) was
the training bottleneck — the GPU sat idle waiting on it. See models.data_transform.

With transform_in_worker=False the item is the old (ParsedScene, SyntheticGroundTruth,
inverse_transform) triple, kept for tests / downstream consumers that need the raw scene.
"""
from __future__ import annotations

import pickle
from time import perf_counter, time as wall_time
from typing import Any

import numpy as np

from torch.utils.data import Dataset

from schema.scene import ParsedScene
from synthetic.config import CurriculumConfig, CurriculumStage
from synthetic.generator import SyntheticSceneGenerator
from synthetic.loaders.base import ScenarioLoader
from synthetic.schema import SyntheticGroundTruth


class SyntheticCISSDataset(Dataset):
    """
    PyTorch Dataset over synthetic CISS-like scenes.

    Args:
        loader: AV2ScenarioLoader, WOMDScenarioLoader, or any ScenarioLoader
        stage: curriculum stage (No randomization, partial, or full)
        curriculum_cfg: optional custom CurriculumConfig; defaults to standard config if None
        samples_per_scenario: number of independent crops per scenario (different index seeds)
        apply_pca: passed to SyntheticSceneGenerator
        base_seed: base randomization seed
        max_scenarios: cap on scenario count for dev/testing (None = all)
        transform_in_worker: if True (default), run scene->PTv3-array conversion inside
            __getitem__ so only small arrays cross IPC; if False, return the raw pydantic
            (ParsedScene, SyntheticGroundTruth) triple.

    Item shape:
        transform_in_worker=True:  (SampleArrays, np.ndarray[3,3], prof|None)
        transform_in_worker=False: (ParsedScene, SyntheticGroundTruth, np.ndarray[3,3], prof|None)
    """

    def __init__(
        self,
        loader: ScenarioLoader,
        stage: CurriculumStage,
        curriculum_cfg: CurriculumConfig | None = None,
        samples_per_scenario: int = 4,
        apply_pca: bool = True,
        base_seed: int = 42,
        max_scenarios: int | None = None,
        profile: bool = False,
        transform_in_worker: bool = True,
    ) -> None:
        self._stage = stage
        self.samples_per_scenario = samples_per_scenario
        # When True, __getitem__ times its phases (in the worker process) and returns
        # them as a trailing prof element; collate_fn aggregates them. See train.py.
        self.profile = profile
        # When True, convert scene->arrays in the worker to shrink the IPC payload.
        self.transform_in_worker = transform_in_worker

        all_ids = loader.list_scenario_ids()
        if max_scenarios is not None:
            all_ids = all_ids[:max_scenarios]
        self._scenario_ids = all_ids

        source = getattr(loader, "source_dataset", "av2")

        self._generator = SyntheticSceneGenerator(
            loader=loader,
            curriculum_cfg=curriculum_cfg,
            apply_pca=apply_pca,
            seed=base_seed,
            source_dataset=source,
        )

    def __len__(self) -> int:
        return len(self._scenario_ids) * self.samples_per_scenario

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        scenario_idx = idx // self.samples_per_scenario
        sample_idx = idx % self.samples_per_scenario
        scenario_id = self._scenario_ids[scenario_idx]

        prof: dict[str, float] | None = {} if self.profile else None
        t0 = perf_counter() if self.profile else 0.0

        scene, gt = self._generator.generate(
            scenario_id, self._stage, sample_idx, prof=prof
        )
        inv_transform = np.array(gt.inverse_transform.values, dtype=np.float64)

        if self.transform_in_worker:
            # Convert to small PTv3 arrays here so only they (not the pydantic scene+gt)
            # cross the IPC boundary. Import lazily to keep torch/model deps out of the
            # raw-scene import path (e.g. proto-only unit tests).
            from models.data_transform.scene_to_point import scene_to_arrays

            t_tf = perf_counter() if self.profile else 0.0
            payload: Any = scene_to_arrays(scene, gt)
            if prof is not None:
                prof["gen_transform"] = perf_counter() - t_tf
            item: tuple[Any, ...] = (payload, inv_transform)
        else:
            item = (scene, gt, inv_transform)

        if prof is None:
            return (*item, None)

        prof["getitem_total"] = perf_counter() - t0
        # Pickled size of exactly what crosses the worker->main IPC queue (the SampleArrays
        # when transforming in the worker, else the fat pydantic scene+gt). This is the
        # number the whole investigation turned on; expect it to drop ~100x once
        # transform_in_worker is on. Costs a real pickle.dumps (~ms), gated behind profile.
        prof["pickle_bytes"] = float(
            len(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL))
        )
        # Epoch wall-clock (comparable across processes, unlike perf_counter) at the instant
        # this worker finished the item. The main process subtracts this on receipt to get
        # the IPC handoff lag (pickle + queue transit + unpickle).
        prof["produced_at"] = wall_time()
        return (*item, prof)

    @staticmethod
    def collate_fn(batch: list[tuple[Any, ...]]) -> dict[str, Any]:
        """
        Collate a list of items into a dict.

        Items have a trailing prof element (dict|None). The rest is either:
          - (SampleArrays, inv_transform)            when transform_in_worker=True
          - (ParsedScene, GroundTruth, inv_transform) when transform_in_worker=False

        In the worker-transform case the batch is assembled here into a ready PTv3
        data_dict + labels (only cheap numpy concatenation — the scene_to_lines work
        already happened in the workers), so the main loop skips build_point_dict.

        When items carry per-item timing dicts (profiling on), they're aggregated into
        `worker_prof` with, per stage, the batch sum/mean/max (seconds). max exposes the
        slow-tail item — the one item that holds up the whole batch and drains the
        prefetch buffer. Absent when profiling is off.
        """
        *payloads_by_field, profs = zip(*(tuple(item) for item in batch))
        out: dict[str, Any] = {}

        if len(payloads_by_field) == 2:
            # (SampleArrays, inv_transform): assemble the PTv3 batch here.
            from models.data_transform.scene_to_point import batch_point_dict

            samples, inv_transforms = payloads_by_field
            data_dict, labels = batch_point_dict(list(samples))
            out["data_dict"] = data_dict
            out["labels"] = labels
            out["inverse_transforms"] = np.stack(inv_transforms)  # (B, 3, 3)
        else:
            # (ParsedScene, GroundTruth, inv_transform): keep raw scenes for the caller.
            scenes, gts, inv_transforms = payloads_by_field
            out["scenes"] = list(scenes)
            out["ground_truths"] = list(gts)
            out["inverse_transforms"] = np.stack(inv_transforms)  # (B, 3, 3)

        item_profs = [p for p in profs if p is not None]
        if item_profs:
            n = len(item_profs)
            # produced_at is an absolute epoch timestamp, not a duration; sum/mean are
            # meaningless. Carry the oldest and newest raw stamps so the main process can
            # turn them into IPC lag on receipt.
            duration_keys = set().union(*(p.keys() for p in item_profs)) - {"produced_at"}
            agg: dict[str, float] = {}
            for k in duration_keys:
                vals = [p.get(k, 0.0) for p in item_profs]
                agg[f"{k}_sum"] = float(sum(vals))
                agg[f"{k}_mean"] = float(sum(vals) / n)
                agg[f"{k}_max"] = float(max(vals))
            produced = [p["produced_at"] for p in item_profs if "produced_at" in p]
            if produced:
                # oldest finished first -> longest wait in the handoff; newest finished last
                # -> the batch couldn't be dispatched before this stamp.
                agg["produced_at_oldest"] = float(min(produced))
                agg["produced_at_newest"] = float(max(produced))
            agg["n_items"] = float(n)
            out["worker_prof"] = agg

        return out
