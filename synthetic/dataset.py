"""
SyntheticCISSDataset — PyTorch Dataset wrapping SyntheticSceneGenerator.

Each item is a (ParsedScene, SyntheticGroundTruth, inverse_transform) triple.
The ParsedScene is in normalized coordinates; inverse_transform is the (3,3)
affine that maps back to original metric coordinates.

Downstream graph construction (ParsedScene → PyG Data) is a separate transform
step outside this module.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:
    # Allow importing this module without torch (e.g., in unit tests)
    Dataset = object  # type: ignore[assignment,misc]

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
        stage: curriculum stage (A, B, or C)
        curriculum_cfg: optional custom CurriculumConfig; defaults to standard A/B/C
        samples_per_scenario: number of independent crops per scenario (different index seeds)
        apply_pca: passed to SyntheticSceneGenerator
        base_seed: base randomization seed
        max_scenarios: cap on scenario count for dev/testing (None = all)

    Item shape:
        (ParsedScene, SyntheticGroundTruth, np.ndarray[3,3])
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
    ) -> None:
        self._stage = stage
        self.samples_per_scenario = samples_per_scenario

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

    def __getitem__(
        self, idx: int
    ) -> tuple[ParsedScene, SyntheticGroundTruth, np.ndarray]:
        scenario_idx = idx // self.samples_per_scenario
        sample_idx = idx % self.samples_per_scenario
        scenario_id = self._scenario_ids[scenario_idx]
        scene, gt = self._generator.generate(scenario_id, self._stage, sample_idx)
        inv_transform = np.array(gt.inverse_transform.values, dtype=np.float64)
        return scene, gt, inv_transform

    @staticmethod
    def collate_fn(
        batch: list[tuple[ParsedScene, SyntheticGroundTruth, np.ndarray]],
    ) -> dict[str, Any]:
        """
        Collate a list of items into a dict.
        Scenes are NOT converted to tensors here — that's a downstream transform.
        inverse_transforms is stacked into (B, 3, 3).
        """
        scenes, gts, inv_transforms = zip(*batch)
        return {
            "scenes": list(scenes),
            "ground_truths": list(gts),
            "inverse_transforms": np.stack(inv_transforms),  # (B, 3, 3)
        }
