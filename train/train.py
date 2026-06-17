from models.pointcept.models import point_transformer_v3 as ptv3
from synthetic.config import CurriculumStage
from synthetic.dataset import SyntheticCISSDataset
from synthetic.loaders import WOMDScenarioLoader
from pathlib import Path
import wandb


train_loader = WOMDScenarioLoader(Path("~/data/waymo_motion/training/").expanduser())
test_loader = WOMDScenarioLoader(Path("~/data/waymo_motion/testing/").expanduser())
val_loader = WOMDScenarioLoader(Path("~/data/waymo_motion/validation/").expanduser())

train_set = SyntheticCISSDataset(train_loader, CurriculumStage.NoRandomization)
val_set = SyntheticCISSDataset(val_loader, CurriculumStage.NoRandomization)
test_set = SyntheticCISSDataset(test_loader, CurriculumStage.NoRandomization)

