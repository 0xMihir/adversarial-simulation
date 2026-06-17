from synthetic import dataset, loaders
from pathlib import Path

from synthetic.config import CurriculumStage

def test_dataset():
    loader = loaders.WOMDScenarioLoader(Path("/depot/ziran/data/shared/waymo/scenario/testing/"))
    ds = dataset.SyntheticCISSDataset(loader=loader, stage=CurriculumStage.NoRandomization, samples_per_scenario=2, max_scenarios=10)
    print(f"Dataset length: {len(ds)}")
    for i in range(len(ds)):
        scene, gt, pca = ds[i]
        print(f"Sample {i}: scene={scene}, gt={gt}, pca={pca}")

        
if __name__ == "__main__":
    test_dataset()