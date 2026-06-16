from synthetic.config import CurriculumStage
from synthetic.loaders import av2_loader, womd_loader
from synthetic.dataset import SyntheticCISSDataset
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT"])
av2_data_root = project_root / "data" / "av2"
womd_data_root = project_root / "data" / "womd"

av2 = av2_loader.AV2ScenarioLoader(av2_data_root)
womd = womd_loader.WOMDScenarioLoader(womd_data_root)

def test_av2_loader():
    scenario_ids = av2.list_scenario_ids()
    assert len(scenario_ids) > 0, "AV2 loader should find some scenarios"
    lane_segments, map_features = av2.load_scenario(scenario_ids[0])
    assert len(lane_segments) > 0, "AV2 loader should produce some lane segments"
    for seg in lane_segments:
        assert seg.centerline_xy.shape[1] == 2, "Centerline should be 2D"
        assert isinstance(seg.left_boundary_feature_ids, list)
        assert isinstance(seg.right_boundary_feature_ids, list)
        for feat_id in seg.left_boundary_feature_ids + seg.right_boundary_feature_ids:
            assert feat_id in map_features, f"Feature {feat_id!r} missing from map_features"


def test_womd_loader():
    scenario_ids = womd.list_scenario_ids()
    assert len(scenario_ids) > 0, "WOMD loader should find some scenarios"
    lane_segments, map_features = womd.load_scenario(scenario_ids[0])
    assert len(lane_segments) > 0, "WOMD loader should produce some lane segments"
    for seg in lane_segments:
        assert seg.centerline_xy.shape[1] == 2, "Centerline should be 2D"
        assert isinstance(seg.left_boundary_feature_ids, list)
        assert isinstance(seg.right_boundary_feature_ids, list)
        for feat_id in seg.left_boundary_feature_ids + seg.right_boundary_feature_ids:
            assert feat_id in map_features, f"Feature {feat_id!r} missing from map_features"
        
def test_synthetic_dataset():
    dataset = SyntheticCISSDataset(loader=av2, stage=CurriculumStage.NoRandomization, samples_per_scenario=2, max_scenarios=5)
    assert len(dataset) == 10, "Dataset length should be num_scenarios * samples_per_scenario"
    item = dataset[0]
    assert isinstance(item, tuple) and len(item) == 3, "Dataset item should be a tuple of (ParsedScene, SyntheticGroundTruth, np.ndarray)"
    parsed_scene, ground_truth, pca_matrix = item
    assert pca_matrix.shape == (3, 3), "PCA matrix should be 3x3"
    

