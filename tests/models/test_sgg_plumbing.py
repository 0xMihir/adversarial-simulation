"""Integration: real generated (scene, gt) -> SampleArrays with valid lane_tokens,
ragged batching in batch_point_dict, and Model returning both heads' logits."""
import subprocess
import sys

import numpy as np
import pytest
import torch

from synthetic.config import CurriculumStage
from synthetic.generator import SyntheticSceneGenerator

from models.data_transform.scene_to_point import (
    SampleArrays,
    batch_point_dict,
    lane_tokens_for_scene,
    scene_to_arrays,
)
from models.seq_grow_graph import decode_tokens
from models.seq_grow_graph.vocab import TOK_BOS, TOK_EOS, TOK_PAD

from tests.synthetic.test_generator import MockLoader, _make_mock_segments


@pytest.mark.parametrize(
    "first_import",
    [
        "from models.model import Model",
        "import models.seq_grow_graph",
        "import models.data_transform.scene_to_point",
    ],
)
def test_no_circular_import(first_import):
    """Each entry module must import cleanly in a fresh interpreter.

    Regression: models.model -> seq_grow_graph/__init__ -> tokenizer -> lane_graph
    triggered data_transform/__init__ -> scene_to_point -> (partially initialized)
    tokenizer. In-process pytest imports mask this because test collection order
    happens to pre-import the modules in a safe order.
    """
    r = subprocess.run(
        [sys.executable, "-c", first_import], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


@pytest.fixture
def scene_gt():
    segments, map_features = _make_mock_segments(3)
    gen = SyntheticSceneGenerator(loader=MockLoader(segments, map_features), seed=42)
    return gen.generate("mock_scenario", CurriculumStage.NoRandomization)


def test_scene_to_arrays_has_decodable_lane_tokens(scene_gt):
    scene, gt = scene_gt
    sample = scene_to_arrays(scene, gt)
    toks = sample.lane_tokens
    assert toks.dtype == np.int64
    assert toks[0] == TOK_BOS and toks[-1] == TOK_EOS
    g = decode_tokens(toks)
    # mock scenario: 3 parallel disconnected lanes (successor links are between
    # side-by-side lanes whose endpoints don't touch, so no merge) — every lane
    # contributes >= 1 edge after length re-split
    assert g.edge_src.shape[0] >= 3
    assert g.nodes.shape[0] >= 4
    # decoded node coords stay within the normalized-coords vocab range
    assert np.all(np.abs(g.nodes) <= 0.65)


def test_lane_tokens_empty_gt(scene_gt):
    _, gt = scene_gt
    gt2 = gt.model_copy(update={"lane_centerlines": {}, "topology": {}})
    assert lane_tokens_for_scene(gt2).tolist() == [TOK_BOS, TOK_EOS]


def test_batch_point_dict_pads_ragged_lane_tokens(scene_gt):
    scene, gt = scene_gt
    s1 = scene_to_arrays(scene, gt)
    s2 = s1._replace(lane_tokens=np.array([TOK_BOS, TOK_EOS], dtype=np.int64))
    data_dict, labels = batch_point_dict([s1, s2])
    lt = data_dict["lane_tokens"]
    assert lt.shape == (2, s1.lane_tokens.shape[0])
    assert lt.dtype == torch.int64
    assert (lt[0] == torch.from_numpy(s1.lane_tokens)).all()
    assert lt[1, 0] == TOK_BOS and lt[1, 1] == TOK_EOS
    assert (lt[1, 2:] == TOK_PAD).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PTv3/spconv requires CUDA")
def test_model_returns_both_heads(scene_gt):
    from models.model import Model

    scene, gt = scene_gt
    sample = scene_to_arrays(scene, gt)
    data_dict, labels = batch_point_dict([sample])
    data_dict = {
        k: v.cuda() if torch.is_tensor(v) else v for k, v in data_dict.items()
    }
    torch.manual_seed(0)
    model = Model().cuda().eval()
    with torch.no_grad():
        cls_logits, seq_logits = model(data_dict)
    assert cls_logits.shape[0] == labels.shape[0]
    T = data_dict["lane_tokens"].shape[1]
    assert seq_logits.shape[:2] == (1, T - 1)
    assert torch.isfinite(seq_logits).all()

    # without lane_tokens the model returns cls logits alone (old behavior)
    data_dict.pop("lane_tokens")
    with torch.no_grad():
        only_cls = model(data_dict)
    assert torch.is_tensor(only_cls) and only_cls.shape == cls_logits.shape
