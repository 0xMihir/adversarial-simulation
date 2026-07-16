"""Tests for SeqGrowGraphHead (shapes, causality, degenerate batches, generate)."""
import numpy as np
import torch

from models.seq_grow_graph import SeqGrowGraphHead, VOCAB_SIZE, decode_tokens
from models.seq_grow_graph.vocab import TOK_BOS, TOK_EOS, TOK_PAD


def _head() -> SeqGrowGraphHead:
    torch.manual_seed(0)
    return SeqGrowGraphHead(mem_dim=64, d_model=64, nhead=4, num_layers=2, dim_feedforward=128).eval()


def _batch(nprim=(5, 3), T=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    prim = torch.randn(sum(nprim), 64, generator=g)
    npr = torch.tensor(nprim, dtype=torch.long)
    seq = torch.randint(0, VOCAB_SIZE - 1, (len(nprim), T), generator=g)
    seq[:, 0] = TOK_BOS
    return prim, npr, seq


def test_forward_shapes():
    head = _head()
    prim, npr, seq = _batch()
    logits = head(prim, npr, seq)
    assert logits.shape == (2, seq.shape[1], VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_causality():
    head = _head()
    prim, npr, seq = _batch(T=16)
    base = head(prim, npr, seq)
    t_perturb = 8
    seq2 = seq.clone()
    seq2[:, t_perturb] = (seq2[:, t_perturb] + 1) % (VOCAB_SIZE - 1)
    pert = head(prim, npr, seq2)
    # logits strictly before the perturbed position are unchanged
    torch.testing.assert_close(base[:, :t_perturb], pert[:, :t_perturb])
    # ... and the perturbation is visible at/after it
    assert not torch.allclose(base[:, t_perturb:], pert[:, t_perturb:])


def test_zero_primitive_sample():
    head = _head()
    prim, _, seq = _batch(nprim=(8, 0))
    npr = torch.tensor([8, 0], dtype=torch.long)
    logits = head(prim, npr, seq)
    assert torch.isfinite(logits).all()


def test_all_zero_primitives():
    head = _head()
    npr = torch.tensor([0, 0], dtype=torch.long)
    prim = torch.zeros(0, 64)
    seq = torch.full((2, 6), TOK_PAD, dtype=torch.long)
    seq[:, 0] = TOK_BOS
    logits = head(prim, npr, seq)
    assert torch.isfinite(logits).all()


def test_pad_positions_dont_affect_others():
    head = _head()
    prim, npr, _ = _batch()
    seq = torch.full((2, 10), TOK_PAD, dtype=torch.long)
    seq[:, 0] = TOK_BOS
    seq[0, 1:6] = torch.arange(5)
    base = head(prim, npr, seq)
    seq2 = seq.clone()
    seq2[0, 8] = 42  # change a padded tail position of sample 0
    pert = head(prim, npr, seq2)
    torch.testing.assert_close(base[1], pert[1])  # other sample untouched
    torch.testing.assert_close(base[0, :8], pert[0, :8])


def test_generate_terminates_and_decodes():
    head = _head()
    prim, npr, _ = _batch()
    seqs = head.generate(prim, npr, max_len=40)
    assert len(seqs) == 2
    for s in seqs:
        assert s[0] == TOK_BOS
        assert s.shape[0] <= 40
        decode_tokens(s.cpu().numpy())  # tolerant decode must not raise


def test_token_weights():
    head = _head()
    assert head.token_weights.shape == (VOCAB_SIZE,)
    assert (head.token_weights[:200] == 2.0).all()
    assert (head.token_weights[200:] == 1.0).all()
