"""
SeqGrowGraphHead — autoregressive lane-graph decoder (arXiv 2507.04822 §3.3).

A transformer decoder predicts the serialized lane-graph token sequence (see
tokenizer.py) conditioned on the scene: instead of the paper's BEV features, the
cross-attention memory here is the per-primitive tokens produced by
PTv3 -> PrimitivePooling ((n_prim, mem_dim) flat across the batch, sample boundaries
given by nprim_per_sample).

A learned global memory token is prepended to every sample's memory so that samples
with zero primitives still have one attendable slot (an all-masked memory row would
produce NaN attention).
"""
from __future__ import annotations

import torch
from torch import nn

from models.seq_grow_graph.vocab import (
    MAX_SEQ_LEN,
    N_COORD_BINS,
    TOK_BOS,
    TOK_EOS,
    TOK_PAD,
    VOCAB_SIZE,
)


class SeqGrowGraphHead(nn.Module):
    """Autoregressive transformer decoder over SeqGrowGraph token sequences."""

    def __init__(
        self,
        mem_dim: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=TOK_PAD)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.mem_proj = nn.Linear(mem_dim, d_model)
        self.global_mem = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.global_mem, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, VOCAB_SIZE)

        # Paper §3.3: node-position tokens (coord bins) get weight 2 in the CE loss.
        weights = torch.ones(VOCAB_SIZE)
        weights[:N_COORD_BINS] = 2.0
        self.register_buffer("token_weights", weights)

    def build_memory(
        self, prim_tokens: torch.Tensor, nprim_per_sample: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack flat per-primitive tokens into padded per-sample memory.

        Args:
            prim_tokens: (sum nprim, mem_dim), ordered per-sample (PrimitivePooling order).
            nprim_per_sample: (B,) long.
        Returns:
            memory: (B, P_max + 1, d_model) — global token at slot 0.
            key_padding_mask: (B, P_max + 1) bool, True = padded.
        """
        counts = nprim_per_sample.tolist()
        B = len(counts)
        parts = torch.split(prim_tokens, counts, dim=0)
        padded = nn.utils.rnn.pad_sequence(parts, batch_first=True)  # (B, P_max, mem_dim)
        if padded.numel() == 0:  # every sample has zero primitives
            padded = prim_tokens.new_zeros(B, 0, prim_tokens.shape[-1])
        memory = self.mem_proj(padded)
        memory = torch.cat([self.global_mem.expand(B, -1, -1), memory], dim=1)
        p_max = memory.shape[1] - 1
        arange = torch.arange(p_max, device=memory.device)
        mask = arange[None, :] >= nprim_per_sample[:, None]  # (B, P_max)
        mask = torch.cat([mask.new_zeros(B, 1), mask], dim=1)  # global slot never masked
        return memory, mask

    def _decode(
        self,
        seq_in: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T = seq_in.shape
        pos = torch.arange(T, device=seq_in.device)
        x = self.token_embed(seq_in) + self.pos_embed(pos)[None]
        # bool mask (True = blocked) to match the bool key_padding masks
        causal = torch.ones(T, T, dtype=torch.bool, device=seq_in.device).triu(1)
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            tgt_is_causal=True,
            tgt_key_padding_mask=seq_in == TOK_PAD,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(out)

    def forward(
        self,
        prim_tokens: torch.Tensor,
        nprim_per_sample: torch.Tensor,
        seq_in: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced logits.

        Args:
            prim_tokens: (sum nprim, mem_dim) pooled primitive features.
            nprim_per_sample: (B,) long.
            seq_in: (B, T) long, already shifted (e.g. full_seq[:, :-1]).
        Returns:
            (B, T, VOCAB_SIZE) logits.
        """
        memory, mem_mask = self.build_memory(prim_tokens, nprim_per_sample)
        return self._decode(seq_in, memory, mem_mask)

    @torch.no_grad()
    def generate(
        self,
        prim_tokens: torch.Tensor,
        nprim_per_sample: torch.Tensor,
        max_len: int | None = None,
    ) -> list[torch.Tensor]:
        """Greedy decode from <BOS>; full re-forward per step (eval/viz only).

        Returns a list of B 1-D LongTensors incl. BOS and (if reached) EOS.
        """
        max_len = min(max_len or self.max_len, self.max_len)
        memory, mem_mask = self.build_memory(prim_tokens, nprim_per_sample)
        B = memory.shape[0]
        device = memory.device
        seq = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        while seq.shape[1] < max_len and not bool(finished.all()):
            logits = self._decode(seq, memory, mem_mask)  # (B, t, V)
            nxt = logits[:, -1].argmax(dim=-1)  # (B,)
            nxt = torch.where(finished, torch.full_like(nxt, TOK_PAD), nxt)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            finished |= nxt == TOK_EOS
        out: list[torch.Tensor] = []
        for b in range(B):
            row = seq[b]
            eos = (row == TOK_EOS).nonzero()
            out.append(row[: int(eos[0, 0]) + 1] if eos.numel() else row)
        return out
