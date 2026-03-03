from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBlock


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layers: int
    n_heads: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True


class MiniGPT(nn.Module):
    """
    Decoder-only GPT model.

    - token embedding
    - positional embedding
    - stack of Transformer blocks
    - final LayerNorm
    - linear output projection to vocab
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.n_embd,
                    cfg.n_heads,
                    attn_dropout=cfg.dropout,
                    resid_dropout=cfg.dropout,
                    mlp_dropout=cfg.dropout,
                    bias=cfg.bias,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B, T) token ids
        targets: (B, T) token ids (optional)
        returns: (logits, loss)
          logits: (B, T, vocab_size)
          loss: scalar or None
        """
        if idx.ndim != 2:
            raise ValueError("idx must have shape (B, T)")
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length T={T} exceeds block_size={self.cfg.block_size}")

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)  # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, C)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            if targets.shape != (B, T):
                raise ValueError("targets must have the same shape as idx (B, T)")
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        idx: (B, T) prompt tokens
        returns: (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]  # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab)

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, float("-inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx


__all__ = ["GPTConfig", "MiniGPT"]


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=16, n_layers=2, n_heads=4, n_embd=32, dropout=0.1)
    m = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = m(x, x)
    print("logits:", tuple(logits.shape), "loss:", float(loss))
