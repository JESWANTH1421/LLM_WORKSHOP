from __future__ import annotations

import torch
import torch.nn as nn

from .attentation import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise MLP used inside a Transformer block."""

    def __init__(
        self,
        n_embd: int,
        *,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden = hidden_mult * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden, bias=bias),
            nn.GELU(),
            nn.Linear(hidden, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block (GPT-style):

      x = x + MHA(LN(x))
      x = x + FFN(LN(x))
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        *,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        mlp_hidden_mult: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            causal=True,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            bias=bias,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(
            n_embd,
            hidden_mult=mlp_hidden_mult,
            dropout=mlp_dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


__all__ = ["FeedForward", "TransformerBlock"]


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, nh = 2, 8, 16, 4
    x = torch.randn(B, T, C)
    block = TransformerBlock(C, nh, attn_dropout=0.1, resid_dropout=0.1, mlp_dropout=0.1)
    y = block(x)
    print(tuple(y.shape))

