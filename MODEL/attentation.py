from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    Expected shapes (common in GPT-like models):
      q, k, v: (B, nh, T, hs)

    Returns:
      out:  (B, nh, T, hs)
      att:  (B, nh, T, T) attention weights after softmax
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (B, nh, T, hs)")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must match in (B, nh, T)")
    if k.shape[-1] != q.shape[-1] or v.shape[-1] != q.shape[-1]:
        raise ValueError("q, k, v must have matching head dim (hs)")

    B, nh, T, hs = q.shape

    # (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

    if attn_mask is not None:
        # Supports masks broadcastable to (B, nh, T, T).
        # - boolean mask: True means "keep", False means "mask out"
        # - float mask: added directly to att (e.g. -inf for masked)
        if attn_mask.dtype == torch.bool:
            att = att.masked_fill(~attn_mask, float("-inf"))
        else:
            att = att + attn_mask

    if causal:
        # Upper-triangular (future) positions are masked out.
        causal_mask = torch.ones((T, T), device=att.device, dtype=torch.bool).tril()
        att = att.masked_fill(~causal_mask, float("-inf"))

    att = F.softmax(att, dim=-1)
    if dropout_p and dropout_p > 0.0:
        att = F.dropout(att, p=dropout_p, training=training)

    out = att @ v  # (B, nh, T, hs)
    return out, att


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention (GPT-style).

    Input:  x of shape (B, T, C)
    Output: y of shape (B, T, C)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        *,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            causal=True,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mha(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention built on `scaled_dot_product_attention`.

    This implementation is self-attention only (q=k=v from x).

    Input:  x of shape (B, T, C)
    Output: y of shape (B, T, C)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        *,
        causal: bool = True,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

        self.n_head = int(n_head)
        self.head_dim = n_embd // n_head
        self.causal = bool(causal)
        self.attn_dropout = float(attn_dropout)

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x: torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y, _att = scaled_dot_product_attention(
            q,
            k,
            v,
            causal=self.causal,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout,
            training=self.training,
        )

        # (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


__all__ = ["scaled_dot_product_attention", "MultiHeadAttention", "CausalSelfAttention"]


if __name__ == "__main__":
    # Quick shape check
    torch.manual_seed(0)
    B, T, C, nh = 2, 8, 16, 4
    x = torch.randn(B, T, C)
    attn = CausalSelfAttention(n_embd=C, n_head=nh, attn_dropout=0.1, resid_dropout=0.1)
    y = attn(x)
    print("x:", tuple(x.shape), "y:", tuple(y.shape))
