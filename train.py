from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

import tokenizer
from config import (
    batch_size,
    block_size,
    learning_rate,
    max_iters,
    n_emb,
    n_heads,
    n_layers,
)
from MODEL.minigpt import GPTConfig, MiniGPT


_text = tokenizer.DATA_PATH.read_text(encoding="utf-8")
_data = torch.tensor(tokenizer.encode(_text), dtype=torch.long)

_n = int(0.9 * len(_data))
train_data = _data[:_n]
val_data = _data[_n:]


def get_batch(
    split: str = "train",
    *,
    device: Optional[torch.device | str] = None,
    bs: int = batch_size,
    seq_len: int = block_size,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a batch (x, y) for next-token prediction.

    x has shape (bs, seq_len) and y has shape (bs, seq_len), where
    y is x shifted by one character in the original stream.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    data = train_data if split == "train" else val_data
    if len(data) < seq_len + 1:
        raise ValueError(
            f"Not enough tokens ({len(data)}) for seq_len={seq_len}. "
            "Use a smaller block_size/seq_len or provide more data."
        )

    ix = torch.randint(0, len(data) - seq_len - 1, (bs,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])

    if device is not None:
        x = x.to(device)
        y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(
    model: MiniGPT, *, device: torch.device, eval_iters: int = 100
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split, device=device)
            _logits, loss = model(x, y)
            losses[k] = loss
        out[split] = float(losses.mean().detach().cpu())
    model.train()
    return out


@torch.no_grad()
def generate_text(
    model: MiniGPT,
    prompt: str,
    max_new_tokens: int,
    *,
    device: torch.device,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    model.eval()

    if prompt == "":
        prompt = tokenizer.itos[0]

    try:
        prompt_ids = tokenizer.encode(prompt)
    except KeyError:
        unknown = sorted({ch for ch in prompt if ch not in tokenizer.stoi})
        raise ValueError(
            "Prompt contains characters not in the vocabulary. "
            f"Unknown: {unknown}"
        ) from None

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _loss = model(idx_cond)  # (1, T, vocab)
        logits = logits[:, -1, :]  # (1, vocab)

        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, k=top_k)
            logits = torch.where(
                logits < v[:, [-1]], torch.full_like(logits, float("-inf")), logits
            )

        probs = F.softmax(logits, dim=-1)  # (1, vocab)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        idx = torch.cat([idx, next_id], dim=1)

    out = idx[0].tolist()
    return tokenizer.decode(out)


def main() -> None:
    torch.manual_seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        n_embd=n_emb,
        dropout=0.1,
    )
    model = MiniGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    log_interval = 50
    eval_interval = 500
    eval_iters = 20

    for step in range(max_iters):
        x, y = get_batch("train", device=device)
        _logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            print(f"step {step}: loss {float(loss.detach().cpu()):.4f}")

        if step % eval_interval == 0:
            losses = estimate_loss(model, device=device, eval_iters=eval_iters)
            print(f"eval {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    losses = estimate_loss(model, device=device, eval_iters=eval_iters)
    print(f"final: train {losses['train']:.4f}, val {losses['val']:.4f}")

    torch.save(model.state_dict(), "minigpt.pt")
    print("Saved checkpoint to minigpt.pt")


if __name__ == "__main__":
    main()


