from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

import tokenizer
from MODEL.minigpt import GPTConfig, MiniGPT
from config import block_size, n_emb, n_heads, n_layers
from train import generate_text


CKPT_PATH = Path(__file__).parent / "minigpt.pt"
LEGACY_CKPT_PATH = Path(__file__).parent / "model.pt"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL: Optional[MiniGPT] = None


def load_model(device: torch.device | None = None) -> MiniGPT:
    if device is None:
        device = _DEVICE
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        n_embd=n_emb,
        dropout=0.0,
    )
    model = MiniGPT(cfg).to(device)

    ckpt_to_load = CKPT_PATH if CKPT_PATH.is_file() else LEGACY_CKPT_PATH

    if ckpt_to_load.is_file():
        state = torch.load(ckpt_to_load, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {ckpt_to_load}")
    else:
        print(
            f"No checkpoint found at {CKPT_PATH} (or legacy {LEGACY_CKPT_PATH}); "
            "using randomly initialized model."
        )

    model.eval()
    return model


def generate(
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    """
    Convenience wrapper used by the UI.
    Loads the model once and generates text from a prompt.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(_DEVICE)

    text = generate_text(
        _MODEL,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=_DEVICE,
        temperature=temperature,
        top_k=top_k,
    )
    return text


def main(prompt: Optional[str] = None, max_new_tokens: int = 200) -> None:
    if prompt is None:
        prompt = input("Enter prompt (must use characters from DATA/input.txt): ")

    text = generate(prompt, max_new_tokens=max_new_tokens)
    print("\n=== Generated text ===\n")
    print(text)


if __name__ == "__main__":
    main()

