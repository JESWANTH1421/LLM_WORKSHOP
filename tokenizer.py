from __future__ import annotations

from pathlib import Path
from typing import List


DATA_PATH = Path(__file__).parent / "DATA" / "input.txt"


def _load_text(path: Path = DATA_PATH) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.read_text(encoding="utf-8")


_text = _load_text()
_chars = sorted(set(_text))

vocab_size: int = len(_chars)
itos = {i: ch for i, ch in enumerate(_chars)}
stoi = {ch: i for i, ch in enumerate(_chars)}


def encode(s: str) -> List[int]:
    return [stoi[ch] for ch in s]


def decode(token_ids: List[int]) -> str:
    return "".join(itos[i] for i in token_ids)


__all__ = ["DATA_PATH", "vocab_size", "itos", "stoi", "encode", "decode"]

