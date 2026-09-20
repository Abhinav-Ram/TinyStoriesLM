"""Model size presets, roughly mirroring the smallest end of the TinyStories
paper's sweep (they went from 1M to ~33M non-embedding params). Sizes here are
picked to be trainable on a CPU-only machine in minutes-to-tens-of-minutes.
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    name: str
    n_layer: int
    n_embd: int
    n_head: int
    block_size: int = 256   # context length
    vocab_size: int = 4096  # set from the trained tokenizer at train time
    dropout: float = 0.0


PRESETS = {
    # ~0.4M total params (~0.1M non-embedding) - paper's "1-layer" style ablation
    "tiny": GPTConfig(name="tiny", n_layer=2, n_embd=64, n_head=2),
    # ~1.3M total params (~0.8M non-embedding)
    "small": GPTConfig(name="small", n_layer=4, n_embd=128, n_head=4),
    # ~5.8M total params (~4.7M non-embedding)
    "medium": GPTConfig(name="medium", n_layer=6, n_embd=256, n_head=8),
}
