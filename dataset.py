"""Tokenize train/val text into flat uint16 token-id arrays, and sample
random fixed-length blocks from them for training (same scheme as
nanoGPT's data pipeline: simple and fast enough for our sizes).
"""
import os

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(tokenizer_dir="tokenizer"):
    return ByteLevelBPETokenizer(
        f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt"
    )


def tokenize_to_bin(text_path, bin_path, tokenizer):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text).ids
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)
    return len(arr)


class BinDataset:
    def __init__(self, bin_path, block_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def get_batch(self, batch_size, device="cpu"):
        n = len(self.data) - self.block_size - 1
        ix = torch.randint(0, n, (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i: i + self.block_size].astype(np.int64)) for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1: i + 1 + self.block_size].astype(np.int64)) for i in ix
        ])
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    tok = load_tokenizer()
    for split in ["train", "val"]:
        n = tokenize_to_bin(f"data/{split}.txt", f"data/{split}.bin", tok)
        print(f"{split}: {n} tokens -> data/{split}.bin")
