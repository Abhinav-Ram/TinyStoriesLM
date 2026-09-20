"""Stage 2: train a small byte-level BPE tokenizer on the training subset.

Deviation from the paper: the paper reuses GPT-Neo's pretrained BPE
tokenizer but restricts it to its top 10K most frequent tokens on the data,
to shrink the embedding table while staying close to a "real" GPT
tokenizer. We instead train a fresh byte-level BPE tokenizer directly on
our (much smaller) training subset, with a vocab size chosen for that
subset. This is simpler, needs no dependency on GPT-Neo's tokenizer files,
and gives comparable behavior at our scale.

Usage:
    python train_tokenizer.py --vocab_size 4096
"""
import argparse

from tokenizers import ByteLevelBPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, default="data/train.txt")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--out_dir", type=str, default="tokenizer")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[args.train_file],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )
    tok.save_model(args.out_dir)
    print(f"Saved tokenizer (vocab_size={tok.get_vocab_size()}) to {args.out_dir}/")


if __name__ == "__main__":
    main()
