"""Stage 1: download TinyStories and carve out small train/val/holdout subsets.

Usage:
    python prepare_data.py --n_train 20000 --n_val 1000 --n_holdout 30 --seed 1337
"""
import argparse
import json
import random

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000, help="number of training stories")
    ap.add_argument("--n_val", type=int, default=1000, help="number of validation stories")
    ap.add_argument("--n_holdout", type=int, default=30,
                     help="number of stories held out for the LLM-graded completion eval")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()

    random.seed(args.seed)

    print("Downloading roneneldan/TinyStories from Hugging Face...")
    ds = load_dataset("roneneldan/TinyStories")
    train_full = ds["train"]
    val_full = ds["validation"]
    print(f"Full dataset: {len(train_full)} train stories, {len(val_full)} validation stories")

    train_idx = random.sample(range(len(train_full)), args.n_train)
    val_pool = list(range(len(val_full)))
    random.shuffle(val_pool)
    val_idx = val_pool[: args.n_val]
    holdout_idx = val_pool[args.n_val: args.n_val + args.n_holdout]

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    def write_txt(path, stories):
        with open(path, "w", encoding="utf-8") as f:
            for s in stories:
                text = s["text"].strip()
                f.write(text + "\n<|endoftext|>\n")

    train_stories = [train_full[i] for i in train_idx]
    val_stories = [val_full[i] for i in val_idx]
    holdout_stories = [val_full[i] for i in holdout_idx]

    write_txt(f"{args.out_dir}/train.txt", train_stories)
    write_txt(f"{args.out_dir}/val.txt", val_stories)

    # Build cut-off prompts for the generation eval: take the first ~40% of
    # words (min 8, max 40 words) as the prefix, keep the rest as reference.
    holdout = []
    for i, s in enumerate(holdout_stories):
        text = s["text"].strip().replace("\n", " ")
        words = text.split(" ")
        cut = max(8, min(40, len(words) * 2 // 5))
        prompt = " ".join(words[:cut])
        reference = " ".join(words[cut:])
        holdout.append({"id": i, "prompt": prompt, "reference": reference})

    with open(f"{args.out_dir}/holdout_prompts.jsonl", "w", encoding="utf-8") as f:
        for row in holdout:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(train_stories)} train stories -> {args.out_dir}/train.txt")
    print(f"Wrote {len(val_stories)} val stories -> {args.out_dir}/val.txt")
    print(f"Wrote {len(holdout)} holdout prompts -> {args.out_dir}/holdout_prompts.jsonl")


if __name__ == "__main__":
    main()
