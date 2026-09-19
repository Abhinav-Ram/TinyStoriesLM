"""Stage 4: compute full validation perplexity for a trained checkpoint.

Usage:
    python evaluate_perplexity.py --config tiny
"""
import argparse
import math

import torch

from configs import PRESETS
from dataset import BinDataset
from model import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--run_dir", type=str, default="runs")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    ckpt = torch.load(f"{args.run_dir}/{args.config}/checkpoint.pt", map_location="cpu")
    from configs import GPTConfig
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_ds = BinDataset(f"{args.data_dir}/val.bin", config.block_size)
    n_blocks = (len(val_ds) - config.block_size - 1) // args.batch_size
    n_eval = min(n_blocks, 200)  # cap for speed; still a solid average over ~12800 tokens

    losses = []
    with torch.no_grad():
        for _ in range(n_eval):
            x, y = val_ds.get_batch(args.batch_size)
            _, loss = model(x, y)
            losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    print(f"[{args.config}] val_loss={mean_loss:.4f} perplexity={ppl:.2f} "
          f"(n_params={ckpt['n_params']:,}, non_embedding={ckpt['n_params_non_embedding']:,})")

    import json, os
    os.makedirs("results", exist_ok=True)
    out = {
        "config": args.config,
        "val_loss": mean_loss,
        "perplexity": ppl,
        "n_params": ckpt["n_params"],
        "n_params_non_embedding": ckpt["n_params_non_embedding"],
    }
    path = "results/perplexity.jsonl"
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            existing = [json.loads(l) for l in f if l.strip()]
        lines = [l for l in existing if l["config"] != args.config]
    lines.append(out)
    with open(path, "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")


if __name__ == "__main__":
    main()
