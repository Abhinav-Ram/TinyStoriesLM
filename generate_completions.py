"""Stage 5: generate story completions from held-out cut-off beginnings, for
every trained model. These get graded by Claude in the next stage.

Usage:
    python generate_completions.py --configs tiny small medium
"""
import argparse
import json

import torch

from configs import GPTConfig, PRESETS
from dataset import load_tokenizer, pick_device
from model import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(PRESETS.keys()))
    ap.add_argument("--run_dir", type=str, default="runs")
    ap.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    ap.add_argument("--prompts_file", type=str, default="data/holdout_prompts.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_file", type=str, default="results/generations.jsonl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    tok = load_tokenizer(args.tokenizer_dir)

    with open(args.prompts_file) as f:
        prompts = [json.loads(l) for l in f]

    import os
    os.makedirs("results", exist_ok=True)

    rows = []
    for cfg_name in args.configs:
        ckpt = torch.load(f"{args.run_dir}/{cfg_name}/checkpoint.pt", map_location="cpu")
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        print(f"Generating with [{cfg_name}]...")
        for p in prompts:
            ids = tok.encode(p["prompt"]).ids
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
            completion_ids = out[0, len(ids):].tolist()
            completion = tok.decode(completion_ids)
            # stop at end-of-text if the model produced one
            completion = completion.split("<|endoftext|>")[0].strip()
            rows.append({
                "config": cfg_name,
                "prompt_id": p["id"],
                "prompt": p["prompt"],
                "reference": p["reference"],
                "completion": completion,
            })

    with open(args.out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} generations -> {args.out_file}")


if __name__ == "__main__":
    main()
