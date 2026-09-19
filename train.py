"""Stage 3: train one model size.

Usage:
    python train.py --config tiny --steps 1500
    python train.py --config small --steps 1500
    python train.py --config medium --steps 1500

Fixed seed everywhere for reproducibility. Optimizer/schedule follow
standard small-GPT practice (AdamW + cosine decay with linear warmup); the
paper does not spell out these hyperparameters, so this is our chosen
default.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from configs import PRESETS
from dataset import BinDataset, load_tokenizer, pick_device
from model import GPT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    device = pick_device()
    print(f"Using device: {device}")

    tok = load_tokenizer(args.tokenizer_dir)
    vocab_size = tok.get_vocab_size()

    config = PRESETS[args.config]
    config.block_size = args.block_size
    config.vocab_size = vocab_size

    model = GPT(config).to(device)
    n_params = model.num_params()
    n_params_ne = model.num_params(non_embedding=True)
    print(f"[{args.config}] total params: {n_params:,} | non-embedding params: {n_params_ne:,}")

    train_ds = BinDataset(f"{args.data_dir}/train.bin", config.block_size)
    val_ds = BinDataset(f"{args.data_dir}/val.bin", config.block_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    run_dir = os.path.join(args.out_dir, args.config)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.jsonl")
    log_f = open(log_path, "w")

    @torch.no_grad()
    def estimate_loss(ds, iters):
        model.eval()
        losses = []
        for _ in range(iters):
            x, y = ds.get_batch(args.batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    t0 = time.time()
    for step in range(args.steps + 1):
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = train_ds.get_batch(args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = estimate_loss(val_ds, args.eval_iters)
            elapsed = time.time() - t0
            rec = {
                "step": step, "train_loss": loss.item(), "val_loss": val_loss,
                "lr": lr, "elapsed_sec": elapsed,
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"step {step:5d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} "
                  f"| ppl {math.exp(min(val_loss, 20)):.2f} | lr {lr:.2e} | {elapsed:.1f}s")

    log_f.close()

    ckpt = {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": vars(config),
        "n_params": n_params,
        "n_params_non_embedding": n_params_ne,
        "args": vars(args),
    }
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
