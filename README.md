# TinyStoriesLM

A small-scale replication of [*TinyStories: How Small Can Language Models Be
and Still Speak Coherent English?*](https://arxiv.org/abs/2305.07759)
(Eldan & Li, 2023): train several tiny decoder-only Transformers on a subset
of the TinyStories dataset, then evaluate them the way the paper does —
LLM-graded story completions plus validation perplexity.

This is a scaled-down replication (CPU-only hardware, a data subset, and
models in the ~0.1M–5M non-embedding-parameter range) meant to reproduce the
paper's *method*, not its exact numbers. See "How this differs from the
paper" at the bottom.

## Approach and choices

Where the paper leaves a detail unspecified (or where our environment
requires a deviation), here's what we picked and why:

| Aspect | Paper | This repo |
|---|---|---|
| Architecture | GPT-Neo, alternating local (window 256) / global attention | Plain GPT-2-style decoder-only Transformer, pre-LN blocks, learned positional embeddings, GELU MLP, tied input/output embeddings, **global** causal attention throughout (context is already short, so windowing wouldn't matter much at this scale) |
| Tokenizer | GPT-Neo's pretrained BPE tokenizer, restricted to its top 10K most frequent tokens | A fresh byte-level BPE tokenizer trained from scratch on our training subset (vocab size chosen for the subset size — see command below); simpler and has no dependency on GPT-Neo's tokenizer files |
| Model sizes | Swept roughly 1M–33M non-embedding params | Three presets: `tiny` (~0.1M non-embed), `small` (~0.8M), `medium` (~4.7M) — see `configs.py` |
| Context length | 512 (some small models use 256) | 256 |
| Optimizer/schedule | Not fully specified | AdamW (β=(0.9, 0.95), weight decay 0.1), linear warmup + cosine decay, grad-norm clipping at 1.0 |
| Dataset | Full TinyStories (~2.1M stories) | A random subset (see command below) so training finishes in minutes on CPU |
| Evaluation | GPT-4 grades generated completions on grammar, creativity, consistency (paper also discusses plot); validation loss/perplexity also reported | Claude grades completions on grammar, creativity, consistency, and plot (1–10 each); validation perplexity also reported |
| Seed | Not specified | Fixed at 1337 everywhere |

## Pipeline — one command per stage

```bash
pip install -r requirements.txt

# 1. Download TinyStories and carve out train/val/holdout subsets
python prepare_data.py --n_train 20000 --n_val 1000 --n_holdout 30

# 2. Train a small BPE tokenizer on the training subset
python train_tokenizer.py --vocab_size 4096

# 3. Tokenize train/val text into binary token-id arrays
python dataset.py

# 4. Train each model size (repeat per config: tiny / small / medium)
python train.py --config tiny --steps 1500
python train.py --config small --steps 1500
python train.py --config medium --steps 1500

# 5. Compute validation perplexity for each
python evaluate_perplexity.py --config tiny
python evaluate_perplexity.py --config small
python evaluate_perplexity.py --config medium

# 6. Generate completions from held-out cut-off story beginnings
python generate_completions.py --configs tiny small medium

# 7. Grade completions with Claude (requires ANTHROPIC_API_KEY)
python grade_completions.py

# 8. Aggregate everything into a results table
python summarize_results.py
```

A sanity run (tiny subset, few dozen steps) was done first to confirm
training loss decreases before scaling up to the real runs.

## Results

*(filled in after training + grading — see `results/results_table.md`)*

## How this differs from the paper

- **Scale**: models are far smaller (≤5M non-embedding params vs. up to
  33M+) and trained on a small random subset of TinyStories rather than the
  full ~2.1M-story dataset, so absolute quality and perplexity are not
  directly comparable to the paper's numbers — this reproduces the *method*
  and the *relative* scaling trend, on hardware without a GPU.
- **Architecture**: standard global causal attention instead of GPT-Neo's
  local/global alternation.
- **Tokenizer**: a from-scratch small BPE tokenizer instead of a restricted
  GPT-Neo tokenizer.
- **Grader**: Claude instead of GPT-4, and completions are generated once per
  prompt (temperature 0.8) rather than averaged over 10 samples per prompt as
  in the paper, to keep the number of API calls small.
