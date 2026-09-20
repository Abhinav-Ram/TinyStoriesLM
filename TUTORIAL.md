# TinyStoriesLM: A Module-by-Module Tutorial

**A hands-on walkthrough of small-language-model pretraining, evaluation, and
the mechanics that fine-tuning shares with it.**

This document is written to be presented to a mixed audience — from people
who have never trained a neural network, to engineers who train LLMs for a
living. Sections are layered: the main text explains *what* and *why*, and
boxed **"Under the hood"** callouts go deeper into the math/implementation
for those who want it. Skip the boxes on a first read if you want the
high-level story.

---

## Table of contents

1. [What is this project?](#1-what-is-this-project)
2. [Why TinyStories? The paper in 5 minutes](#2-why-tinystories-the-paper-in-5-minutes)
3. [Pretraining vs. fine-tuning — and why this project teaches both](#3-pretraining-vs-fine-tuning--and-why-this-project-teaches-both)
4. [The 30,000-foot view of the pipeline](#4-the-30000-foot-view-of-the-pipeline)
5. [Codebase map](#5-codebase-map)
6. [Module 1 — `configs.py`: choosing a model size](#6-module-1--configspy-choosing-a-model-size)
7. [Module 2 — `model.py`: the Transformer, built from scratch](#7-module-2--modelpy-the-transformer-built-from-scratch)
8. [Module 3 — `prepare_data.py`: getting and splitting data](#8-module-3--prepare_datapy-getting-and-splitting-data)
9. [Module 4 — `train_tokenizer.py`: turning text into numbers](#9-module-4--train_tokenizerpy-turning-text-into-numbers)
10. [Module 5 — `dataset.py`: feeding the model efficiently](#10-module-5--datasetpy-feeding-the-model-efficiently)
11. [Module 6 — `train.py`: the training loop](#11-module-6--trainpy-the-training-loop)
12. [Module 7 — `evaluate_perplexity.py`: the classic language-model metric](#12-module-7--evaluate_perplexitypy-the-classic-language-model-metric)
13. [Module 8 — `generate_completions.py`: sampling from the model](#13-module-8--generate_completionspy-sampling-from-the-model)
14. [Module 9 — `grade_completions.py`: LLM-as-a-judge](#14-module-9--grade_completionspy-llm-as-a-judge)
15. [Module 10 — `summarize_results.py`: turning numbers into a story](#15-module-10--summarize_resultspy-turning-numbers-into-a-story)
16. [From this repo to real fine-tuning: what would change](#16-from-this-repo-to-real-fine-tuning-what-would-change)
17. [Glossary](#17-glossary)
18. [Talking points / discussion questions for a live session](#18-talking-points--discussion-questions-for-a-live-session)

---

## 1. What is this project?

TinyStoriesLM is a small, runnable replication of Eldan & Li's 2023 paper
*["TinyStories: How Small Can Language Models Be and Still Speak Coherent
English?"](https://arxiv.org/abs/2305.07759)*. In one sentence, the paper's
finding is:

> If you shrink the *vocabulary and complexity of the training data itself*
> (children's stories using a 3–4 year old's vocabulary), you can shrink the
> *model* by 1000x compared to GPT-2-scale models and still get fluent,
> grammatical, coherent short stories out of it.

This repo lets you reproduce that finding end to end, on a laptop, in
minutes rather than days:

- Train several GPT-style language models **from scratch** at different
  sizes (roughly 0.1M to 5M parameters — small enough to train on a CPU or
  laptop GPU).
- Evaluate them the way the paper does: **validation perplexity** (a
  standard quantitative metric) plus **LLM-as-a-judge grading** of
  generated story completions (a more human-aligned qualitative metric).
- Compare how quality scales with model size.

Secondarily, it's a **teaching codebase**: every stage of a real LLM
training pipeline — data prep, tokenization, the training loop, evaluation —
is implemented from scratch in plain PyTorch, in under 700 lines total, so
you can read the *entire* mechanism instead of trusting a framework's
abstractions.

---

## 2. Why TinyStories? The paper in 5 minutes

Before ~2023, the prevailing assumption was: *coherent English generation
requires billions of parameters* (GPT-2 = 124M–1.5B, GPT-3 = 175B). The
TinyStories paper challenged this by asking: how much of that scale is
"paying for" the model's ability to speak fluent English at all, versus
paying for genuine world knowledge and reasoning?

Their method:
1. Use GPT-4/GPT-3.5 to synthesize ~2.1 million short stories, each
   constrained to a vocabulary a 3–4 year old would understand, and each
   built around 1–3 randomly chosen words/features it must include (to keep
   the dataset diverse instead of repetitive).
2. Train GPT-Neo-architecture models from **1M to ~33M parameters** on this
   dataset, from random initialization (no pretraining on general web text).
3. Evaluate not with a benchmark score, but by literally asking GPT-4 to
   grade generated story completions like a teacher grading a student's
   creative writing assignment — on grammar, creativity, and consistency.

**The result**: even a 1–2 layer, ~1M-parameter model can produce
grammatically correct, locally coherent short stories — something that was
considered impossible for a model that small. Quality scales smoothly with
both model size *and* depth (number of layers matters more than width for
a fixed parameter budget, up to a point).

**Why this matters for learning**: it's the cleanest possible setting to
watch a language model go from random noise to coherent text, because the
task (children's stories) is simple enough that a tiny model can plausibly
solve it, and improvements in model size are *visible in generated text* —
you don't need a research budget to see the effect.

---

## 3. Pretraining vs. fine-tuning — and why this project teaches both

**Important terminology check, because the project's own name promises
"fine-tuning" but the pipeline as run here does something slightly
different — being precise about this is itself a useful lesson.**

- **Pretraining**: start from a randomly initialized model (weights are
  noise) and train it on a large corpus with the generic "predict the next
  token" objective. This is what `train.py` in this repo does — every model
  starts from `nn.init.normal_(..., std=0.02)` in `model.py`, i.e. from
  scratch.
- **Fine-tuning**: start from an *already pretrained* model's weights (which
  already "know" grammar, facts, and reasoning patterns from a huge, general
  corpus) and continue training on a smaller, more specific dataset — usually
  with a much lower learning rate and far fewer steps, since you're
  *adjusting* existing knowledge, not building it from nothing.

**So which is this repo doing?** Strictly, it's a **from-scratch
pretraining** run — small-scale, but pretraining. TinyStories itself,
notably, is a pretraining paper: it shows that a small *architecture*
trained from scratch on a small, curated corpus can reach fluency, which is
a different and arguably more surprising claim than "fine-tuning a large
pretrained model on a narrow domain works" (which is well established).

**Why it still teaches fine-tuning mechanics**: every piece of machinery
you need for fine-tuning is exactly what this repo builds and exercises —
a data pipeline, a tokenizer, a training loop with an optimizer and a
learning-rate schedule, checkpointing, and an evaluation harness. The
*only* thing that would change to turn this into a fine-tuning pipeline is
**where the weights start from** and **the training hyperparameters**. See
[section 16](#16-from-this-repo-to-real-fine-tuning-what-would-change) for
exactly what that change would look like in this code.

---

## 4. The 30,000-foot view of the pipeline

```mermaid
flowchart LR
    A[HuggingFace: roneneldan/TinyStories<br/>2.12M stories] -->|prepare_data.py| B[train.txt / val.txt<br/>+ holdout prompts]
    B -->|train_tokenizer.py| C[BPE tokenizer<br/>vocab.json + merges.txt]
    B -->|dataset.py| D[train.bin / val.bin<br/>uint16 token arrays]
    C --> D
    D -->|train.py x3 sizes| E[checkpoint.pt<br/>tiny / small / medium]
    E -->|evaluate_perplexity.py| F[perplexity per size]
    E -->|generate_completions.py| G[generated completions<br/>from holdout prompts]
    G -->|grade_completions.py<br/>Ollama: gemma3/mistral/qwen3| H[grammar/creativity/<br/>consistency/plot scores]
    F --> I[summarize_results.py]
    H --> I
    I --> J[results_table.md]
```

Eight commands, each doing one clearly-scoped job (a deliberate design
choice — see "one command per stage" in the README). Everything downstream
of a stage only depends on files the upstream stage wrote, so you can
re-run any single stage without re-running the others.

---

## 5. Codebase map

```
TinyStoriesLM/
├── configs.py                 # model size presets (tiny/small/medium)
├── model.py                   # the GPT architecture (pure PyTorch, no framework)
├── prepare_data.py            # Stage 1: download + subset the dataset
├── train_tokenizer.py         # Stage 2: train a BPE tokenizer
├── dataset.py                 # Stage 3 helper: tokenize to binary, batch sampling, device pick
├── train.py                   # Stage 4: the training loop
├── evaluate_perplexity.py     # Stage 5: validation perplexity
├── generate_completions.py    # Stage 6: autoregressive generation from prompts
├── grade_completions.py       # Stage 7: LLM-as-a-judge grading (local, via Ollama)
├── summarize_results.py       # Stage 8: aggregate into a results table
├── requirements.txt
├── README.md                  # setup + exact commands + paper-vs-repo choices
└── TUTORIAL.md                # this file
```

Data flows through the filesystem, not in-memory: each stage reads files
the previous stage wrote (`data/*.txt`, `tokenizer/*.json`, `data/*.bin`,
`runs/<config>/checkpoint.pt`, `results/*.jsonl`). This is a deliberate,
very common pattern in ML pipelines — it makes each stage independently
resumable, debuggable, and cacheable.

---

## 6. Module 1 — `configs.py`: choosing a model size

```python
@dataclass
class GPTConfig:
    name: str
    n_layer: int      # how many Transformer blocks stacked
    n_embd: int        # the width of every vector flowing through the model
    n_head: int        # how many attention heads split that width
    block_size: int = 256   # max context length (tokens the model can see at once)
    vocab_size: int = 4096  # set later, from the trained tokenizer
    dropout: float = 0.0

PRESETS = {
    "tiny":   GPTConfig(name="tiny",   n_layer=2, n_embd=64,  n_head=2),
    "small":  GPTConfig(name="small",  n_layer=4, n_embd=128, n_head=4),
    "medium": GPTConfig(name="medium", n_layer=6, n_embd=256, n_head=8),
}
```

**What's going on**: a Transformer's size is (almost) fully determined by
three numbers — depth (`n_layer`), width (`n_embd`), and how many parallel
"attention heads" split that width (`n_head`, must divide `n_embd` evenly).
This tiny file is where the paper's central experiment — "how does quality
scale with size?" — becomes three concrete configurations you can train and
compare.

**Why these particular numbers**: they're chosen to land at roughly
0.1M / 0.8M / 4.7M non-embedding parameters — the smallest end of the
paper's 1M–33M sweep, picked so all three train in minutes on a laptop
CPU/GPU rather than requiring a cluster. See the README's paper-vs-repo
table for the exact parameter counts our code reports.

> **Under the hood — where do the parameters come from?**
> For a Transformer block, parameter count is dominated by:
> - Attention: `4 × n_embd²` (query, key, value, and output projections)
> - MLP: `8 × n_embd²` (a 4x-wide hidden layer, up and back down)
>
> So each block costs roughly `12 × n_embd²` parameters, and total
> non-embedding parameters ≈ `n_layer × 12 × n_embd²`. Notice this grows
> **quadratically** with width but only **linearly** with depth — one
> reason the TinyStories paper found depth to be a more parameter-efficient
> way to add capacity than width, at this scale.

---

## 7. Module 2 — `model.py`: the Transformer, built from scratch

This is the conceptual heart of the repo. It implements a GPT-2/GPT-Neo
style **decoder-only Transformer** in ~140 lines with zero dependencies
beyond raw PyTorch tensors — no `transformers` library, so every operation
is visible.

### 7.1 The overall shape

```
tokens ──► token embedding ──┐
                              ├─(add)─► [Block] × n_layer ──► LayerNorm ──► linear ──► next-token probabilities
positions ──► position embedding ──┘
```

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks  = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f    = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying — see below
```

- **Token embedding**: a lookup table, one learned vector per vocabulary
  entry. Turns discrete token IDs into continuous vectors the network can
  do math on.
- **Position embedding**: because attention (below) has no inherent sense
  of word order, we add a learned vector per position (0, 1, 2, ... up to
  `block_size`) so the model can tell "the first word" from "the fifth
  word".
- **Weight tying** (`lm_head.weight = self.tok_emb.weight`): the same
  matrix is used both to turn tokens into vectors *and*, transposed, to
  turn the final hidden vector back into a probability over the
  vocabulary. This roughly halves the embedding-related parameter count and
  is standard practice in small LMs (the intuition: "how similar is this
  hidden state to the embedding of word X" is a sensible way to answer
  "how likely is word X next").

### 7.2 One Transformer block

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention with a residual connection
        x = x + self.mlp(self.ln2(x))    # MLP with a residual connection
        return x
```

This is **pre-LayerNorm** style (normalize *before* the sub-layer, not
after) — the modern default because it makes very deep Transformers train
more stably. The `x = x + ...` pattern is a **residual/skip connection**:
each block only has to learn a *change* to make to its input, not
reconstruct the whole representation, which is what makes training deep
networks tractable at all.

### 7.3 Self-attention — the mechanism that made Transformers work

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        B, T, C = x.shape                      # batch, sequence length, width
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ...
```

**In plain language**: for every token, attention asks "which *other*
tokens in this sequence should I look at to decide what comes next, and
how much weight should I give each one?" Concretely:
- Every token produces a **query** (what am I looking for?), a **key**
  (what do I contain?), and a **value** (what do I actually offer if
  attended to).
- A token's query is compared (dot product) against every other token's
  key to get attention *scores*, which are turned into weights via
  softmax, and used to compute a weighted average of all the **values**.

**`is_causal=True`** is the single most important line for a *language
model* specifically: it masks out attention to future tokens, so token 5
can attend to tokens 0–5 but never to token 6+. Without this, the model
could "cheat" by looking at the answer during training — this is what
makes the architecture "decoder-only" / autoregressive.

**Multi-head**: instead of one attention computation over the full width,
we split it into `n_head` independent, smaller attention computations
(`head_dim = n_embd / n_head`) run in parallel, each free to specialize in
different kinds of relationships (e.g. one head might learn to track "the
last noun mentioned", another "the subject of the sentence"). Their
outputs are concatenated back to the full width.

> **Under the hood — deviation from the paper**: TinyStories uses GPT-Neo,
> which alternates *local* (windowed) and *global* attention across
> layers, to save compute at longer context lengths. This repo uses plain
> global causal attention (`F.scaled_dot_product_attention` above) in every
> layer. At our context length (256 tokens), the compute difference is
> negligible, and it keeps the implementation much easier to read — a
> reasonable simplification to flag explicitly rather than silently.

### 7.4 The MLP ("feed-forward") sub-layer

```python
class MLP(nn.Module):
    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))   # n_embd -> 4*n_embd -> n_embd
```

Where attention mixes information *between* tokens, the MLP processes
*each token independently*, expanding to 4x width, applying a nonlinearity
(GELU), and projecting back down. Intuitively: attention gathers relevant
context, the MLP is where most of the "computation"/"lookup" on that
gathered information happens. This alternation — mix across tokens, then
compute per token, repeat — is the entire recipe of a Transformer.

### 7.5 Generation (sampling)

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        logits, _ = self(idx[:, -self.config.block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx
```

The model only ever predicts **one next token at a time**, appends it to
the sequence, and repeats — this is why LLM generation is inherently
sequential and comparatively slow. `temperature` and `top_k` control the
randomness/creativity of sampling — see [section 13](#13-module-8--generate_completionspy-sampling-from-the-model)
for what these actually do.

---

## 8. Module 3 — `prepare_data.py`: getting and splitting data

```python
ds = load_dataset("roneneldan/TinyStories")   # 2,119,719 train / 21,990 validation stories
train_idx = random.sample(range(len(train_full)), args.n_train)
```

**Three splits, three different jobs** — a distinction worth being
precise about, because conflating them is one of the most common ML
mistakes:
- **Train** (`data/train.txt`): what the model's weights are updated on.
- **Validation** (`data/val.txt`): held-out stories used only to *measure*
  the model (perplexity) — never trained on, so a low score here means the
  model actually generalizes rather than having memorized the training set.
- **Holdout prompts** (`data/holdout_prompts.jsonl`): a further-separated
  slice of validation stories, cut off partway through (first ~40% of
  words), used specifically for the generation+grading evaluation. These
  are kept as a *third* split (not reused from the perplexity validation
  set in the same run) purely for hygiene — in this codebase they happen to
  be drawn from the same validation pool but never overlap index-wise with
  the perplexity sample within a single `prepare_data.py` call.

**Why only a subset (20,000 of 2.1M stories)?** The paper trains on the
full dataset; this repo intentionally uses a small random subset so that
training finishes in minutes instead of hours/days on consumer hardware.
This is the single biggest reason our absolute numbers won't match the
paper's — see the README's "how this differs" section.

---

## 9. Module 4 — `train_tokenizer.py`: turning text into numbers

Neural networks operate on numbers, not characters — a **tokenizer**'s job
is to convert text into a sequence of integers (and back). This repo
trains a **byte-level BPE (Byte-Pair Encoding)** tokenizer from scratch on
the training subset:

```python
tok = ByteLevelBPETokenizer()
tok.train(files=[args.train_file], vocab_size=4096, min_frequency=2)
```

**How BPE works, in brief**: start with individual bytes as the vocabulary,
then repeatedly find the *most frequent adjacent pair* of tokens in the
corpus and merge them into a new single token, up to `vocab_size` entries.
The result is a vocabulary of frequent whole words (`" the"`, `" said"`)
for common patterns, falling back to smaller sub-word or byte pieces for
rare/unseen words — this is exactly how GPT-2/GPT-3/GPT-4's tokenizers work.
Byte-level (rather than character-level) means it can represent *any*
string without an "unknown token" fallback, since every possible byte is a
valid starting token.

**Why 4096 and not GPT-2's 50,257?** A smaller vocabulary means a smaller
(and faster-to-train) embedding table, and is appropriate here because our
training corpus is both small and lexically simple (children's stories) —
a 50K vocabulary would mostly go unused and would eat a disproportionate
share of a tiny model's total parameters. The paper takes a related but
different approach: reusing GPT-Neo's full pretrained tokenizer but
restricting it to its top 10K most-frequent tokens on this data. We train a
fresh tokenizer instead, mainly for simplicity (no dependency on a
GPT-Neo tokenizer file).

---

## 10. Module 5 — `dataset.py`: feeding the model efficiently

Two unglamorous but important engineering choices live here.

**1. Tokenize once, store as a flat binary array of integers:**

```python
ids = tokenizer.encode(text).ids
np.array(ids, dtype=np.uint16).tofile(bin_path)   # data/train.bin, data/val.bin
```

`uint16` (values 0–65,535) is enough to hold token IDs for any vocab up to
65K, and is half the size of the more obvious `int32`. This whole-corpus
tokenize-once step avoids re-tokenizing text on every single training step.

**2. Sample random fixed-length windows via memory-mapping:**

```python
self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")   # doesn't load the whole file into RAM

def get_batch(self, batch_size, device):
    ix = torch.randint(0, len(self.data) - self.block_size - 1, (batch_size,))
    x = torch.stack([self.data[i : i+block_size] for i in ix])      # inputs
    y = torch.stack([self.data[i+1 : i+1+block_size] for i in ix])  # targets = inputs shifted by 1
```

`np.memmap` lets the OS page data in from disk on demand rather than
loading the entire tokenized corpus into RAM up front — this is the same
trick used by production-scale training pipelines (e.g. nanoGPT), just at
a much smaller scale here. Note `y` is simply `x` shifted one position to
the right: language modeling's "labels" are just *the next token in the
sequence itself* — no separate annotation is needed, which is exactly why
pretraining is called "self-supervised."

**3. Device auto-detection:**

```python
def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
```

Picks an NVIDIA GPU if present, otherwise Apple Silicon's MPS backend,
otherwise falls back to CPU — so the exact same code runs (at different
speeds) on a cloud GPU box, a MacBook, or a plain CPU machine.

---

## 11. Module 6 — `train.py`: the training loop

This is the piece every other ML framework (`transformers.Trainer`,
PyTorch Lightning, etc.) wraps in abstractions — here it's ~40 lines you
can read top to bottom.

```python
for step in range(args.steps):
    lr = get_lr(step, warmup, max_steps, max_lr, min_lr)   # 1. schedule the learning rate
    x, y = train_ds.get_batch(batch_size, device)          # 2. get a random batch
    _, loss = model(x, y)                                  # 3. forward pass + loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                                        # 4. backward pass (compute gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 5. clip exploding gradients
    optimizer.step()                                       # 6. update weights
```

This six-line loop *is* how every neural network, from this 0.1M-parameter
model to GPT-4-scale models, is trained. Scale changes the infrastructure
around it (distributed training, mixed precision, sharded optimizers) but
not this fundamental loop.

**The pieces worth understanding individually:**

- **Loss function**: `model(x, y)` internally computes
  `F.cross_entropy(logits, targets)` — for each position, "how surprised
  was the model by the actual next token, given the probability it
  assigned?" Cross-entropy is minimized when the model puts all its
  probability mass on the correct token.
- **AdamW optimizer**: an adaptive-learning-rate variant of gradient
  descent (each parameter effectively gets its own learning rate, based on
  the recent history of its gradients) with decoupled weight decay (a mild
  pull toward zero that acts as regularization). It's the default choice
  for training Transformers.
- **Warmup + cosine decay** (`get_lr`): start the learning rate near zero
  and ramp it up over the first ~100 steps (warmup — prevents early, noisy
  gradients from destabilizing the randomly-initialized weights), hold
  near the peak, then smoothly decay it following a cosine curve toward a
  small minimum by the end of training. This schedule shape is close to
  universal across modern LLM training.
- **Gradient clipping**: if the combined gradient norm across all
  parameters exceeds 1.0, scale it down. A cheap insurance policy against
  occasional large/unstable updates.
- **Fixed seed** (`torch.manual_seed(1337)`, set once at the top): makes
  the random batch sampling and weight initialization reproducible —
  running the same command twice gives (nearly*) identical results, which
  matters enormously for debugging and for fair comparisons between model
  sizes. *(Full bit-for-bit determinism on GPU also requires disabling
  certain nondeterministic CUDA kernels, which this repo doesn't bother
  with, since we're not chasing that level of reproducibility.)*

Every `eval_interval` steps, the loop pauses to compute **validation
loss** (on data never trained on) alongside training loss — watching these
two together is how you detect overfitting (train loss keeps dropping,
val loss stalls or rises).

---

## 12. Module 7 — `evaluate_perplexity.py`: the classic language-model metric

```python
ppl = math.exp(mean_cross_entropy_loss)
```

**Perplexity** is just `e^(average cross-entropy loss)`, but the
exponentiation makes it much more interpretable: it's roughly *"the
effective number of equally-likely choices the model was uncertain
between, on average, at each position."* A perplexity of 1 means perfect,
certain prediction; a perplexity equal to the vocabulary size means the
model is doing no better than guessing uniformly at random. Lower is
better, and because it only depends on probabilities the model assigns to
the *actual* held-out text, it needs no human or LLM judgment — a fast,
fully automatic, if somewhat abstract, quality signal.

**Caveat worth surfacing to any audience**: perplexity is tokenizer- and
vocabulary-dependent — you cannot directly compare the perplexity numbers
between two models that use different tokenizers or vocab sizes, only
between models sharing the same one (as our tiny/small/medium all do,
since they share one tokenizer). This is also why we don't try to compare
perplexity against the local Ollama grader models — they use entirely
different tokenizers.

---

## 13. Module 8 — `generate_completions.py`: sampling from the model

This stage takes the `holdout_prompts.jsonl` cut-off story beginnings and
asks each trained model to continue them — this is what actually gets
graded in the next stage.

**Two knobs that control the character of generated text:**

- **`temperature`** (default 0.8): before converting model outputs into a
  probability distribution, divide them by this value. Temperature < 1
  sharpens the distribution (more confident, more repetitive, "safer");
  temperature > 1 flattens it (more random, more prone to nonsense).
  Temperature = 0 would be fully deterministic ("always pick the single
  most likely next token" — called *greedy decoding*).
- **`top_k`** (default 50): before sampling, discard all but the 50
  highest-probability next tokens. This prevents the model from
  occasionally sampling a wildly unlikely token from the distribution's
  long tail, which tends to derail generation into gibberish.

Both are standard sampling controls used identically at every LLM scale —
the same parameters exist (often under the same names) in the OpenAI and
Anthropic APIs.

---

## 14. Module 9 — `grade_completions.py`: LLM-as-a-judge

This is the paper's most distinctive methodological contribution, and
arguably the more interesting half of the evaluation (versus the purely
quantitative perplexity number).

**The idea**: instead of scoring against a fixed reference answer (like
BLEU/ROUGE, which don't work well for open-ended creative text — there's no
single "correct" story continuation), *ask another language model to grade
it*, the way a teacher grades a student's creative writing:

```
Grade the completion on these four dimensions, 1-10 each:
- grammar: Is the completion grammatically correct English?
- creativity: Is the completion original/creative rather than generic or repetitive?
- consistency: Does the completion stay consistent with the beginning?
- plot: Does the completion develop a sensible, coherent continuation?
```

The paper uses GPT-4 as the judge (calling this method "GPT-Eval"); **this
repo, by design, uses a locally-hosted open-weight model via
[Ollama](https://ollama.com)** (your choice of `gemma3`, `mistral`, or
`qwen3`) instead — no cloud API, no cost per call, everything stays on your
machine:

```python
requests.post(f"{host}/api/generate", json={
    "model": model, "prompt": rubric_prompt, "format": "json",
    "options": {"temperature": 0, "num_predict": 200},
})
```

**Why this matters as a general technique, beyond this project**:
LLM-as-a-judge has become one of the standard ways to evaluate open-ended
generation quality at scale (chatbot arenas, RLHF reward models, and most
production eval harnesses all use variants of this). Its core weakness is
exactly its core strength — you've replaced "no automatic metric captures
this" with "a black-box model's judgment, which has its own biases, blind
spots, and (for smaller local judge models) simply weaker judgment than a
frontier model." That's a real, worth-stating caveat for the numbers this
particular repo produces: gemma3/mistral/qwen3 are meaningfully weaker
graders than GPT-4, so treat scores as a directional signal, not a precise
measurement — the same caution you'd want to apply to any LLM-judged eval
in production.

**Temperature 0, `format: "json"`**: for a *grading* task, you want the
judge to be as consistent/deterministic as possible (unlike generation,
where randomness is a feature) — hence temperature 0 here, versus 0.8 for
the story generation itself.

---

## 15. Module 10 — `summarize_results.py`: turning numbers into a story

The final stage just joins `results/perplexity.jsonl` and
`results/grades.jsonl` on model config name and averages the per-dimension
grades across all holdout prompts, producing one markdown table:

| config | n_params | non_embed_params | val_ppl | grammar | creativity | consistency | plot | avg |
|---|---|---|---|---|---|---|---|---|
| tiny | ... | ... | ... | ... | ... | ... | ... | ... |
| small | ... | ... | ... | ... | ... | ... | ... | ... |
| medium | ... | ... | ... | ... | ... | ... | ... | ... |

**What to actually look for when presenting this table**: does perplexity
go down *and* grading scores go up as model size increases, roughly
monotonically? If so, that's the paper's headline scaling result,
reproduced at your own (much smaller) scale. It's also worth explicitly
checking whether the two metrics *agree* with each other — a case where
perplexity improves but grading scores don't (or vice versa) is a genuinely
interesting talking point about what each metric does and doesn't capture.

---

## 16. From this repo to real fine-tuning: what would change

Since this project doubles as a fine-tuning demonstration, here's the
precise, concrete answer to "what would I change in this exact code to
turn it into a fine-tuning pipeline instead of pretraining?"

1. **Initialize from pretrained weights, not random noise.** In
   `train.py`, replace `model = GPT(config)` (which triggers `model.py`'s
   `self.apply(self._init_weights)` — random init) with code that loads an
   existing checkpoint's weights into the model first. In practice, this
   almost always also means switching to a real pretrained model class
   (e.g. loading a small pretrained GPT-2 via Hugging Face `transformers`)
   rather than this repo's custom `GPT` class, since you need the *exact*
   architecture and tokenizer the checkpoint was trained with.
2. **Use a much smaller learning rate.** This repo's `--lr 3e-4` is
   appropriate for training from scratch. Fine-tuning typically uses
   1e-5 to 5e-5 — an order of magnitude (or two) smaller — because you're
   nudging already-good weights, not building representations from zero;
   a large learning rate would rapidly destroy what the model already
   knows ("catastrophic forgetting").
3. **Train for far fewer steps**, often a single pass or a few passes over
   a small, task-specific dataset, instead of the multi-epoch runs used
   for pretraining.
4. **Data changes shape, not code.** `prepare_data.py`'s pattern (write
   train/val text, tokenize, sample batches) is identical whether the text
   is "2 million generic short stories" (pretraining corpus) or "500
   examples of the specific task you're fine-tuning for" (e.g.
   instruction-response pairs, or stories in a particular character's
   voice) — only the *source* and *size* of the data differ, not the
   pipeline shape.
5. **The evaluation harness barely changes at all.** Perplexity and
   LLM-graded generation both remain perfectly valid ways to measure a
   fine-tuned model — `evaluate_perplexity.py`, `generate_completions.py`,
   and `grade_completions.py` would work against a fine-tuned checkpoint
   with zero code changes, only pointed at a different checkpoint file.

**The takeaway for a presentation**: pretraining and fine-tuning are the
*same training loop*, applied to different starting weights, different
data, and different (mostly smaller/shorter) hyperparameters for
fine-tuning. If you understand this repo's `train.py`, you understand the
mechanical core of fine-tuning too — the differences are in *inputs and
settings*, not in *mechanism*.

---

## 17. Glossary

| Term | Plain-language definition |
|---|---|
| **Token** | The atomic unit a language model reads/writes — usually a word piece, not a full word or single character. |
| **Tokenizer** | The algorithm that converts text ↔ a sequence of token IDs (integers). |
| **BPE (Byte-Pair Encoding)** | A tokenizer-building algorithm: iteratively merge the most frequent adjacent symbol pair into a new symbol, building up common whole-word tokens from smaller pieces. |
| **Embedding** | A learned lookup table mapping a discrete ID (token, or position) to a continuous vector. |
| **Attention** | The mechanism by which each token gathers information from other tokens in the sequence, weighted by learned relevance (query/key/value). |
| **Causal / autoregressive** | Predicting each token only from the tokens before it, never from future tokens — required for text generation to make sense. |
| **Transformer block** | One repeated unit of a Transformer: attention (mixes across tokens) followed by an MLP (processes each token independently), each wrapped in a residual connection and normalization. |
| **Residual / skip connection** | Adding a layer's input back to its output (`x = x + f(x)`), which makes very deep networks trainable. |
| **Weight tying** | Sharing one matrix between the input embedding and the output projection layer, saving parameters. |
| **Pretraining** | Training a model from random initialization on a large, general corpus with a self-supervised objective (next-token prediction). |
| **Fine-tuning** | Continuing training an already-pretrained model on a smaller, more specific dataset, usually with a lower learning rate. |
| **Cross-entropy loss** | The training objective: how much probability mass the model failed to put on the actually-correct next token, averaged over a batch. |
| **Perplexity** | `e^(cross-entropy loss)` — an interpretable proxy for "how many plausible choices was the model uncertain among," lower is better. |
| **Learning rate schedule** | A plan for how the optimizer's step size changes over training (here: warmup then cosine decay). |
| **Gradient clipping** | Capping the size of a parameter update to prevent rare, destabilizingly large updates. |
| **Temperature (sampling)** | A knob controlling how "confident vs. random" text generation is; lower = safer/more repetitive, higher = more random. |
| **Top-k sampling** | Restricting sampling to only the k most likely next tokens, to avoid picking absurd low-probability tokens. |
| **LLM-as-a-judge** | Using one language model to grade another model's output quality, in lieu of a fixed reference answer or human grader. |
| **Non-embedding parameters** | A model's total parameters minus its embedding tables — often reported separately because embedding size scales with vocabulary choice, not "model capability" in the way layer/width do. |

---

## 18. Talking points / discussion questions for a live session

Use these to drive an interactive walkthrough rather than a pure lecture:

- **For beginners**: "We just watched a model go from producing random
  bytes to grammatical English, using nothing but 'guess the next token,
  get corrected, repeat' millions of times. Why does such a simple
  objective produce something that looks like understanding?"
- **For intermediate engineers**: "Perplexity improved between two model
  sizes, but did the LLM-judged creativity/plot scores actually move in
  the same direction? What would it mean if they didn't?"
- **For advanced/research-minded engineers**: "The paper's real claim
  isn't 'small models are as good as big ones' — it's 'a sufficiently
  simple *data distribution* dramatically lowers the model capacity needed
  for fluency.' What does that suggest about where the majority of a
  frontier model's parameters are actually 'spent'?"
- **Live demo idea**: train `tiny` for a very short run live (a couple of
  minutes), then generate a completion at step 0 (near-random weights) vs.
  the final checkpoint, side by side — the qualitative jump is usually the
  single most convincing moment in a presentation like this.
- **Bridge to fine-tuning**: "If we swapped `model = GPT(config)` for
  `model = load_pretrained_gpt2()`, dropped the learning rate 10x, and
  pointed `prepare_data.py` at 500 of *our own* examples instead of 20,000
  generic stories — what in this exact codebase would we still be able to
  reuse untouched?" (Answer: `dataset.py`, the shape of `train.py`'s loop,
  and all of the evaluation stage — see [section 16](#16-from-this-repo-to-real-fine-tuning-what-would-change).)
