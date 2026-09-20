"""Stage 6: LLM-grade each model's completions, the way the TinyStories
paper uses GPT-4 as a grader ("GPT-Eval"): for each completion, ask the
grader to score grammar, creativity, plot, and consistency with the story
beginning, on a 1-10 scale.

Deviation from the paper: the paper uses GPT-4 (a cloud API). Per project
preference, grading here runs entirely locally against an Ollama server
(no cloud calls, no API key) using a locally-hosted open-weight model of
your choice.

Requires Ollama running locally (https://ollama.com) with the chosen model
already pulled, e.g.:
    ollama pull gemma3

Usage:
    python grade_completions.py --model gemma3
    python grade_completions.py --model mistral
    python grade_completions.py --model qwen3
"""
import argparse
import json
import re
import time

import requests

RUBRIC_PROMPT = """You are grading a short story completion written by a small \
language model that was shown only the beginning of a children's story and \
had to continue it. The intended audience is young children, so the story \
should be simple, but still grammatically correct, coherent, and creative.

Story beginning (given to the model):
\"\"\"{prompt}\"\"\"

Model's completion (what the model generated after the beginning):
\"\"\"{completion}\"\"\"

Grade the completion on these four dimensions, each on a 1-10 integer scale \
(10 = excellent):
- grammar: Is the completion grammatically correct English?
- creativity: Is the completion original/creative rather than generic or repetitive?
- consistency: Does the completion stay consistent with the beginning (characters, \
setting, tone, and events already established)?
- plot: Does the completion develop a sensible, coherent continuation of the plot \
(as opposed to rambling or looping)?

Respond with ONLY a JSON object, no other text, no reasoning, in exactly this form:
{{"grammar": <int>, "creativity": <int>, "consistency": <int>, "plot": <int>}}
"""


def grade_one(host, model, prompt, completion, max_retries=5, timeout=120):
    msg_text = RUBRIC_PROMPT.format(prompt=prompt, completion=completion or "(empty)")
    payload = {
        "model": model,
        "prompt": msg_text,
        "stream": False,
        "think": False,          # ignored by models/versions that don't support it
        "format": "json",
        "options": {"temperature": 0, "num_predict": 200},
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json()["response"].strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0))
        except Exception as e:
            wait = 2 ** attempt
            print(f"  grading error ({e}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("grading failed after retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results/generations.jsonl")
    ap.add_argument("--output", type=str, default="results/grades.jsonl")
    ap.add_argument("--model", type=str, default="gemma3",
                     choices=["gemma3", "mistral", "qwen3"],
                     help="Ollama model to use as the grader (must already be pulled)")
    ap.add_argument("--host", type=str, default="http://localhost:11434",
                     help="Ollama server URL")
    args = ap.parse_args()

    try:
        r = requests.get(f"{args.host}/api/tags", timeout=5)
        r.raise_for_status()
        available = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        if args.model not in available:
            raise SystemExit(
                f"Model '{args.model}' not found in Ollama (available: {available}). "
                f"Run `ollama pull {args.model}` first."
            )
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"Could not reach Ollama at {args.host}. Is it running? (`ollama serve`)"
        )

    with open(args.input) as f:
        rows = [json.loads(l) for l in f]

    print(f"Grading {len(rows)} completions locally with Ollama model '{args.model}'...")
    results = []
    for i, row in enumerate(rows):
        scores = grade_one(args.host, args.model, row["prompt"], row["completion"])
        out = {**row, "grader": args.model, **scores}
        results.append(out)
        print(f"  [{i+1}/{len(rows)}] {row['config']} prompt#{row['prompt_id']}: {scores}")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote grades -> {args.output}")


if __name__ == "__main__":
    main()
