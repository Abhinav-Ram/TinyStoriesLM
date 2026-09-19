"""Stage 6: LLM-grade each model's completions with the Claude API, the way
the TinyStories paper uses GPT-4 as a grader ("GPT-Eval"): for each
completion, ask the grader to score grammar, creativity, plot, and
consistency with the story beginning, on a 1-10 scale.

Requires the ANTHROPIC_API_KEY environment variable.

Usage:
    python grade_completions.py --input results/generations.jsonl
"""
import argparse
import json
import os
import re
import time

import anthropic

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

Respond with ONLY a JSON object, no other text, in exactly this form:
{{"grammar": <int>, "creativity": <int>, "consistency": <int>, "plot": <int>}}
"""


def grade_one(client, model, prompt, completion, max_retries=5):
    msg_text = RUBRIC_PROMPT.format(prompt=prompt, completion=completion or "(empty)")
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": msg_text}],
            )
            text = resp.content[0].text.strip()
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
    ap.add_argument("--model", type=str, default="claude-sonnet-5")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic()

    with open(args.input) as f:
        rows = [json.loads(l) for l in f]

    print(f"Grading {len(rows)} completions with {args.model}...")
    results = []
    for i, row in enumerate(rows):
        scores = grade_one(client, args.model, row["prompt"], row["completion"])
        out = {**row, **scores}
        results.append(out)
        print(f"  [{i+1}/{len(rows)}] {row['config']} prompt#{row['prompt_id']}: {scores}")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote grades -> {args.output}")


if __name__ == "__main__":
    main()
