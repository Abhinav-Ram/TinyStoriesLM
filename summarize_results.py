"""Stage 7: aggregate perplexity + grading results into a markdown table.

Usage:
    python summarize_results.py
"""
import json
from collections import defaultdict

from configs import PRESETS


def main():
    with open("results/perplexity.jsonl") as f:
        ppl = {json.loads(l)["config"]: json.loads(l) for l in f}

    grades = defaultdict(list)
    try:
        with open("results/grades.jsonl") as f:
            for l in f:
                row = json.loads(l)
                grades[row["config"]].append(row)
    except FileNotFoundError:
        pass

    dims = ["grammar", "creativity", "consistency", "plot"]
    header = ["config", "n_params", "non_embed_params", "val_ppl"] + dims + ["avg"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for name in PRESETS:
        if name not in ppl:
            continue
        p = ppl[name]
        row = [name, f"{p['n_params']:,}", f"{p['n_params_non_embedding']:,}", f"{p['perplexity']:.2f}"]
        rows = grades.get(name, [])
        dim_means = []
        for d in dims:
            if rows:
                m = sum(r[d] for r in rows) / len(rows)
                dim_means.append(m)
                row.append(f"{m:.2f}")
            else:
                row.append("-")
        if dim_means:
            row.append(f"{sum(dim_means)/len(dim_means):.2f}")
        else:
            row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    print(table)
    with open("results/results_table.md", "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
