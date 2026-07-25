#!/usr/bin/env python3
"""Scatter: original tool size (x, tokens) vs CATS percent savings (y, %).

Overlays all populated tokenizers on one axis -- faint scatter per tokenizer
plus a binned-median trend line each -- so you can see whether the tokenizers
AGREE on the size->savings relationship (the point of the generalization check).
One point per converted unique tool.

X-axis: original (normalized JSON) token count, per tokenizer.
Y-axis: per-tool percent reduction (higher = CATS smaller).

Run::

    uv run --project cats-converter python eval/part1/savings_vs_size.py eval/out/records.jsonl
    # optional 2nd arg: output image path
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TOKENIZERS = ("tiktoken", "qwen", "anthropic")
COLORS = {"tiktoken": "tab:blue", "qwen": "tab:orange", "anthropic": "tab:green"}


def _points(rows, tok):
    jk, ck = f"tokens_json_{tok}", f"tokens_cats_{tok}"
    pts = []
    for r in rows:
        if r["bucket"] != "converted":
            continue
        j, c = r.get(jk), r.get(ck)
        if j is None or c is None or j <= 0:
            continue
        pts.append((j, (j - c) / j * 100))
    return pts


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def _binned_median(pts, n_bins=10):
    pts = sorted(pts)
    if len(pts) < n_bins:
        return [], []
    size = len(pts) / n_bins
    bx, by = [], []
    for b in range(n_bins):
        chunk = pts[int(b * size):int((b + 1) * size)]
        if chunk:
            bx.append(statistics.median([p[0] for p in chunk]))
            by.append(statistics.median([p[1] for p in chunk]))
    return bx, by


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/out/records.jsonl")
    out_img = Path(sys.argv[2]) if len(sys.argv) > 2 else path.parent / "savings_vs_size_all.png"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    any_data = False
    for tok in TOKENIZERS:
        pts = _points(rows, tok)
        if not pts:
            print(f"{tok}: no data (run --with-anthropic if anthropic)")
            continue
        any_data = True
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        r = _pearson(xs, ys)
        print(f"{tok}: n={len(pts)}  size range {min(xs)}..{max(xs)}  Pearson r={r:.3f}")
        ax.scatter(xs, ys, s=8, alpha=0.12, edgecolors="none", color=COLORS[tok])
        bx, by = _binned_median(pts)
        if bx:
            ax.plot(bx, by, color=COLORS[tok], linewidth=2.2, marker="o",
                    markersize=4, label=f"{tok} (r={r:.2f}, n={len(pts)})")

    if not any_data:
        print("No tokenizer columns populated -- nothing to plot.")
        return

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Original tool size -- normalized JSON tokens")
    ax.set_ylabel("CATS token reduction (%)  --  higher = CATS smaller")
    ax.set_title("Token savings vs. original tool size (all tokenizers)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, title="binned-median trend")
    fig.tight_layout()
    out_img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img, dpi=150)
    print(f"Wrote: {out_img}")


if __name__ == "__main__":
    main()