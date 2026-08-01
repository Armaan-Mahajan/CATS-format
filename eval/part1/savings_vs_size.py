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
from matplotlib.transforms import Bbox  # noqa: E402

# Journal figure style (two-column A4, 170 mm text width, sans-serif body).
matplotlib.rcParams.update({
    "figure.figsize": (6.7, 2.8),   # 170mm text width; keep height tight
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,             # embed Type 1/TrueType, never Type 3
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = REPO_ROOT / "figures"

TOKENIZERS = ("tiktoken", "qwen", "anthropic")
COLORS = {"tiktoken": "tab:blue", "qwen": "tab:orange", "anthropic": "tab:green"}
# Grayscale-safe: series differ by marker AND line style, not just color.
MARKERS = {"tiktoken": "o", "qwen": "s", "anthropic": "^"}
LINESTYLES = {"tiktoken": "-", "qwen": "--", "anthropic": ":"}


def _save_paper_figure(fig, name: str, png_dir: Path) -> None:
    """Dual-save PDF (paper) + 300 dpi PNG (repo) from the same figure object.

    An explicit full-figure bbox pins the page to exactly 6.7 in wide so the
    PDF drops into ``\\linewidth`` without scaling.
    """
    full_bbox = Bbox([[0.0, 0.0], list(fig.get_size_inches())])
    for out, kw in ((FIGURES_DIR / f"{name}.pdf", {}), (png_dir / f"{name}.png", {"dpi": 300})):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches=full_bbox, **kw)
        print(f"Wrote {out}")


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

    fig, ax = plt.subplots(figsize=(6.7, 2.8))
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
        # Rasterize only the scatter layer: text/axes stay vector, file stays small.
        ax.scatter(xs, ys, s=5, alpha=0.18, edgecolors="none", color=COLORS[tok],
                   rasterized=True)
        bx, by = _binned_median(pts)
        if bx:
            ax.plot(bx, by, color=COLORS[tok], linewidth=1.6,
                    linestyle=LINESTYLES[tok], marker=MARKERS[tok],
                    markersize=3.5, label=f"{tok} (r={r:.2f}, n={len(pts)})")

    if not any_data:
        print("No tokenizer columns populated -- nothing to plot.")
        return

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Original tool size — normalized JSON tokens")
    ax.set_ylabel("Token reduction (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, title="binned-median trend")
    fig.tight_layout(pad=0.4)
    _save_paper_figure(fig, out_img.stem, out_img.parent)


if __name__ == "__main__":
    main()