#!/usr/bin/env python3
"""Scatter: original tool size (x, tokens) vs CATS percent savings (y, %).

Three side-by-side panels (one per tokenizer) with a log-scaled x-axis.
Each panel shows that tokenizer's faint scatter plus its equal-count
binned-median trend line. Bin edges are count-based (not log-spaced) and
must stay fixed so cited medians (e.g. ~18% on Qwen) do not move.
One point per converted unique tool.

X-axis: original (normalized JSON) token count, per tokenizer (log scale).
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
from matplotlib.ticker import ScalarFormatter  # noqa: E402
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
# Darker trend colors so the binned-median line reads over the faint scatter cloud.
TREND_COLORS = {"tiktoken": "#08519c", "qwen": "#a63603", "anthropic": "#006d2c"}
# Grayscale-safe: series differ by marker AND line style, not just color.
MARKERS = {"tiktoken": "o", "qwen": "s", "anthropic": "^"}
LINESTYLES = {"tiktoken": "-", "qwen": "--", "anthropic": ":"}

# Short per-panel headings (subplot labels, not a figure-level title).
PANEL_TITLES = {
    "tiktoken": "tiktoken",
    "qwen": "Qwen3.5-35B-A3B",
    "anthropic": "Anthropic",
}

# Plain-integer log ticks spanning the observed size range (~34..1310).
LOG_XTICKS = (50, 100, 200, 400, 800)


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

    series: list[tuple[str, list[tuple[float, float]], float, list[float], list[float]]] = []
    for tok in TOKENIZERS:
        pts = _points(rows, tok)
        if not pts:
            print(f"{tok}: no data (run --with-anthropic if anthropic)")
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        r = _pearson(xs, ys)
        bx, by = _binned_median(pts)
        print(f"{tok}: n={len(pts)}  size range {min(xs)}..{max(xs)}  Pearson r={r:.3f}")
        if by:
            print(f"  final binned-median: {by[-1]:.2f}%")
        series.append((tok, pts, r, bx, by))

    if not series:
        print("No tokenizer columns populated -- nothing to plot.")
        return

    all_xs = [p[0] for _, pts, _, _, _ in series for p in pts]
    all_ys = [p[1] for _, pts, _, _, _ in series for p in pts]
    # Pad log x-limits so the ~34..1310 tail is fully visible (no clipping).
    x_lo = min(all_xs) / 1.15
    x_hi = max(all_xs) * 1.15
    y_lo = min(0.0, min(all_ys) - 2.0)
    y_hi = 60.0

    fig, axes = plt.subplots(
        1, len(series), figsize=(6.7, 2.8), sharex=True, sharey=True, squeeze=False,
    )
    panels = axes[0]
    for ax, (tok, pts, r, bx, by) in zip(panels, series):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Rasterize only the scatter layer: text/axes stay vector, file stays small.
        # Keep scatter faint so the binned-median trend reads clearly on top.
        ax.scatter(
            xs, ys, s=4, alpha=0.10, edgecolors="none", color=COLORS[tok],
            marker=MARKERS[tok], rasterized=True, zorder=1,
        )
        if bx:
            ax.plot(
                bx, by, color=TREND_COLORS[tok], linewidth=2.2,
                linestyle=LINESTYLES[tok], marker=MARKERS[tok], markersize=4.0,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(LOG_XTICKS)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.minorticks_off()
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(PANEL_TITLES[tok], fontsize=9)
        ax.grid(alpha=0.3, which="major")
        # r / n annotation in the free upper-right corner (points trend down).
        ax.text(
            0.96, 0.96, f"r = {r:.2f}\nn = {len(pts)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
        )

    panels[0].set_ylabel("Token reduction (%)")
    panels[len(panels) // 2].set_xlabel(
        "Original tool size — normalized JSON tokens (log scale)"
    )
    fig.tight_layout(pad=0.4)
    _save_paper_figure(fig, out_img.stem, out_img.parent)


if __name__ == "__main__":
    main()