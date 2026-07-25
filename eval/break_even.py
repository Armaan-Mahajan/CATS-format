#!/usr/bin/env python3
"""Calibrated break-even: when does CATS (per-feature primer overhead) beat raw JSON?

Synthetic projection by sampling converted tools from Part 1 records. The calibrated
primer is rebuilt per sample via the converter's real manifest-driven primer builder.

Run::

    uv run --project cats-converter python eval/break_even.py
    uv run --project cats-converter python eval/break_even.py eval/out/records.jsonl
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CATS_CONVERTER = REPO_ROOT / "cats-converter"
sys.path.insert(0, str(CATS_CONVERTER))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import cats  # noqa: E402
from eval.part1.anthropic_cache import AnthropicPrimerTokenCache  # noqa: E402
from eval.part1.constants import BUCKET_CONVERTED  # noqa: E402
from eval.part1.pipeline import (  # noqa: E402
    convert_unique_tools,
    load_bfcl_tools,
    validate_and_aggregate,
)
from eval.part1.tokenizers import (  # noqa: E402
    count_anthropic_tokens,
    default_qwen_counter,
    default_tiktoken_counter,
    ensure_hf_cache_in_repo,
)
from from_json import normalize_map_python_types  # noqa: E402
from nodes import Document, RawSchema, ToolBlock  # noqa: E402
from parser import parse_text  # noqa: E402
from primer import (  # noqa: E402
    _assemble_primer,
    assemble_prompt_sections,
    build_manifest,
    generate_primer_from_json,
)

# Auditable primer dependency (public API → manifest walk → assembly).
PRIMER_BUILDER = (
    "from_json_with_report(assume_closed=True, map_python_types=True) → "
    "build_manifest(Document) → _assemble_primer (same flags as Part 1); "
    "full-grammar worst-case uses generate_primer_from_json(..., full_grammar=True) "
    "(Part 1 flags by default)"
)

TOOL_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]
N_TRIALS = 1000
RNG_SEED = 42
SAVINGS_THRESHOLDS = (10, 15, 20, 25)
CEILING_NS = (50,)
TARGET_RPM = 85
ANTHROPIC_PRIMER_SAVE_EVERY = 25

TOKENIZER_SPECS = (
    ("tiktoken", "tiktoken o200k_base — GPT-5.X", "tokens_json_tiktoken", "tokens_cats_tiktoken"),
    ("qwen", "qwen — Qwen3.5-35B-A3B", "tokens_json_qwen", "tokens_cats_qwen"),
    ("anthropic", "anthropic — Claude Sonnet 4.6 / Opus 4.6", "tokens_json_anthropic", "tokens_cats_anthropic"),
)

TOKENIZER_COLORS = {
    "tiktoken": "tab:blue",
    "qwen": "tab:orange",
    "anthropic": "tab:green",
}

_PRIMER_DUMMY_TOOL = {
    "name": "break_even_placeholder",
    "description": "Placeholder for full-grammar primer generation.",
    "parameters": {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "additionalProperties": False,
    },
}


class _RequestRateLimiter:
    """Thread-safe minimum spacing between Anthropic request starts."""

    def __init__(self, target_rpm: float = TARGET_RPM) -> None:
        self._min_interval = 60.0 / target_rpm
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


def _fixed_cats_overhead_text(primer_text: str, manifest) -> str:
    """Primer + section framing + output contract (tool bodies counted separately)."""
    return assemble_prompt_sections(primer_text, "```\n\n```", manifest)


def _interpolate_break_even(
    ns: list[int],
    mean_json: list[float],
    mean_cats: list[float],
) -> float | None:
    diffs = [cats - json_ for json_, cats in zip(mean_json, mean_cats)]
    if diffs[0] <= 0:
        return float(ns[0]) if diffs[0] == 0 else 1.0
    for i in range(len(ns) - 1):
        if diffs[i] > 0 >= diffs[i + 1]:
            n1, n2 = ns[i], ns[i + 1]
            d1, d2 = diffs[i], diffs[i + 1]
            if d1 == d2:
                return float(n1)
            return n1 + d1 / (d1 - d2) * (n2 - n1)
    return None


def _interpolate_savings_threshold(
    ns: list[int],
    savings_pcts: list[float],
    threshold: float,
) -> float | None:
    """Return N where percent savings crosses ``threshold`` (linear interpolation)."""
    if savings_pcts[-1] < threshold:
        return None
    if savings_pcts[0] >= threshold:
        return float(ns[0])
    for i in range(len(ns) - 1):
        s1, s2 = savings_pcts[i], savings_pcts[i + 1]
        if s1 < threshold <= s2:
            n1, n2 = ns[i], ns[i + 1]
            if s2 == s1:
                return float(n1)
            return n1 + (threshold - s1) / (s2 - s1) * (n2 - n1)
    return None


def _per_tool_median_savings_pct(pool: list[dict], json_key: str, cats_key: str) -> float | None:
    pcts: list[float] = []
    for row in pool:
        json_tok = row.get(json_key)
        cats_tok = row.get(cats_key)
        if json_tok is None or cats_tok is None or json_tok <= 0:
            continue
        pcts.append((json_tok - cats_tok) / json_tok * 100)
    if not pcts:
        return None
    return float(statistics.median(pcts))


def _load_pool(records_path: Path) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in records_path.open(encoding="utf-8") if line.strip()]
    converted = [r for r in rows if r.get("bucket") == BUCKET_CONVERTED]
    return converted, len(rows) - len(converted)


def _build_tool_index(pool_hashes: set[str]) -> dict[str, dict]:
    """Re-convert BFCL tools for normalized schemas (primer builder input)."""
    print("Rebuilding tool index from BFCL (one-time)…")
    occurrences = load_bfcl_tools()
    aggregates, invalid_entries, _ = validate_and_aggregate(
        occurrences,
        normalize_map_python_types=normalize_map_python_types,
    )
    invalid_hashes = {agg.tool_hash for agg, _ in invalid_entries}
    invalid_reasons = {agg.tool_hash: reason for agg, reason in invalid_entries}
    results = convert_unique_tools(
        aggregates,
        convert_with_report=cats.convert_with_report,
        invalid_hashes=invalid_hashes,
        invalid_reasons=invalid_reasons,
    )
    index = {
        r.tool_hash: {"normalized_tool": r.normalized_tool}
        for r in results
        if r.bucket == BUCKET_CONVERTED
        and r.normalized_tool is not None
        and r.tool_hash in pool_hashes
    }
    missing = pool_hashes - set(index)
    if missing:
        raise SystemExit(
            f"normalized schemas missing for {len(missing)} converted record(s); re-run Part 1."
        )
    print(f"  indexed {len(index)} converted tools")
    return index


def _tools_from_normalized(normalized_tool: dict) -> list[ToolBlock | RawSchema]:
    conversion = cats.convert_with_report_for_tool_calling(normalized_tool)
    document = parse_text(conversion.cats_text)
    return list(document.tools)


def _calibrated_overhead_text(tool_index: dict[str, dict], hashes: list[str]) -> str:
    tools: list[ToolBlock | RawSchema] = []
    for tool_hash in dict.fromkeys(hashes):
        tools.extend(_tools_from_normalized(tool_index[tool_hash]["normalized_tool"]))
    manifest = build_manifest(Document(tools=tools))
    if manifest.all_fallback:
        raise RuntimeError("calibrated primer requested for all-fallback sample")
    return _fixed_cats_overhead_text(
        _assemble_primer(manifest, full_grammar=False),
        manifest,
    )


def _full_grammar_overhead_text() -> str:
    result = generate_primer_from_json(_PRIMER_DUMMY_TOOL, full_grammar=True)
    return _fixed_cats_overhead_text(result.primer_text, result.manifest)


class _Counters:
    def __init__(
        self,
        *,
        include_anthropic: bool,
        primer_cache: AnthropicPrimerTokenCache | None = None,
    ) -> None:
        self.tiktoken = default_tiktoken_counter()
        self.qwen = default_qwen_counter()
        self._anthropic = None
        self._primer_cache = primer_cache
        self._limiter: _RequestRateLimiter | None = None
        self._anthropic_fetches = 0
        if include_anthropic:
            import os

            if os.environ.get("ANTHROPIC_API_KEY_PART1"):
                try:
                    from anthropic import Anthropic

                    self._anthropic = Anthropic()
                    if primer_cache is not None:
                        self._limiter = _RequestRateLimiter()
                except Exception:
                    pass

    def _count_anthropic_api(self, text: str) -> int:
        from anthropic import RateLimitError

        assert self._anthropic is not None
        assert self._limiter is not None
        for attempt in range(6):
            try:
                self._limiter.acquire()
                return count_anthropic_tokens(self._anthropic, text)
            except RateLimitError:
                time.sleep(2**attempt + random.uniform(0, 1))
            except Exception as exc:
                message = str(exc)
                if "429" in message or "rate" in message.lower():
                    time.sleep(2**attempt + random.uniform(0, 1))
                    continue
                raise
        raise RuntimeError("Anthropic rate limit retries exhausted for primer overhead text")

    def count_anthropic_primer(self, text: str) -> int | None:
        if self._anthropic is None or self._primer_cache is None:
            return None
        cached = self._primer_cache.get(text)
        if cached is not None:
            return cached
        count = self._count_anthropic_api(text)
        self._primer_cache.set(text, count)
        self._anthropic_fetches += 1
        if self._anthropic_fetches % ANTHROPIC_PRIMER_SAVE_EVERY == 0:
            self._primer_cache.save()
        return count

    def count(self, tok: str, text: str) -> int | None:
        if tok == "tiktoken":
            return self.tiktoken.count(text)
        if tok == "qwen":
            return self.qwen.count(text)
        if tok == "anthropic":
            return self.count_anthropic_primer(text)
        return None


def _run_calibrated_trials(
    pool: list[dict],
    tool_index: dict[str, dict],
    counters: _Counters,
    *,
    tok: str,
    json_key: str,
    cats_key: str,
    n_tools: int,
    n_trials: int,
    rng: random.Random,
    overhead_cache: dict[frozenset[str], int],
) -> tuple[float, float, float, float, float]:
    json_totals: list[float] = []
    cats_totals: list[float] = []
    primer_totals: list[float] = []
    wins = 0

    for _ in range(n_trials):
        sample = [rng.choice(pool) for _ in range(n_tools)]
        hashes = [r["tool_hash"] for r in sample]
        feature_key = frozenset(hashes)

        if feature_key not in overhead_cache:
            overhead_text = _calibrated_overhead_text(tool_index, hashes)
            count = counters.count(tok, overhead_text)
            if count is None:
                raise RuntimeError(f"tokenizer {tok!r} failed on calibrated overhead")
            overhead_cache[feature_key] = count

        overhead_tokens = overhead_cache[feature_key]
        json_cost = sum(r[json_key] for r in sample)
        cats_cost = sum(r[cats_key] for r in sample) + overhead_tokens
        json_totals.append(json_cost)
        cats_totals.append(cats_cost)
        primer_totals.append(overhead_tokens)
        if cats_cost < json_cost:
            wins += 1

    mean_json = sum(json_totals) / n_trials
    mean_cats = sum(cats_totals) / n_trials
    mean_primer = sum(primer_totals) / n_trials
    mean_savings = (mean_json - mean_cats) / mean_json * 100 if mean_json else 0.0
    return mean_json, mean_cats, mean_savings, wins / n_trials, mean_primer


def _run_constant_primer_trials(
    pool: list[dict],
    *,
    tok: str,
    json_key: str,
    cats_key: str,
    primer_tokens: int,
    n_tools: int,
    n_trials: int,
    rng: random.Random,
) -> tuple[float, float, float, float]:
    json_totals: list[float] = []
    cats_totals: list[float] = []
    wins = 0
    for _ in range(n_trials):
        sample = [rng.choice(pool) for _ in range(n_tools)]
        json_cost = sum(r[json_key] for r in sample)
        cats_cost = sum(r[cats_key] for r in sample) + primer_tokens
        json_totals.append(json_cost)
        cats_totals.append(cats_cost)
        if cats_cost < json_cost:
            wins += 1
    mean_json = sum(json_totals) / n_trials
    mean_cats = sum(cats_totals) / n_trials
    mean_savings = (mean_json - mean_cats) / mean_json * 100 if mean_json else 0.0
    return mean_json, mean_cats, mean_savings, wins / n_trials


def _annotate_savings_thresholds(
    ax: plt.Axes,
    *,
    tok: str,
    threshold_results: dict[str, dict],
) -> None:
    """Mark percent-savings crossing N on a mean-tokens panel (vertical guides)."""
    ymin = ax.get_ylim()[0]
    # Stack above break-even labels (12 / 28 pt) but stay inside the plot.
    y_offsets = (42, 54, 66, 78)
    for idx, threshold in enumerate(SAVINGS_THRESHOLDS):
        entry = threshold_results.get(str(threshold), {})
        if not entry.get("reached"):
            continue
        n_cross = entry.get("n")
        if n_cross is None:
            continue
        ax.axvline(n_cross, color="tab:purple", linestyle=":", linewidth=0.9, alpha=0.55)
        ax.annotate(
            f"{threshold}% saved @ N≈{n_cross:.1f}",
            (n_cross, ymin),
            xytext=(6, y_offsets[idx % len(y_offsets)]),
            textcoords="offset points",
            fontsize=6,
            color="tab:purple",
        )


def _plot_tokenizer_panel(
    ax: plt.Axes,
    *,
    tok: str,
    cal: dict,
    fg: dict | None,
    title: str,
    threshold_results: dict[str, dict],
) -> None:
    ns = TOOL_COUNTS
    mean_json = [cal["by_n"][str(n)]["mean_json"] for n in ns]
    mean_cats = [cal["by_n"][str(n)]["mean_cats"] for n in ns]

    ax.plot(ns, mean_json, "o:", color="tab:blue", linewidth=2, label="JSON tools")
    ax.plot(
        ns,
        mean_cats,
        "s-",
        color="tab:orange",
        linewidth=2,
        label="CATS + calibrated primer",
    )
    if fg:
        mean_fg = [fg["by_n"][str(n)]["mean_cats"] for n in ns]
        ax.plot(
            ns,
            mean_fg,
            "^:",
            color="gray",
            linewidth=1.2,
            alpha=0.55,
            label="uncalibrated worst case (full grammar)",
        )

    ax.set_ylabel("Mean prompt tokens")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)

    be = cal.get("break_even_n")
    if be is not None:
        ax.axvline(be, color="tab:green", linestyle=":", linewidth=1.5)
        ax.annotate(
            f"calibrated break-even ≈ {be:.1f}",
            (be, ax.get_ylim()[0]),
            xytext=(6, 12),
            textcoords="offset points",
            fontsize=8,
            color="tab:green",
        )
    if fg and fg.get("break_even_n") is not None:
        be_fg = fg["break_even_n"]
        ax.axvline(be_fg, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.annotate(
            f"full-grammar ≈ {be_fg:.1f}",
            (be_fg, ax.get_ylim()[0]),
            xytext=(6, 28),
            textcoords="offset points",
            fontsize=7,
            color="gray",
        )

    mp = cal["by_n"][str(ns[0])].get("mean_primer_tokens")
    if mp is not None:
        ax.annotate(
            f"mean calibrated primer @ N={ns[0]}: {mp:.0f} tok",
            (0.02, 0.98),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

    _annotate_savings_thresholds(ax, tok=tok, threshold_results=threshold_results)


def _plot_combined_tokens_panel(
    ax: plt.Axes,
    *,
    calibrated: dict[str, dict],
) -> None:
    """Mean CATS prompt tokens vs N — all tokenizers on one axis."""
    ns = TOOL_COUNTS
    for tok, color in TOKENIZER_COLORS.items():
        cal = calibrated.get(tok)
        if cal is None:
            continue
        mean_cats = [cal["by_n"][str(n)]["mean_cats"] for n in ns]
        ax.plot(
            ns,
            mean_cats,
            color=color,
            linewidth=2.2,
            marker="s",
            markersize=3,
            label=f"{tok} (CATS + primer)",
        )

    ax.set_ylabel("Mean prompt tokens")
    ax.set_title("Mean CATS prompt tokens vs. N (all tokenizers)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, title="tokenizers")


def _plot_break_even(
    out_path: Path,
    calibrated: dict[str, dict],
    full_grammar: dict[str, dict],
    labels: dict[str, str],
    *,
    threshold_results: dict[str, dict],
    asymptotic_medians: dict[str, float | None],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panel_map = {
        "tiktoken": axes[0, 0],
        "qwen": axes[0, 1],
        "anthropic": axes[1, 0],
    }
    for tok, ax in panel_map.items():
        if tok not in calibrated:
            ax.set_visible(False)
            continue
        _plot_tokenizer_panel(
            ax,
            tok=tok,
            cal=calibrated[tok],
            fg=full_grammar.get(tok),
            title=labels.get(tok, tok),
            threshold_results=threshold_results.get(tok, {}),
        )

    _plot_combined_tokens_panel(axes[1, 1], calibrated=calibrated)

    for ax in axes.flat:
        ax.set_xlabel("Number of tools in prompt (sampled with replacement)")
        ax.set_xticks(TOOL_COUNTS)
        ax.tick_params(axis="x", labelsize=6, rotation=45)

    medians = [asymptotic_medians.get(tok) for tok in ("tiktoken", "qwen", "anthropic")]
    if all(m is not None for m in medians):
        footnote = (
            "as N→∞, percent savings approaches the per-tool median reduction "
            f"(~{medians[0]:.0f}%/{medians[1]:.0f}%/{medians[2]:.0f}%), "
            "since the fixed primer cost becomes negligible relative to total prompt size."
        )
    else:
        footnote = (
            "as N→∞, percent savings approaches the per-tool median reduction, "
            "since the fixed primer cost becomes negligible relative to total prompt size."
        )
    fig.text(0.5, 0.01, footnote, ha="center", fontsize=8, style="italic")

    fig.suptitle(
        "CATS break-even (calibrated primer; synthetic N-tool prompts)",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def _threshold_report(
    tok: str,
    savings_series: list[float],
    asymptotic_median: float | None,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for threshold in SAVINGS_THRESHOLDS:
        n_cross = _interpolate_savings_threshold(TOOL_COUNTS, savings_series, threshold)
        if n_cross is None:
            entry: dict = {
                "reached": False,
                "n": None,
            }
            if asymptotic_median is not None:
                entry["asymptotic_median_pct_per_tool"] = round(asymptotic_median, 2)
                entry["note"] = (
                    f"not reached by N={TOOL_COUNTS[-1]} "
                    f"(asymptotes toward ~{asymptotic_median:.1f}% per-tool median)"
                )
            else:
                entry["note"] = f"not reached by N={TOOL_COUNTS[-1]}"
        else:
            entry = {
                "reached": True,
                "n": round(n_cross, 3),
            }
        out[str(threshold)] = entry
    return out


def main() -> None:
    records_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "eval" / "out" / "records.jsonl"
    out_dir = records_path.parent
    out_json = out_dir / "break_even.json"
    out_png = out_dir / "break_even.png"
    primer_cache_path = REPO_ROOT / "eval" / "cache" / "anthropic_primer_tokens.json"

    ensure_hf_cache_in_repo(REPO_ROOT)

    pool, excluded = _load_pool(records_path)
    if not pool:
        raise SystemExit(f"No converted tools in {records_path}")

    pool_hashes = {r["tool_hash"] for r in pool}
    tool_index = _build_tool_index(pool_hashes)

    has_anthropic_cols = all(r.get("tokens_json_anthropic") is not None for r in pool)
    primer_cache = AnthropicPrimerTokenCache(primer_cache_path)
    counters = _Counters(include_anthropic=has_anthropic_cols, primer_cache=primer_cache)
    if has_anthropic_cols and counters._anthropic is None:
        print("WARNING: anthropic per-tool columns present but API client unavailable — anthropic panel skipped.")

    full_overhead = _full_grammar_overhead_text()
    full_primer_by_tok: dict[str, int] = {}
    for tok, _, _, _ in TOKENIZER_SPECS:
        if tok == "anthropic" and not has_anthropic_cols:
            continue
        if tok == "anthropic" and counters._anthropic is None:
            continue
        count = counters.count(tok, full_overhead)
        if count is not None:
            full_primer_by_tok[tok] = count
    if counters._anthropic is not None:
        primer_cache.save()

    print()
    print("CATS calibrated break-even (synthetic projection)")
    print(f"  Primer builder: {PRIMER_BUILDER}")
    print("  Method: sampling with replacement from converted tools; fallen-back tools")
    print("          excluded (raw JSON in both arms → no per-tool savings).")
    print("  Calibrated primer: rebuilt per sample from union of sampled tools' features.")
    print("  Note: break-even is an average over the corpus feature mix; rarer features")
    print("        inflate primer size — depends on feature diversity, not N alone.")
    print(f"  Pool: {len(pool)} converted tools ({excluded} non-converted excluded)")
    print(f"  RNG seed: {RNG_SEED}")
    print(f"  Trials per N: {N_TRIALS}")
    print(f"  N values: {TOOL_COUNTS}")
    if has_anthropic_cols and counters._anthropic is not None:
        print(f"  Anthropic primer cache: {primer_cache_path} ({len(primer_cache)} entries at start)")
    print()

    rng = random.Random(RNG_SEED)
    labels = {key: label for key, label, _, _ in TOKENIZER_SPECS}

    asymptotic_medians: dict[str, float | None] = {
        tok: _per_tool_median_savings_pct(pool, json_key, cats_key)
        for tok, _, json_key, cats_key in TOKENIZER_SPECS
    }

    output: dict = {
        "primer_builder": PRIMER_BUILDER,
        "method_note": (
            "Synthetic projection with replacement from converted pool. "
            "Calibrated primer per sample via build_manifest + _assemble_primer "
            "(Part 1 conversion flags). "
            "Fallen-back tools excluded."
        ),
        "rng_seed": RNG_SEED,
        "n_trials": N_TRIALS,
        "tool_counts": TOOL_COUNTS,
        "pool_converted": len(pool),
        "pool_excluded_non_converted": excluded,
        "per_tool_median_savings_pct": {
            tok: round(m, 2) if m is not None else None for tok, m in asymptotic_medians.items()
        },
        "percent_savings_thresholds": {},
        "calibrated_vs_full_grammar_ratio": {},
        "tokenizers": {},
    }

    calibrated_plot: dict[str, dict] = {}
    full_grammar_plot: dict[str, dict] = {}
    threshold_results: dict[str, dict] = {}

    print(f"{'tokenizer':<42} {'cal BE':>8} {'full BE':>8} {'full primer':>12}")
    print("-" * 74)

    overhead_cache: dict[str, dict[frozenset[str], int]] = {
        tok: {} for tok, _, _, _ in TOKENIZER_SPECS
    }

    for tok, label, json_key, cats_key in TOKENIZER_SPECS:
        if tok == "anthropic" and (not has_anthropic_cols or counters._anthropic is None):
            print(f"WARNING: skipping {tok} calibrated panel — columns or API unavailable.")
            continue
        if not all(r.get(json_key) is not None and r.get(cats_key) is not None for r in pool):
            print(f"WARNING: skipping {tok} — token columns missing.")
            continue

        cal_by_n: dict[str, dict] = {}
        cal_json_series: list[float] = []
        cal_cats_series: list[float] = []
        savings_series: list[float] = []

        for n in TOOL_COUNTS:
            mean_json, mean_cats, mean_savings, win_rate, mean_primer = _run_calibrated_trials(
                pool,
                tool_index,
                counters,
                tok=tok,
                json_key=json_key,
                cats_key=cats_key,
                n_tools=n,
                n_trials=N_TRIALS,
                rng=rng,
                overhead_cache=overhead_cache[tok],
            )
            cal_by_n[str(n)] = {
                "mean_json": round(mean_json, 2),
                "mean_cats": round(mean_cats, 2),
                "mean_savings_pct": round(mean_savings, 2),
                "mean_primer_tokens": round(mean_primer, 2),
                "win_rate": round(win_rate, 4),
                "n_trials": N_TRIALS,
            }
            cal_json_series.append(mean_json)
            cal_cats_series.append(mean_cats)
            savings_series.append(mean_savings)

        if tok == "anthropic":
            primer_cache.save()

        cal_be = _interpolate_break_even(TOOL_COUNTS, cal_json_series, cal_cats_series)
        tok_thresholds = _threshold_report(tok, savings_series, asymptotic_medians.get(tok))
        threshold_results[tok] = tok_thresholds
        output["percent_savings_thresholds"][tok] = tok_thresholds

        fg_by_n: dict[str, dict] = {}
        fg_json_series: list[float] = []
        fg_cats_series: list[float] = []
        fg_primer = full_primer_by_tok.get(tok)
        if fg_primer is not None:
            for n in TOOL_COUNTS:
                mean_json, mean_cats, mean_savings, win_rate = _run_constant_primer_trials(
                    pool,
                    tok=tok,
                    json_key=json_key,
                    cats_key=cats_key,
                    primer_tokens=fg_primer,
                    n_tools=n,
                    n_trials=N_TRIALS,
                    rng=rng,
                )
                fg_by_n[str(n)] = {
                    "mean_json": round(mean_json, 2),
                    "mean_cats": round(mean_cats, 2),
                    "mean_savings_pct": round(mean_savings, 2),
                    "win_rate": round(win_rate, 4),
                    "n_trials": N_TRIALS,
                }
                fg_json_series.append(mean_json)
                fg_cats_series.append(mean_cats)
        fg_be = (
            _interpolate_break_even(TOOL_COUNTS, fg_json_series, fg_cats_series)
            if fg_primer is not None
            else None
        )

        ceiling: dict[str, dict] = {}
        if fg_primer is not None:
            for ceiling_n in CEILING_NS:
                key = str(ceiling_n)
                if key in cal_by_n:
                    mean_primer = cal_by_n[key]["mean_primer_tokens"]
                    ratio = mean_primer / fg_primer if fg_primer else None
                    ceiling[f"n_{ceiling_n}"] = {
                        "mean_calibrated_primer_tokens": mean_primer,
                        "full_grammar_primer_tokens": fg_primer,
                        "ratio_calibrated_to_full": round(ratio, 4) if ratio is not None else None,
                    }
        output["calibrated_vs_full_grammar_ratio"][tok] = ceiling

        cal_display = f"{cal_be:.2f}" if cal_be is not None else f"> {TOOL_COUNTS[-1]}"
        fg_display = f"{fg_be:.2f}" if fg_be is not None else "n/a"
        fg_primer_display = str(fg_primer) if fg_primer is not None else "—"
        print(f"{label:<42} {cal_display:>8} {fg_display:>8} {fg_primer_display:>12}")

        if cal_be is not None and cal_be > 1.0:
            print(f"  → CATS loses for small prompts; calibrated break-even ≈ {cal_be:.1f} tools.")
        elif cal_be is not None and cal_be <= 1.0:
            print(f"  → Calibrated CATS already cheaper at N=1 (break-even ≤ 1).")
        else:
            print(f"  → Calibrated CATS does not break even by N={TOOL_COUNTS[-1]}.")

        if fg_be is not None:
            print(
                f"  → Uncalibrated full-grammar worst case breaks even at ≈ {fg_be:.1f} tools "
                f"(primer+framing={fg_primer} tok)."
            )

        for threshold in SAVINGS_THRESHOLDS:
            entry = tok_thresholds[str(threshold)]
            if entry["reached"]:
                print(f"  → {threshold}% savings at N ≈ {entry['n']:.1f}")
            else:
                print(f"  → {threshold}% savings: {entry.get('note', 'not reached')}")

        for ceiling_n in CEILING_NS:
            ckey = f"n_{ceiling_n}"
            if ckey in ceiling:
                ratio = ceiling[ckey]["ratio_calibrated_to_full"]
                mp = ceiling[ckey]["mean_calibrated_primer_tokens"]
                print(
                    f"  → N={ceiling_n} calibrated/full primer ratio: {ratio:.3f} "
                    f"({mp:.0f} / {fg_primer} tok)"
                )

        tok_out = {
            "label": label,
            "calibrated": {
                "break_even_n": round(cal_be, 3) if cal_be is not None else None,
                "by_n": cal_by_n,
            },
            "full_grammar_worst_case": {
                "primer_tokens": fg_primer,
                "break_even_n": round(fg_be, 3) if fg_be is not None else None,
                "by_n": fg_by_n,
            },
            "percent_savings_thresholds": tok_thresholds,
            "calibrated_vs_full_grammar_ratio": ceiling,
        }
        output["tokenizers"][tok] = tok_out
        calibrated_plot[tok] = tok_out["calibrated"]
        if fg_by_n:
            full_grammar_plot[tok] = tok_out["full_grammar_worst_case"]

    print()
    out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")

    _plot_break_even(
        out_png,
        calibrated_plot,
        full_grammar_plot,
        labels,
        threshold_results=threshold_results,
        asymptotic_medians=asymptotic_medians,
    )
    print(f"Wrote {out_png}")
    if counters._anthropic is not None:
        primer_cache.save()
        print(f"Anthropic primer cache: {primer_cache_path} ({len(primer_cache)} entries)")


if __name__ == "__main__":
    main()
