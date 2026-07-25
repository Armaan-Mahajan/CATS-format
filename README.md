# CATS — Compact Agent Tool Schema

CATS is a compact notation for the tool (function) definitions sent to large language models in tool-calling APIs. JSON Schema, the usual format, was designed to be parsed by machines; CATS is designed to be read by a language model. Encoding the same tool definition in CATS rather than JSON Schema reduces the number of input tokens the model must process, with a small and model-dependent effect on tool-calling accuracy.

This repository contains the format specification, a reference converter (JSON Schema ↔ CATS), and the full evaluation behind a forthcoming paper.

## Summary of findings

- **Token cost.** Across a public benchmark of real tool definitions (the Berkeley Function Calling Leaderboard, BFCL), CATS reduces per-tool input tokens by a median of roughly 30%, and this holds across the OpenAI, Qwen, and Anthropic tokenizers. Savings shrink as a tool grows larger, and there is a break-even point (around 6 tools in a prompt) below which the one-time format primer is not yet paid off.
- **Accuracy.** Measured against JSON Schema placed in the prompt, CATS carries a small accuracy cost that varies by model. On GPT-5.4 it is conclusively non-inferior at the pre-specified 3-percentage-point margin (−0.7 pp, p = 0.45). On Claude Sonnet 4.6 (−1.9 pp, p = 0.045) and Qwen3.5-35B-A3B (−2.6 pp, p = 0.005) non-inferiority is **inconclusive**: both show a statistically measurable gap whose confidence interval extends past the margin, so non-inferiority is neither confirmed nor ruled out.

CATS is presented as a token/accuracy trade-off, not a free lunch. The exact figures, confidence intervals, and per-model verdicts are committed under `results/`, and are set out in full in the forthcoming paper.

## Repository layout


| Path                                            | Contents                                                                               |
| ----------------------------------------------- | -------------------------------------------------------------------------------------- |
| `spec.md`                                       | The CATS format specification.                                                         |
| `protocol.md`                                   | How to use CATS with a model (the primer and output contract).                         |
| `converter_demo.py`, `converter_demo_inline.py` | Runnable examples of the converter.                                                    |
| `cats-converter/`                               | The reference converter library (JSON Schema ↔ CATS) and its test suite.               |
| `eval/`                                         | The evaluation harness — Part 1 (coverage and token cost) and Part 2 (model accuracy). |
| `results/`                                      | The raw data and figures behind every reported number.                                 |


## Installation

Requires Python ≥3.11, <3.14, and [uv](https://github.com/astral-sh/uv).

```bash
uv sync --project cats-converter
```

This installs the converter and its dependencies (including `bfcl-eval==2026.3.23`, the benchmark used in the evaluation) from public PyPI.

## Using the converter

The public API lives in `cats-converter/cats.py`. For a minimal end-to-end example, run:

```bash
uv run --project cats-converter python converter_demo_inline.py
```

`converter_demo.py` takes a JSON Schema tool definition as a command-line argument and prints its CATS form.

## Reproducing the evaluation

The `eval/` harness and the `cats-converter/tests/test_eval_part1.py` test expect the **full repository layout** — `eval/` and `cats-converter/` as siblings under the repository root. They will not run from an isolated copy of `cats-converter/` alone.

All commands below are run from the repository root.

### Part 1 — coverage and token cost

Library-only and reproducible offline:

```bash
uv run --project cats-converter python eval/run_part1.py --output-dir eval/out
```

Cached Anthropic token counts are committed under `eval/cache/`, so an Anthropic API key is not required; `eval/part1/anthropic_fetch.py` can refresh them if desired. The figures are produced by `eval/part1/coverage_charts.py`, `eval/part1/token_stats.py`, `eval/part1/savings_vs_size.py`, `eval/json_vs_cats_scatter.py`, and `eval/break_even.py`, each of which reads the `records.jsonl` written above. Note that the first run downloads tokenizer weights into `eval/.hf_cache/`.

### Part 2 — model accuracy

**The published statistics can be recomputed without re-running any inference.** The scored per-cell outputs are committed under `results/part2/raw/`, so this is the only Part 2 command most readers need:

```bash
uv run --project cats-converter python eval/part2/compute_stats.py
uv run --project cats-converter python eval/part2/compute_robustness.py
uv run --project cats-converter python eval/part2/plot_part2_graphs.py
```

Re-running the **inference** costs real money across three providers, so there is deliberately no single `run_part2.py`. It is a four-phase pipeline, run in order, with API keys in a repo-root `.env`:

| Phase | Script | Does |
| ----- | ------ | ---- |
| 1 | `eval/part2/full_run_phase1.py` | Builds and validates the run artifacts and prints a cost estimate. **Submits nothing** — it halts here on purpose so the cost can be reviewed first. |
| 2 | `eval/part2/full_run_phase2.py` | Submits the OpenAI and Anthropic batch jobs and starts the Qwen sync run. |
| 3 | `eval/part2/full_run_phase3.py` | Polls the batch jobs, fetches results, and builds the immutable raw store. Re-runnable; will not overwrite `raw_store.jsonl`. |
| 4 | `eval/part2/full_run_phase4.py` | Scores every cell from the raw store. Re-runnable. |

Phases 2–4 take a `--run-dir` argument identifying the run directory created by phase 1. The `live_multiple` subsample used in Part 2 is fixed in `eval/part2/live_multiple_subsample.json`; `eval/part2/sample_live_multiple.py` reproduces that draw from the recorded seed and verifies it against the committed IDs.

## License

MIT — see [LICENSE](LICENSE). © 2026 Armaan Mahajan.

## Citation

The paper describing CATS is in preparation. Until it is available, please cite this repository:

```bibtex
@misc{cats2026,
  author       = {Mahajan, Armaan},
  title        = {{CATS}: {Compact Agent Tool Schema}},
  year         = {2026},
  howpublished = {\url{https://github.com/Armaan-Mahajan/CATS-format}},
  note         = {Specification available as \texttt{spec.md}; usage protocol available as \texttt{protocol.md}}
}
```

