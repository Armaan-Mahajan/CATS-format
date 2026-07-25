"""BFCL category list and paths for Part 1."""

from __future__ import annotations

INCLUDED_CATEGORIES: tuple[str, ...] = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "irrelevance",
    "live_irrelevance",
    "live_relevance",
)

QWEN_TOKENIZER_MODEL = "Qwen/Qwen3.5-35B-A3B"
ANTHROPIC_COUNT_MODEL = "claude-sonnet-4-6"
TIKTOKEN_ENCODING = "o200k_base"

BUCKET_CONVERTED = "converted"
BUCKET_FELL_BACK = "fell_back"
BUCKET_INVALID_INPUT = "invalid_input"
