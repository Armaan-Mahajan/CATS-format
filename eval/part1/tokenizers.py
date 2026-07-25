"""Token counting helpers for Part 1."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

import tiktoken
from transformers import AutoTokenizer

from eval.part1.constants import ANTHROPIC_COUNT_MODEL, QWEN_TOKENIZER_MODEL, TIKTOKEN_ENCODING


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenCounter:
    def __init__(self, encoding_name: str = TIKTOKEN_ENCODING) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


class QwenTokenizerCounter:
    def __init__(self, model_id: str = QWEN_TOKENIZER_MODEL) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


@lru_cache(maxsize=1)
def default_tiktoken_counter() -> TiktokenCounter:
    return TiktokenCounter()


@lru_cache(maxsize=1)
def default_qwen_counter() -> QwenTokenizerCounter:
    return QwenTokenizerCounter()


def count_anthropic_tokens(client: object, text: str, *, model: str = ANTHROPIC_COUNT_MODEL) -> int:
    response = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return int(response.input_tokens)


def ensure_hf_cache_in_repo(repo_root: os.PathLike[str] | str) -> None:
    """Point HF cache at eval/.hf_cache so tokenizer files stay in-repo."""
    hf_home = os.path.join(os.fspath(repo_root), "eval", ".hf_cache")
    os.environ.setdefault("HF_HOME", hf_home)
