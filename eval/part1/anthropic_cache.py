"""Anthropic token-count caches for Part 1 and break-even primer overhead."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


def anthropic_text_key(text: str) -> str:
    """Stable cache key for arbitrary text (SHA-256 hex)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AnthropicTokenCache:
    """Persistent flat dict: tool_hash → {tokens_json_anthropic, tokens_cats_anthropic}."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, int]] = {}
        if path.is_file():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, tool_hash: str) -> dict[str, int] | None:
        with self._lock:
            entry = self._data.get(tool_hash)
            return dict(entry) if entry is not None else None

    def set(self, tool_hash: str, tokens_json: int, tokens_cats: int) -> None:
        with self._lock:
            self._data[tool_hash] = {
                "tokens_json_anthropic": tokens_json,
                "tokens_cats_anthropic": tokens_cats,
            }

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class AnthropicPrettyTokenCache:
    """Persistent flat dict: tool_hash → tokens_json_pretty_anthropic."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, int] = {}
        if path.is_file():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, tool_hash: str) -> int | None:
        with self._lock:
            value = self._data.get(tool_hash)
            return int(value) if value is not None else None

    def set(self, tool_hash: str, tokens_json: int) -> None:
        with self._lock:
            self._data[tool_hash] = int(tokens_json)

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class AnthropicPrimerTokenCache:
    """Persistent flat dict: primer-text hash → input token count."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, int] = {}
        if path.is_file():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, text: str) -> int | None:
        key = anthropic_text_key(text)
        with self._lock:
            value = self._data.get(key)
            return int(value) if value is not None else None

    def set(self, text: str, count: int) -> None:
        key = anthropic_text_key(text)
        with self._lock:
            self._data[key] = int(count)

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
