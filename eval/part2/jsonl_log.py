"""Authoritative JSONL request/response log (P2-4B)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "eval" / "part2" / "logs"


@dataclass(frozen=True)
class LogRecord:
    timestamp: str
    run_id: str
    mocked: bool
    entry_id: str
    category: str
    model: str
    condition: str
    provider: str
    skipped: bool
    skip_reason: str | None
    request_body: dict[str, Any] | None
    response_body: dict[str, Any] | None
    parse_outcome: str | None
    semantically_valid: bool | None
    notes: str | None = None


class JsonlRequestLogger:
    def __init__(self, *, provider: str, run_id: str) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"{provider}_{run_id}.jsonl"
        self.run_id = run_id
        self.provider = provider

    def append(self, record: LogRecord) -> Path:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return self.path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mock_response_body(mock_kind: str, *, prompt_text: str | None, native_output: Any) -> dict[str, Any]:
    return {
        "mock": True,
        "mock_kind": mock_kind,
        "prompt_text": prompt_text,
        "native_output": native_output,
    }
