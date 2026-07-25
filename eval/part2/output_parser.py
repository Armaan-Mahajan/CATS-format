"""Ground-truth-blind parser for condition (a)/(b) model text responses.

Per ``protocol.md`` §5.2–5.3: locate a fenced JSON block, parse it, read
``name`` and ``arguments``.

Multiple fenced blocks
----------------------
If the model emits more than one fenced block (e.g. prose with an illustrative
example before the real answer), this parser uses the **last** fenced block.
Reasoning: models that reason in prose often place an example first and the
actual tool call last; taking the first block would score illustrative JSON as
the call. The rule is deterministic and does not inspect block contents.

This module never receives the expected tool name or ground truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

_FENCED_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


class ParseOutcome(str, Enum):
    """Syntactic parse result for a model text response."""

    VALID = "valid"
    NO_FENCED_BLOCK = "no_fenced_block"
    INVALID_JSON = "invalid_json"
    INVALID_SHAPE = "invalid_shape"


@dataclass(frozen=True)
class ParsedModelOutput:
    outcome: ParseOutcome
    name: str | None = None
    arguments: dict[str, Any] | None = None

    @property
    def syntactically_valid(self) -> bool:
        return self.outcome == ParseOutcome.VALID


def parse_tool_call_response(text: str) -> ParsedModelOutput:
    """Parse a model's free-text response into a tool call, or report syntactic failure."""
    if not text or not text.strip():
        return ParsedModelOutput(ParseOutcome.NO_FENCED_BLOCK)

    blocks = _FENCED_BLOCK_RE.findall(text)
    if not blocks:
        return ParsedModelOutput(ParseOutcome.NO_FENCED_BLOCK)

    payload_text = blocks[-1].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return ParsedModelOutput(ParseOutcome.INVALID_JSON)

    if not isinstance(payload, dict):
        return ParsedModelOutput(ParseOutcome.INVALID_SHAPE)

    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str):
        return ParsedModelOutput(ParseOutcome.INVALID_SHAPE)
    if not isinstance(arguments, dict):
        return ParsedModelOutput(ParseOutcome.INVALID_SHAPE)

    return ParsedModelOutput(
        outcome=ParseOutcome.VALID,
        name=name,
        arguments=arguments,
    )
