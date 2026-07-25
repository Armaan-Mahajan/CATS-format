"""Extract user-facing text from BFCL ``question`` fields."""

from __future__ import annotations

import copy
from typing import Any

from bfcl_eval.model_handler.utils import extract_system_prompt


def extract_question_turn(question: list[Any]) -> list[dict[str, Any]]:
    """Return the inner message list from a BFCL ``question`` value.

    Observed shape across all 985 Part 2 entries:

    - ``question`` is a ``list`` of length 1.
    - ``question[0]`` is a ``list`` of message dicts with ``role`` and ``content``.
    - Roles are ``user`` (always) and sometimes ``system`` (34 entries).
    """
    if not question or not isinstance(question, list):
        raise ValueError(f"expected non-empty question list, got {question!r}")
    if len(question) != 1:
        raise ValueError(
            f"expected exactly one conversation turn in question, got {len(question)}"
        )
    turn = question[0]
    if not isinstance(turn, list):
        raise ValueError(f"expected question[0] to be a message list, got {turn!r}")
    for index, message in enumerate(turn):
        if not isinstance(message, dict):
            raise ValueError(f"question message {index} is not a dict: {message!r}")
        if "role" not in message or "content" not in message:
            raise ValueError(f"question message {index} missing role/content: {message!r}")
    return turn


def split_question_messages(question: list[Any]) -> tuple[str | None, str]:
    """Split BFCL question turn into embedded system text and user text.

    Uses BFCL's own :func:`extract_system_prompt`, which removes the system
    message from a *copy* of the turn and returns its content (or ``None``).
    """
    turn = extract_question_turn(question)
    messages = copy.deepcopy(turn)
    embedded_system = extract_system_prompt(messages)
    user_parts = [str(message["content"]) for message in messages if message.get("role") == "user"]
    if not user_parts:
        raise ValueError(f"no user message in question turn after system extraction: {turn!r}")
    return embedded_system, user_parts[-1]


def extract_embedded_system_prompt(question: list[Any]) -> str | None:
    """Embedded ``system``-role content from the question field, if any."""
    embedded_system, _user_text = split_question_messages(question)
    return embedded_system


def extract_user_message(question: list[Any]) -> str:
    """User-facing prompt text only — never merged with embedded system content."""
    _embedded_system, user_text = split_question_messages(question)
    return user_text


def prepend_embedded_system(tool_system_prompt: str, embedded_system: str | None) -> str:
    """Prefix tool schema system prompt with embedded BFCL system text (conditions a/b)."""
    if embedded_system is None:
        return tool_system_prompt
    return f"{embedded_system}\n\n{tool_system_prompt}"


def entries_with_embedded_system(corpus: list[Any]) -> list[str]:
    """Return entry IDs whose question turn includes an embedded system message."""
    affected: list[str] = []
    for entry in corpus:
        turn = extract_question_turn(entry.question)
        if any(message.get("role") == "system" for message in turn):
            affected.append(entry.id)
    return affected
