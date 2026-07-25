"""Parse live provider responses into checker-ready model output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from eval.part2.checker import Part2Condition
from eval.part2.output_parser import ParseOutcome, parse_tool_call_response
from eval.part2.providers import Provider


@dataclass(frozen=True)
class ExtractedModelOutput:
    prompt_text: str | None
    model_output: list[dict[str, Any]] | None
    parse_outcome: ParseOutcome | None
    response_tool_name: str | None


def _parse_json_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"unexpected tool arguments type: {type(raw)!r}")


def extract_openai_prompt_text(response: dict[str, Any]) -> str:
    if response.get("output_text"):
        return str(response["output_text"])
    parts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)


def extract_openai_native_output(response: dict[str, Any]) -> list[dict[str, Any]] | None:
    calls: list[dict[str, Any]] = []
    for item in response.get("output", []):
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        calls.append({name: _parse_json_args(item.get("arguments", "{}"))})
    return calls or None


def extract_anthropic_prompt_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in response.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def extract_anthropic_native_output(response: dict[str, Any]) -> list[dict[str, Any]] | None:
    calls: list[dict[str, Any]] = []
    for block in response.get("content", []):
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        if not isinstance(name, str):
            continue
        calls.append({name: block.get("input") or {}})
    return calls or None


def extract_openrouter_prompt_text(response: dict[str, Any]) -> str:
    for choice in response.get("choices", []):
        content = (choice.get("message") or {}).get("content")
        if content:
            return str(content)
    return ""


def extract_openrouter_native_output(response: dict[str, Any]) -> list[dict[str, Any]] | None:
    calls: list[dict[str, Any]] = []
    for choice in response.get("choices", []):
        message = choice.get("message") or {}
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if not isinstance(name, str):
                continue
            calls.append({name: _parse_json_args(fn.get("arguments", "{}"))})
    return calls or None


def extract_model_output(
    *,
    provider: Provider,
    condition: Part2Condition,
    response: dict[str, Any],
) -> ExtractedModelOutput:
    if condition.is_prompt_mode:
        if provider == Provider.OPENAI:
            text = extract_openai_prompt_text(response)
        elif provider == Provider.ANTHROPIC:
            text = extract_anthropic_prompt_text(response)
        else:
            text = extract_openrouter_prompt_text(response)
        parsed = parse_tool_call_response(text)
        if not parsed.syntactically_valid:
            return ExtractedModelOutput(
                prompt_text=text,
                model_output=None,
                parse_outcome=parsed.outcome,
                response_tool_name=parsed.name,
            )
        assert parsed.name is not None and parsed.arguments is not None
        return ExtractedModelOutput(
            prompt_text=text,
            model_output=[{parsed.name: parsed.arguments}],
            parse_outcome=parsed.outcome,
            response_tool_name=parsed.name,
        )

    if provider == Provider.OPENAI:
        native = extract_openai_native_output(response)
    elif provider == Provider.ANTHROPIC:
        native = extract_anthropic_native_output(response)
    else:
        native = extract_openrouter_native_output(response)

    response_name = None
    if native and len(native) == 1:
        response_name = next(iter(native[0]))
    return ExtractedModelOutput(
        prompt_text=None,
        model_output=native,
        parse_outcome=ParseOutcome.VALID if native else ParseOutcome.NO_FENCED_BLOCK,
        response_tool_name=response_name,
    )
