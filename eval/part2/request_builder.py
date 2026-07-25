"""Build provider request bodies for all Part 2 conditions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from primer import build_system_prompt, generate_primer_from_json

from eval.part2.checker import Part2Condition
from eval.part2.corpus import CorpusEntry
from eval.part2.json_prompt import build_json_system_prompt
from eval.part2.providers import MODEL_CONFIGS, Provider
from eval.part2.question import (
    extract_embedded_system_prompt,
    extract_user_message,
    prepend_embedded_system,
)
from eval.part2.settings import (
    anthropic_messages_body,
    openai_responses_body,
    openrouter_chat_body,
)
from eval.part2.tools_prep import (
    native_api_tool_name,
    normalize_entry_tools,
    tools_for_native_api,
)


@dataclass(frozen=True)
class PreparedRequest:
    """Everything needed to call (or mock) one model×condition×entry cell."""

    entry_id: str
    category: str
    model: str
    condition: Part2Condition
    provider: Provider
    skipped: bool
    skip_reason: str | None
    user_message: str
    embedded_system_prompt: str | None
    system_prompt: str | None
    request_body: dict[str, Any] | None
    native_tool_names: tuple[str, ...]


def _openai_native_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["parameters"],
            "strict": False,
        }
        for tool in tools
    ]


def _anthropic_native_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool["parameters"],
        }
        for tool in tools
    ]


def _openrouter_native_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


def _cats_all_fallback(tools: list[dict[str, Any]]) -> bool:
    primer = generate_primer_from_json(copy.deepcopy(tools))
    return primer.all_fallback


def build_request(
    entry: CorpusEntry,
    *,
    model: str,
    condition: Part2Condition,
) -> PreparedRequest:
    """Construct the API request for one eval cell (no network I/O)."""
    config = MODEL_CONFIGS[model]
    embedded_system = extract_embedded_system_prompt(entry.question)
    user_message = extract_user_message(entry.question)
    normalized_tools = normalize_entry_tools(entry.function)

    if condition == Part2Condition.CATS_IN_PROMPT:
        if _cats_all_fallback(normalized_tools):
            return PreparedRequest(
                entry_id=entry.id,
                category=entry.category,
                model=model,
                condition=condition,
                provider=config.provider,
                skipped=True,
                skip_reason="cats_all_fallback",
                user_message=user_message,
                embedded_system_prompt=embedded_system,
                system_prompt=None,
                request_body=None,
                native_tool_names=(),
            )
        primer = generate_primer_from_json(copy.deepcopy(normalized_tools))
        tool_system_prompt = build_system_prompt(primer)
        system_prompt = prepend_embedded_system(tool_system_prompt, embedded_system)
        if config.provider == Provider.OPENAI:
            body = openai_responses_body(
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        elif config.provider == Provider.ANTHROPIC:
            body = anthropic_messages_body(
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        else:
            body = openrouter_chat_body(
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        return PreparedRequest(
            entry_id=entry.id,
            category=entry.category,
            model=model,
            condition=condition,
            provider=config.provider,
            skipped=False,
            skip_reason=None,
            user_message=user_message,
            embedded_system_prompt=embedded_system,
            system_prompt=system_prompt,
            request_body=body,
            native_tool_names=(),
        )

    if condition == Part2Condition.JSON_IN_PROMPT:
        tool_system_prompt = build_json_system_prompt(normalized_tools)
        system_prompt = prepend_embedded_system(tool_system_prompt, embedded_system)
        if config.provider == Provider.OPENAI:
            body = openai_responses_body(
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        elif config.provider == Provider.ANTHROPIC:
            body = anthropic_messages_body(
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        else:
            body = openrouter_chat_body(
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        return PreparedRequest(
            entry_id=entry.id,
            category=entry.category,
            model=model,
            condition=condition,
            provider=config.provider,
            skipped=False,
            skip_reason=None,
            user_message=user_message,
            embedded_system_prompt=embedded_system,
            system_prompt=system_prompt,
            request_body=body,
            native_tool_names=(),
        )

    native_tools = tools_for_native_api(
        normalized_tools,
        renames_dots=config.native_renames_dots,
    )
    native_names = tuple(tool["name"] for tool in native_tools)
    if config.provider == Provider.OPENAI:
        api_tools = _openai_native_tools(native_tools)
        body = openai_responses_body(
            model=model,
            system_prompt=None,
            user_message=user_message,
            tools=api_tools,
            embedded_system=embedded_system,
        )
    elif config.provider == Provider.ANTHROPIC:
        api_tools = _anthropic_native_tools(native_tools)
        body = anthropic_messages_body(
            model=model,
            system_prompt=embedded_system,
            user_message=user_message,
            tools=api_tools,
        )
    else:
        api_tools = _openrouter_native_tools(native_tools)
        body = openrouter_chat_body(
            model=model,
            system_prompt=embedded_system,
            user_message=user_message,
            tools=api_tools,
        )
    return PreparedRequest(
        entry_id=entry.id,
        category=entry.category,
        model=model,
        condition=condition,
        provider=config.provider,
        skipped=False,
        skip_reason=None,
        user_message=user_message,
        embedded_system_prompt=embedded_system,
        system_prompt=embedded_system,
        request_body=body,
        native_tool_names=native_names,
    )


def original_tool_names(entry: CorpusEntry) -> tuple[str, ...]:
    return tuple(tool["name"] for tool in entry.function)


def expected_response_tool_name(
    original_name: str,
    *,
    condition: Part2Condition,
    native_renames_dots: bool,
) -> str:
    if condition == Part2Condition.NATIVE_TOOLS and native_renames_dots:
        return native_api_tool_name(original_name, renames_dots=True)
    return original_name
