"""P2-4 inference settings shared across conditions within each model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval.part2.providers import OPENROUTER_ROUTING, Provider


@dataclass(frozen=True)
class InferenceSettings:
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 512


OPENAI_SETTINGS = InferenceSettings()
ANTHROPIC_SETTINGS = InferenceSettings()
OPENROUTER_SETTINGS = InferenceSettings()


def openai_responses_body(
    *,
    model: str,
    system_prompt: str | None,
    user_message: str,
    tools: list[dict[str, Any]] | None = None,
    embedded_system: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI Responses API body for prompt or native mode.

    For native mode, embedded BFCL system text is sent as a ``developer``-role
  message (matching BFCL's OpenAI Responses handler), not merged into user text.
    """
    body: dict[str, Any] = {
        "model": model,
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "medium"},
        "temperature": OPENAI_SETTINGS.temperature,
        "top_p": OPENAI_SETTINGS.top_p,
        "max_output_tokens": OPENAI_SETTINGS.max_output_tokens,
        "store": True,
    }
    if tools is None:
        body["instructions"] = system_prompt
        body["input"] = user_message
    else:
        input_messages: list[dict[str, str]] = []
        if embedded_system is not None:
            input_messages.append({"role": "developer", "content": embedded_system})
        input_messages.append({"role": "user", "content": user_message})
        body["input"] = input_messages
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


def anthropic_messages_body(
    *,
    model: str,
    system_prompt: str | None,
    user_message: str,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": ANTHROPIC_SETTINGS.max_output_tokens,
        "temperature": ANTHROPIC_SETTINGS.temperature,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "medium"},
        "metadata": {"user_id": "cats-part2-eval"},
        "messages": [{"role": "user", "content": user_message}],
    }
    if system_prompt is not None:
        body["system"] = system_prompt
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = {"type": "auto"}
    return body


def openrouter_chat_body(
    *,
    model: str,
    system_prompt: str | None,
    user_message: str,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": OPENROUTER_SETTINGS.temperature,
        "top_p": OPENROUTER_SETTINGS.top_p,
        "max_tokens": OPENROUTER_SETTINGS.max_output_tokens,
        "reasoning": {"effort": "none"},
        **OPENROUTER_ROUTING,
    }
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


def provider_for_model(provider: Provider) -> Provider:
    return provider
