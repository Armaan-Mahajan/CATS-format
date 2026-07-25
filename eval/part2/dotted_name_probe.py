#!/usr/bin/env python3
"""Probe whether pinned Part 2 models accept or rename dotted native tool names.

Run from cats-converter/:

    uv run python ../eval/part2/dotted_name_probe.py

Uses ``live_simple_2-2-0`` (tool ``uber.ride``) and P2-4 locked settings.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from eval.part2.corpus import load_category_entries  # noqa: E402

OPENAI_MODEL = "gpt-5.4-2026-03-05"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENROUTER_MODEL = "qwen/qwen3.5-35b-a3b"
OPENROUTER_ROUTING = {
    "provider": {"order": ["Parasail"], "allow_fallbacks": False},
}
ENTRY_ID = "live_simple_2-2-0"
TOOL_NAME = "uber.ride"


def _http_json(
    *, method: str, url: str, headers: dict[str, str], body: dict[str, Any]
) -> tuple[int, dict[str, Any] | str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    for key, value in headers.items():
        req.add_header(key, value)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def _user_message(entry) -> str:
    turn = entry.question[0]
    if isinstance(turn, list):
        for msg in turn:
            if msg.get("role") == "user":
                return str(msg["content"])
    raise ValueError(f"unexpected question shape for {entry.id}")


def _openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool["parameters"],
        "strict": True,
    }


def _anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool["parameters"],
    }


def _openrouter_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["parameters"],
        },
    }


def _extract_openai_names(response: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in response.get("output", []):
        if item.get("type") == "function_call" and item.get("name"):
            names.append(str(item["name"]))
    return names


def _extract_anthropic_names(response: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for block in response.get("content", []):
        if block.get("type") == "tool_use" and block.get("name"):
            names.append(str(block["name"]))
    return names


def _extract_openrouter_names(response: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for choice in response.get("choices", []):
        message = choice.get("message") or {}
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name"):
                names.append(str(fn["name"]))
    return names


def probe_openai(tool: dict[str, Any], user_message: str, api_key: str) -> dict[str, Any]:
    body = {
        "model": OPENAI_MODEL,
        "input": [{"role": "user", "content": user_message}],
        "tools": [_openai_tool(tool)],
        "tool_choice": "auto",
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "medium"},
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 512,
        "store": True,
    }
    status, response = _http_json(
        method="POST",
        url="https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}"},
        body=body,
    )
    return {
        "provider": "openai",
        "model": OPENAI_MODEL,
        "http_status": status,
        "request_tool_name": tool["name"],
        "response_tool_names": _extract_openai_names(response)
        if isinstance(response, dict)
        else [],
        "raw_response": response,
    }


def probe_anthropic(tool: dict[str, Any], user_message: str, api_key: str) -> dict[str, Any]:
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 512,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "medium"},
        "tool_choice": {"type": "auto"},
        "tools": [_anthropic_tool(tool)],
        "metadata": {"user_id": "cats-part2-dotted-probe"},
        "messages": [{"role": "user", "content": user_message}],
    }
    status, response = _http_json(
        method="POST",
        url="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        body=body,
    )
    return {
        "provider": "anthropic",
        "model": ANTHROPIC_MODEL,
        "http_status": status,
        "request_tool_name": tool["name"],
        "response_tool_names": _extract_anthropic_names(response)
        if isinstance(response, dict)
        else [],
        "raw_response": response,
    }


def probe_openrouter(tool: dict[str, Any], user_message: str, api_key: str) -> dict[str, Any]:
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "tools": [_openrouter_tool(tool)],
        "tool_choice": "auto",
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 512,
        "reasoning": {"effort": "none"},
        **OPENROUTER_ROUTING,
    }
    status, response = _http_json(
        method="POST",
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/Armaan-Mahajan/CATS-format",
            "X-OpenRouter-Title": "CATS Part 2 Dotted Name Probe",
        },
        body=body,
    )
    return {
        "provider": "openrouter",
        "model": OPENROUTER_MODEL,
        "http_status": status,
        "request_tool_name": tool["name"],
        "response_tool_names": _extract_openrouter_names(response)
        if isinstance(response, dict)
        else [],
        "raw_response": response,
    }


def main() -> None:
    entry = next(
        e for e in load_category_entries("live_simple") if e.id == ENTRY_ID
    )
    tool = entry.function[0]
    user_message = _user_message(entry)
    print(f"Entry: {ENTRY_ID}")
    print(f"Request tool name: {tool['name']!r}")
    print(f"User message: {user_message}\n")

    results: list[dict[str, Any]] = []
    if key := os.environ.get("OPENAI_API_KEY_PART2"):
        print("Probing OpenAI…")
        results.append(probe_openai(tool, user_message, key))
    else:
        print("SKIP OpenAI — OPENAI_API_KEY_PART2 not set")

    if key := os.environ.get("ANTHROPIC_API_KEY_PART2"):
        print("Probing Anthropic…")
        results.append(probe_anthropic(tool, user_message, key))
    else:
        print("SKIP Anthropic — ANTHROPIC_API_KEY_PART2 not set")

    if key := os.environ.get("OPENROUTER_API_KEY_PART2"):
        print("Probing OpenRouter/Qwen…")
        results.append(probe_openrouter(tool, user_message, key))
    else:
        print("SKIP OpenRouter — OPENROUTER_API_KEY_PART2 not set")

    print("\n=== Results ===")
    for row in results:
        print(json.dumps(row, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
