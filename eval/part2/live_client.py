"""Live provider HTTP client for Part 2 orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from eval.part2.checker import Part2Condition
from eval.part2.http_client import http_json_with_retry
from eval.part2.pilot_entries import PILOT_METADATA_TAG
from eval.part2.providers import Provider
from eval.part2.request_builder import PreparedRequest
from eval.part2.response_parser import extract_model_output


_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/Armaan-Mahajan/CATS-format",
    "X-OpenRouter-Title": "CATS Part 2 Pilot",
}


@dataclass(frozen=True)
class LiveCallResult:
    http_status: int
    response_body: dict[str, Any] | list[Any] | str | None
    sent_request_body: dict[str, Any] | None
    extracted: Any
    retry_count: int
    backend: str | None
    error: str | None
    completed: bool


def _openrouter_backend(
    response: dict[str, Any], generation_id: str | None, api_key: str
) -> str | None:
    if isinstance(response.get("provider"), str):
        return response["provider"]
    meta = response.get("openrouter_metadata")
    if isinstance(meta, dict):
        attempts = meta.get("attempts") or []
        if attempts and isinstance(attempts[0], dict) and attempts[0].get("provider"):
            return str(attempts[0]["provider"])
    if not generation_id:
        return None
    status, body = http_json_with_retry(
        method="GET",
        url=f"https://openrouter.ai/api/v1/generation?id={generation_id}",
        headers={"Authorization": f"Bearer {api_key}", **_OPENROUTER_HEADERS},
    )
    if status == 200 and isinstance(body, dict):
        data = body.get("data") or body
        if isinstance(data, dict) and data.get("provider_name"):
            return str(data["provider_name"])
    return None


class LiveProviderClient:
    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        metadata_tag: str = PILOT_METADATA_TAG,
    ) -> None:
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY_PART2")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY_PART2")
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY_PART2")
        self.metadata_tag = metadata_tag

    def call(
        self,
        prepared: PreparedRequest,
        *,
        condition: Part2Condition,
    ) -> LiveCallResult:
        if prepared.skipped or prepared.request_body is None:
            return LiveCallResult(
                http_status=0,
                response_body=None,
                sent_request_body=None,
                extracted=None,
                retry_count=0,
                backend=None,
                error=None,
                completed=True,
            )

        body = json.loads(json.dumps(prepared.request_body))
        provider = prepared.provider

        try:
            if provider == Provider.OPENAI:
                status, response = http_json_with_retry(
                    method="POST",
                    url="https://api.openai.com/v1/responses",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    body=body,
                )
            elif provider == Provider.ANTHROPIC:
                body["metadata"] = {"user_id": self.metadata_tag}
                status, response = http_json_with_retry(
                    method="POST",
                    url="https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    body=body,
                )
            else:
                status, response = http_json_with_retry(
                    method="POST",
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        **_OPENROUTER_HEADERS,
                    },
                    body=body,
                )
        except Exception as exc:  # noqa: BLE001 — pilot must log, not crash the run
            return LiveCallResult(
                http_status=0,
                response_body=None,
                sent_request_body=body,
                extracted=None,
                retry_count=0,
                backend=None,
                error=f"{type(exc).__name__}: {exc}",
                completed=False,
            )

        if status != 200 or not isinstance(response, dict):
            return LiveCallResult(
                http_status=status,
                response_body=response if isinstance(response, (dict, list, str)) else None,
                sent_request_body=body,
                extracted=None,
                retry_count=0,
                backend=None,
                error=f"HTTP {status}",
                completed=False,
            )

        extracted = extract_model_output(
            provider=provider,
            condition=condition,
            response=response,
        )
        backend = None
        if provider == Provider.OPENROUTER:
            gen_id = response.get("id")
            backend = _openrouter_backend(
                response,
                str(gen_id) if gen_id else None,
                self.openrouter_api_key or "",
            )

        return LiveCallResult(
            http_status=status,
            response_body=response,
            sent_request_body=body,
            extracted=extracted,
            retry_count=0,
            backend=backend,
            error=None,
            completed=True,
        )
