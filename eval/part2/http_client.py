"""HTTP helpers with 429 retry for real provider calls."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF_S = 2.0


def http_json_with_retry(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff_s: float = DEFAULT_INITIAL_BACKOFF_S,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    """POST/GET JSON with exponential backoff on HTTP 429."""
    payload = None if body is None else json.dumps(body).encode("utf-8")
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=payload, method=method)
        for key, value in headers.items():
            req.add_header(key, value)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read().decode("utf-8", errors="replace")
            if status == 429 and attempt < max_retries:
                delay = initial_backoff_s * (2**attempt)
                time.sleep(delay)
                attempt += 1
                continue
            try:
                parsed: dict[str, Any] | list[Any] | str = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
            return status, parsed
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < max_retries:
                delay = initial_backoff_s * (2**attempt)
                time.sleep(delay)
                attempt += 1
                continue
            return 0, str(exc)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return status, parsed
