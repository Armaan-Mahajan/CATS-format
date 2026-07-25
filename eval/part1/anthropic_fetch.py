"""Shared Anthropic count_tokens fetching (rate-limited, logged, cached)."""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Protocol

from eval.part1.anthropic_cache import AnthropicPrettyTokenCache, AnthropicTokenCache
from eval.part1.tokenizers import count_anthropic_tokens

TARGET_RPM = 85


class RequestRateLimiter:
    """Thread-safe minimum spacing between request starts (global across workers)."""

    def __init__(self, target_rpm: float = TARGET_RPM) -> None:
        self._min_interval = 60.0 / target_rpm
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


def format_duration(seconds: float) -> str:
    total = int(seconds)
    if total >= 3600:
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


class _CompactRow(Protocol):
    tool_hash: str
    normalized_tool: object
    cats_text: str


def _count_with_retries(
    *,
    client: object,
    limiter: RequestRateLimiter,
    text: str,
    row_id: str,
) -> int:
    from anthropic import RateLimitError

    def _limited_count(payload: str) -> int:
        limiter.acquire()
        return count_anthropic_tokens(client, payload)

    for attempt in range(6):
        try:
            return _limited_count(text)
        except RateLimitError:
            time.sleep(2**attempt + random.uniform(0, 1))
            continue
        except Exception as exc:
            message = str(exc)
            if "429" in message or "rate" in message.lower():
                time.sleep(2**attempt + random.uniform(0, 1))
                continue
            raise
    raise RuntimeError(f"Anthropic rate limit retries exhausted for {row_id}")


def fetch_anthropic_compact_counts(
    rows: list[_CompactRow],
    cache: AnthropicTokenCache,
    *,
    json_text_for: Callable[[object], str],
    max_workers: int = 2,
) -> None:
    """Fetch compact JSON + CATS Anthropic counts (Part 1 runner)."""
    from anthropic import Anthropic

    client = Anthropic()
    limiter = RequestRateLimiter()
    eligible = [row for row in rows if row.normalized_tool is not None and row.cats_text is not None]
    pending = [row for row in eligible if cache.get(row.tool_hash) is None]
    eligible_count = len(eligible)
    cached_count = eligible_count - len(pending)
    total = len(pending)

    if not pending:
        print("all Anthropic counts cached — nothing to fetch.")
        return

    print(
        f"Anthropic tokenization: {eligible_count} eligible, "
        f"{cached_count} cached, {total} to fetch ({total * 2} requests)."
    )

    def _count_one(row: _CompactRow) -> tuple[str, int, int]:
        json_text = json_text_for(row.normalized_tool)
        json_tokens = _count_with_retries(
            client=client,
            limiter=limiter,
            text=json_text,
            row_id=row.tool_hash,
        )
        cats_tokens = _count_with_retries(
            client=client,
            limiter=limiter,
            text=row.cats_text,
            row_id=row.tool_hash,
        )
        return row.tool_hash, json_tokens, cats_tokens

    _run_fetch_loop(
        pending=pending,
        total=total,
        max_workers=max_workers,
        cache=cache,
        count_one=_count_one,
        on_success=lambda tool_hash, json_tokens, cats_tokens: cache.set(
            tool_hash, json_tokens, cats_tokens
        ),
    )


def fetch_anthropic_pretty_json_counts(
    entries: list[tuple[str, str, int]],
    cache: AnthropicPrettyTokenCache,
    *,
    max_workers: int = 2,
) -> None:
    """Fetch pretty-printed JSON Anthropic counts (token_stats robustness).

    Each entry is ``(tool_hash, pretty_json_text, cats_tokens)``. CATS counts are
    taken from records.jsonl and used only for per-line logging.
    """
    from anthropic import Anthropic

    client = Anthropic()
    limiter = RequestRateLimiter()
    eligible_count = len(entries)
    pending = [(tool_hash, text, cats) for tool_hash, text, cats in entries if cache.get(tool_hash) is None]
    cached_count = eligible_count - len(pending)
    total = len(pending)

    if not pending:
        print("all Anthropic pretty-printed counts cached — nothing to fetch.")
        return

    print(
        f"Anthropic tokenization (pretty-printed JSON): {eligible_count} eligible, "
        f"{cached_count} cached, {total} to fetch ({total} requests)."
    )

    def _count_one(item: tuple[str, str, int]) -> tuple[str, int, int]:
        tool_hash, pretty_text, cats_tokens = item
        json_tokens = _count_with_retries(
            client=client,
            limiter=limiter,
            text=pretty_text,
            row_id=tool_hash,
        )
        return tool_hash, json_tokens, cats_tokens

    _run_fetch_loop(
        pending=pending,
        total=total,
        max_workers=max_workers,
        cache=cache,
        count_one=_count_one,
        on_success=lambda tool_hash, json_tokens, _cats: cache.set(tool_hash, json_tokens),
    )


def _run_fetch_loop(
    *,
    pending: list,
    total: int,
    max_workers: int,
    cache: AnthropicTokenCache | AnthropicPrettyTokenCache,
    count_one: Callable,
    on_success: Callable[[str, int, int], None],
) -> None:
    print("(per-line % is Anthropic-tokenizer only; final averaged savings differ.)")
    loop_start = time.monotonic()
    done = 0
    counted = 0
    errors = 0
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_row = {pool.submit(count_one, row): row for row in pending}
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            row_key = row.tool_hash if hasattr(row, "tool_hash") else row[0]
            try:
                tool_hash, json_tokens, cats_tokens = future.result()
            except Exception as exc:
                done += 1
                errors += 1
                failed.append(row_key)
                print(f"[{done}/{total}] {row_key[:8]} FAILED: {exc}")
                if done % 50 == 0:
                    cache.save()
                continue

            on_success(tool_hash, json_tokens, cats_tokens)
            done += 1
            counted += 1
            elapsed = time.monotonic() - loop_start
            if json_tokens > 0:
                pct = (json_tokens - cats_tokens) / json_tokens * 100
            else:
                pct = 0.0
            if done < total and done > 0:
                eta_seconds = elapsed / done * (total - done)
                eta = format_duration(eta_seconds)
            else:
                eta = "0:00"
            print(
                f"[{done}/{total}] {tool_hash[:8]}  "
                f"json {json_tokens} -> cats {cats_tokens}  ({pct:+.1f}%)  | "
                f"elapsed {format_duration(elapsed)}  ETA {eta}"
            )
            if done % 50 == 0:
                cache.save()

    cache.save()
    elapsed_total = time.monotonic() - loop_start
    print(
        f"Done: {counted} counted, {errors} errors, "
        f"{format_duration(elapsed_total)} elapsed. Cache saved."
    )
    if failed:
        print(f"{len(failed)} tools failed and were not counted; re-run to retry them.")
