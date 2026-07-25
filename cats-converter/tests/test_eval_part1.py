"""Unit tests for Part 1 eval pipeline (no network)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cats-converter"))
sys.path.insert(0, str(REPO_ROOT))

import cats  # noqa: E402
from eval.part1.anthropic_cache import AnthropicTokenCache  # noqa: E402
from eval.part1.constants import (  # noqa: E402
    BUCKET_CONVERTED,
    BUCKET_FELL_BACK,
    BUCKET_INVALID_INPUT,
)
from eval.part1.pipeline import (  # noqa: E402
    RawToolOccurrence,
    attach_token_counts,
    canonical_tool_json,
    convert_unique_tools,
    tool_hash,
    validate_and_aggregate,
)
from from_json import _apply_input_normalizations, normalize_map_python_types  # noqa: E402


def _closed_tool(name: str = "t", **parameters_extra) -> dict:
    params = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    params.update(parameters_extra)
    return {"name": name, "description": "demo", "parameters": params}


class TestPipelineBuckets:
    def test_converted_with_both_flags(self) -> None:
        tool = _closed_tool()
        aggs, invalid, _stats = validate_and_aggregate(
            [RawToolOccurrence(tool=tool, category="simple_python")],
            normalize_map_python_types=normalize_map_python_types,
        )
        results = convert_unique_tools(
            aggs,
            convert_with_report=cats.convert_with_report,
            invalid_hashes=set(),
            invalid_reasons={},
        )
        assert len(results) == 1
        assert results[0].bucket == BUCKET_CONVERTED
        assert results[0].cats_text is not None
        assert "t" in results[0].cats_text

    def test_invalid_input_unknown_type(self) -> None:
        tool = {
            "name": "bad",
            "description": "x",
            "parameters": {"type": "not_a_real_type"},
        }
        aggs, invalid, stats = validate_and_aggregate(
            [RawToolOccurrence(tool=tool, category="simple_python")],
            normalize_map_python_types=normalize_map_python_types,
        )
        assert stats.invalid_after_rename == 1
        results = convert_unique_tools(
            aggs,
            convert_with_report=cats.convert_with_report,
            invalid_hashes={invalid[0][0].tool_hash},
            invalid_reasons={invalid[0][0].tool_hash: invalid[0][1]},
        )
        assert results[0].bucket == BUCKET_INVALID_INPUT

    def test_fell_back_oneof(self) -> None:
        tool = {
            "name": "u",
            "description": "union",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"oneOf": [{"type": "string"}, {"type": "integer"}]}
                },
                "additionalProperties": False,
            },
        }
        aggs, invalid, _ = validate_and_aggregate(
            [RawToolOccurrence(tool=tool, category="simple_python")],
            normalize_map_python_types=normalize_map_python_types,
        )
        results = convert_unique_tools(
            aggs,
            convert_with_report=cats.convert_with_report,
            invalid_hashes=set(),
            invalid_reasons={},
        )
        assert results[0].bucket == BUCKET_FELL_BACK
        assert results[0].fallback_reason


class TestDedupe:
    def test_identical_tools_collapse_occurrence_count(self) -> None:
        tool = _closed_tool()
        occs = [
            RawToolOccurrence(tool=copy.deepcopy(tool), category="simple_python"),
            RawToolOccurrence(tool=copy.deepcopy(tool), category="multiple"),
        ]
        aggs, _invalid, _ = validate_and_aggregate(
            occs, normalize_map_python_types=normalize_map_python_types
        )
        assert len(aggs) == 1
        assert aggs[0].raw_occurrence_count == 2
        assert aggs[0].source_categories == {"simple_python", "multiple"}
        assert aggs[0].tool_hash == tool_hash(
            normalize_map_python_types(tool)[0]
        )

    def test_map_python_types_changes_hash(self) -> None:
        raw = {
            "name": "t",
            "description": "d",
            "parameters": {"type": "dict", "properties": {"x": {"type": "string"}}},
        }
        normalized, report = normalize_map_python_types(raw)
        assert report.dict_count == 1
        assert tool_hash(raw) != tool_hash(normalized)
        assert normalized["parameters"]["type"] == "object"


class TestRenameStats:
    def test_dict_alias_counted_in_validation_pass(self) -> None:
        tool = {
            "name": "t",
            "description": "d",
            "parameters": {"type": "dict", "properties": {}},
        }
        _aggs, _invalid, stats = validate_and_aggregate(
            [RawToolOccurrence(tool=tool, category="simple_python")],
            normalize_map_python_types=normalize_map_python_types,
        )
        assert stats.tools_touched_by_rename == 1
        assert stats.dict_renames == 1

class TestTokenAttachment:
    def test_token_fields_only_on_converted(self) -> None:
        tool = _closed_tool()
        aggs, _, _ = validate_and_aggregate(
            [RawToolOccurrence(tool=tool, category="simple_python")],
            normalize_map_python_types=normalize_map_python_types,
        )
        results = convert_unique_tools(
            aggs,
            convert_with_report=cats.convert_with_report,
            invalid_hashes=set(),
            invalid_reasons={},
        )
        attach_token_counts(
            results,
            tiktoken_count=lambda s: len(s),
            qwen_count=lambda s: len(s) + 1,
        )
        row = results[0]
        assert row.tokens_json_tiktoken is not None
        assert row.tokens_cats_tiktoken is not None
        assert row.delta_tiktoken == row.tokens_cats_tiktoken - row.tokens_json_tiktoken

    def test_non_converted_rows_have_null_tokens(self) -> None:
        from eval.part1.pipeline import UniqueToolResult

        row = UniqueToolResult(
            tool_hash="x",
            source_categories=["simple_python"],
            raw_occurrence_count=1,
            bucket=BUCKET_FELL_BACK,
            fallback_reason="oneOf",
        )
        attach_token_counts(
            [row],
            tiktoken_count=lambda s: len(s),
            qwen_count=lambda s: len(s),
        )
        assert row.tokens_json_tiktoken is None


class TestAnthropicCache:
    def test_cache_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        cache = AnthropicTokenCache(path)
        cache.set("abc", 10, 7)
        cache.save()
        reloaded = AnthropicTokenCache(path)
        assert reloaded.get("abc") == {
            "tokens_json_anthropic": 10,
            "tokens_cats_anthropic": 7,
        }


class TestBfclEnvelopeShape:
    def test_map_and_assume_closed_reach_parameters_subtree(self) -> None:
        tool = {
            "name": "t",
            "description": "d",
            "parameters": {
                "type": "dict",
                "properties": {"x": {"type": "float"}},
            },
        }
        normalized, report = normalize_map_python_types(tool)
        assert report.dict_count == 1
        assert report.float_count == 1
        _, records, _ = _apply_input_normalizations(
            tool, assume_closed=True, map_python_types=True
        )
        locations = {r.location for r in records}
        assert "/parameters" in locations


class TestIsolation:
    def test_input_dict_not_mutated_by_flags(self) -> None:
        tool = {
            "name": "t",
            "description": "d",
            "parameters": {
                "type": "dict",
                "properties": {"x": {"type": "string"}},
            },
        }
        snapshot = copy.deepcopy(tool)
        baseline = cats.convert(copy.deepcopy(tool))
        cats.convert_with_report(
            copy.deepcopy(tool),
            assume_closed=True,
            map_python_types=True,
        )
        assert tool == snapshot
        assert cats.convert(copy.deepcopy(tool)) == baseline
