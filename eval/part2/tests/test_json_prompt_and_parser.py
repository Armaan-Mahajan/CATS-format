"""Tests for condition (b) prompt builder and ground-truth-blind output parser."""

from __future__ import annotations

import copy
import json

import pytest
from eval.part1.pipeline import canonical_tool_json
from from_json import normalize_map_python_types
from primer import build_system_prompt, generate_primer_from_json

from eval.part2.corpus import load_category_entries, load_part2_corpus
from eval.part2.json_prompt import (
    _JSON_INTRO,
    build_json_system_prompt,
    canonical_tools_array_json,
    output_contract_section,
    required_uniformity_from_tools,
)
from eval.part2.output_parser import ParseOutcome, parse_tool_call_response


def _normalized_tools(entry) -> list[dict]:
    return [
        normalize_map_python_types(copy.deepcopy(tool))[0] for tool in entry.function
    ]


def _cats_system_prompt(tools: list[dict]) -> str:
    result = generate_primer_from_json(copy.deepcopy(tools))
    if result.all_fallback:
        pytest.skip("entry is all-fallback; condition (a) not applicable")
    return build_system_prompt(result)


class TestOutputParser:
    def test_no_fenced_block(self) -> None:
        result = parse_tool_call_response("I cannot help with that.")
        assert result.outcome == ParseOutcome.NO_FENCED_BLOCK
        assert not result.syntactically_valid

    def test_valid_single_block(self) -> None:
        text = (
            'Here is the call:\n\n```json\n'
            '{"name": "get_user_info", "arguments": {"user_id": 42}}\n```'
        )
        result = parse_tool_call_response(text)
        assert result.syntactically_valid
        assert result.name == "get_user_info"
        assert result.arguments == {"user_id": 42}

    def test_invalid_json(self) -> None:
        text = "```json\n{not valid json}\n```"
        result = parse_tool_call_response(text)
        assert result.outcome == ParseOutcome.INVALID_JSON
        assert not result.syntactically_valid

    def test_invalid_shape_missing_arguments(self) -> None:
        text = '```json\n{"name": "foo"}\n```'
        result = parse_tool_call_response(text)
        assert result.outcome == ParseOutcome.INVALID_SHAPE

    def test_invalid_shape_wrong_name_type(self) -> None:
        text = '```json\n{"name": 1, "arguments": {}}\n```'
        result = parse_tool_call_response(text)
        assert result.outcome == ParseOutcome.INVALID_SHAPE

    def test_multiple_blocks_uses_last(self) -> None:
        text = (
            "Example only:\n"
            '```json\n{"name": "wrong_tool", "arguments": {}}\n```\n\n'
            "Actual call:\n"
            '```json\n{"name": "right_tool", "arguments": {"x": 1}}\n```'
        )
        result = parse_tool_call_response(text)
        assert result.syntactically_valid
        assert result.name == "right_tool"
        assert result.arguments == {"x": 1}


def _entry_by_id(entry_id: str):
    for entry in load_part2_corpus():
        if entry.id == entry_id:
            return entry
    raise KeyError(entry_id)


@pytest.mark.parametrize(
    "entry_id",
    [
        "live_simple_0-0-0",
        "live_simple_50-22-0",
        "multiple_0",
        "multiple_42",
        "live_multiple_100-42-4",
    ],
)
def test_output_contract_matches_between_conditions(entry_id: str) -> None:
    entry = _entry_by_id(entry_id)
    tools = _normalized_tools(entry)
    cats_prompt = _cats_system_prompt(tools)
    json_prompt = build_json_system_prompt(tools)

    assert output_contract_section(cats_prompt) == output_contract_section(json_prompt)


def test_json_tool_block_round_trips() -> None:
    entry = load_category_entries("multiple")[0]
    tools = _normalized_tools(entry)
    payload = json.loads(canonical_tools_array_json(tools))
    assert payload == [json.loads(canonical_tool_json(t)) for t in tools]


def test_json_intro_is_fixed() -> None:
    tools = _normalized_tools(load_category_entries("live_simple")[0])
    prompt = build_json_system_prompt(tools)
    assert prompt.startswith(_JSON_INTRO)
