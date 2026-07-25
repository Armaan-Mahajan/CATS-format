"""§8.2 validation-inert keywords are dropped on JSON Schema → CATS input."""

from __future__ import annotations

import pytest

from from_json import _IGNORABLE, from_json_with_report
from nodes import ToolBlock
from to_cats import to_cats
from to_json import to_json


def _bfcl_tool(
    properties: dict,
    *,
    required: list[str] | None = None,
    **params_extra: object,
) -> dict:
    params: dict = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required is not None:
        params["required"] = required
    params.update(params_extra)
    return {"name": "demo", "description": "tool", "parameters": params}


def _encoded_tool(tool: dict) -> ToolBlock:
    result = from_json_with_report(tool)
    assert result.fallbacks == [], result.fallbacks
    assert isinstance(result.ast, ToolBlock)
    return result.ast


def _field_required(tool_block: ToolBlock, name: str) -> bool:
    for field in tool_block.fields:
        if field.name == name:
            return field.required
    raise AssertionError(f"field {name!r} not found")


class TestDroppedKeywordsMechanism:
    def test_optional_shares_ignorable_with_title_readonly(self) -> None:
        assert "optional" in _IGNORABLE
        assert {"title", "readOnly", "writeOnly", "$comment"} <= _IGNORABLE

    def test_readonly_still_dropped_via_same_mechanism(self) -> None:
        tool = _bfcl_tool(
            {"x": {"type": "string", "readOnly": True}},
            required=["x"],
        )
        block = _encoded_tool(tool)
        recovered = to_json(block)
        assert "readOnly" not in recovered["properties"]["x"]


class TestOptionalKeyword:
    @pytest.mark.parametrize(
        "prop_schema",
        [
            {"type": "integer", "optional": True},
            {"type": "string", "optional": True},
            {"type": "boolean", "optional": True},
        ],
    )
    def test_property_level_optional_not_in_required_encodes(
        self, prop_schema: dict
    ) -> None:
        tool = _bfcl_tool({"count": prop_schema}, required=[])
        block = _encoded_tool(tool)
        assert _field_required(block, "count") is False
        cats = to_cats(block)
        assert "count*" not in cats
        params = to_json(block)["properties"]["count"]
        assert "optional" not in params

    def test_parameters_level_optional_array_dropped(self) -> None:
        tool = _bfcl_tool(
            {
                "a": {"type": "string"},
                "b": {"type": "integer", "optional": True},
            },
            required=["a"],
            optional=["b"],
        )
        block = _encoded_tool(tool)
        assert _field_required(block, "a") is True
        assert _field_required(block, "b") is False
        recovered = to_json(block)
        assert "optional" not in recovered
        assert recovered.get("required") == ["a"]

    def test_optional_true_but_in_required_encodes_as_required(self) -> None:
        tool = _bfcl_tool(
            {"id": {"type": "integer", "optional": True}},
            required=["id"],
        )
        block = _encoded_tool(tool)
        assert _field_required(block, "id") is True
        cats = to_cats(block)
        assert "id*" in cats
        assert "optional" not in to_json(block)["properties"]["id"]

    @pytest.mark.parametrize("optional_value", ["true", "yes", ["maybe"]])
    def test_optional_value_never_inspected(self, optional_value: object) -> None:
        tool = _bfcl_tool(
            {"flag": {"type": "boolean", "optional": optional_value}},
            required=[],
        )
        block = _encoded_tool(tool)
        assert _field_required(block, "flag") is False
        assert "optional" not in to_json(block)["properties"]["flag"]
