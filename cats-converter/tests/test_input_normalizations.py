"""Tests for §7.7 opt-in input normalizations (assume_closed, map_python_types)."""

from __future__ import annotations

import cats
from from_json import from_json_with_report
from nodes import AnyType, Number, RawSchema, ToolBlock
from to_cats import to_cats


def _open_tool(**overrides: object) -> dict:
    tool = {
        "name": "t",
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    tool.update(overrides)
    return tool


class TestAssumeClosed:
    def test_omitted_additional_properties_default_falls_back(self) -> None:
        result = from_json_with_report(_open_tool())
        assert isinstance(result.ast, RawSchema)
        assert result.assumed_closed == []

    def test_omitted_with_flag_encodes_closed(self) -> None:
        result = from_json_with_report(_open_tool(), assume_closed=True)
        assert isinstance(result.ast, ToolBlock)
        assert len(result.assumed_closed) == 1
        assert result.assumed_closed[0].location == "/"

    def test_explicit_true_with_flag_still_falls_back(self) -> None:
        result = from_json_with_report(
            _open_tool(additionalProperties=True),
            assume_closed=True,
        )
        assert isinstance(result.ast, RawSchema)
        assert result.assumed_closed == []

    def test_explicit_false_with_flag_encodes(self) -> None:
        result = from_json_with_report(
            _open_tool(additionalProperties=False),
            assume_closed=True,
        )
        assert isinstance(result.ast, ToolBlock)
        assert result.assumed_closed == []

    def test_typed_open_additional_properties_unaffected(self) -> None:
        result = from_json_with_report(
            _open_tool(additionalProperties={"type": "string"}),
            assume_closed=True,
        )
        assert isinstance(result.ast, RawSchema)
        assert result.assumed_closed == []


class TestMapPythonTypes:
    def test_float_alias_renames_to_number(self) -> None:
        result = from_json_with_report({"type": "float"}, map_python_types=True)
        assert isinstance(result.ast, Number)
        assert result.fallback_count == 0
        assert result.python_type_renames.float_count == 1
        assert result.python_type_renames.touched_locations == ["/"]
        assert to_cats(result.ast) == "number"

    def test_dict_alias_renames_then_open_object_falls_back(self) -> None:
        result = from_json_with_report({"type": "dict"}, map_python_types=True)
        assert isinstance(result.ast, RawSchema)
        assert result.ast.schema == {"type": "object"}
        assert result.python_type_renames.dict_count == 1

    def test_tuple_alias_renames_to_array(self) -> None:
        from nodes import Array

        result = from_json_with_report({"type": "tuple"}, map_python_types=True)
        assert isinstance(result.ast, Array)
        assert result.fallback_count == 0
        assert result.python_type_renames.tuple_count == 1
        assert to_cats(result.ast) == "array"

    def test_unknown_type_left_untouched_and_falls_back(self) -> None:
        result = from_json_with_report(
            {"type": "foobar"},
            map_python_types=True,
        )
        assert isinstance(result.ast, RawSchema)
        assert result.python_type_renames.touched_locations == []

    def test_any_drops_type_and_encodes_as_cats_any(self) -> None:
        result = from_json_with_report({"type": "any"}, map_python_types=True)
        assert isinstance(result.ast, AnyType)
        assert to_cats(result.ast) == "any"
        assert result.python_type_renames.any_count == 1
        assert result.python_type_renames.touched_locations == ["/"]

    def test_any_with_constraints_still_falls_back(self) -> None:
        result = from_json_with_report(
            {"type": "any", "minLength": 1},
            map_python_types=True,
        )
        assert isinstance(result.ast, RawSchema)
        assert result.python_type_renames.any_count == 1


class TestCompositionDictOpenObject:
    """``dict`` + omitted ``additionalProperties`` under each flag combination."""

    SCHEMA = {
        "name": "t",
        "type": "dict",
        "properties": {"x": {"type": "string"}},
    }

    def test_both_flags_off_falls_back(self) -> None:
        result = from_json_with_report(self.SCHEMA)
        assert isinstance(result.ast, RawSchema)

    def test_map_only_still_open_falls_back(self) -> None:
        result = from_json_with_report(self.SCHEMA, map_python_types=True)
        assert isinstance(result.ast, RawSchema)
        assert result.python_type_renames.dict_count == 1

    def test_assume_only_invalid_dict_type_falls_back(self) -> None:
        result = from_json_with_report(self.SCHEMA, assume_closed=True)
        assert isinstance(result.ast, RawSchema)
        assert result.assumed_closed == []

    def test_both_flags_encode_closed(self) -> None:
        result = from_json_with_report(
            self.SCHEMA,
            assume_closed=True,
            map_python_types=True,
        )
        assert isinstance(result.ast, ToolBlock)
        assert result.python_type_renames.dict_count == 1
        assert len(result.assumed_closed) == 1


class TestReporting:
    def test_assumed_closed_records_populate(self) -> None:
        tool = _open_tool(
            properties={
                "outer": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                }
            }
        )
        result = from_json_with_report(tool, assume_closed=True)
        locations = {record.location for record in result.assumed_closed}
        assert "/" in locations
        assert "/properties/outer" in locations

    def test_python_type_rename_counts_populate(self) -> None:
        schema = {
            "name": "mix",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "a": {"type": "float"},
                "b": {"type": "dict", "additionalProperties": False, "properties": {}},
                "c": {"type": "tuple"},
                "d": {"type": "any"},
            },
        }
        result = from_json_with_report(schema, map_python_types=True)
        report = result.python_type_renames
        assert report.float_count == 1
        assert report.dict_count == 1
        assert report.tuple_count == 1
        assert report.any_count == 1
        assert len(report.touched_locations) == 4

    def test_public_api_threads_flags_and_reports(self) -> None:
        result = cats.convert_with_report(
            _open_tool(),
            assume_closed=True,
        )
        assert "t\n" in result.cats_text
        assert len(result.assumed_closed) == 1
        assert result.python_type_renames.touched_locations == []
