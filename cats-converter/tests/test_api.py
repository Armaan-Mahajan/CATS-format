"""Tests for the public ``cats`` module (cats.py)."""

from __future__ import annotations

import copy

import pytest

import oracle
from cats import (
    ConversionResult,
    FallbackRecord,
    ValidationError,
    ValidationWarning,
    convert,
    convert_for_tool_calling,
    convert_with_report,
    convert_with_report_for_tool_calling,
    to_json_schema,
    validate,
)

CLEAN_TOOL = {
    "name": "echo",
    "type": "object",
    "properties": {"message": {"type": "string", "minLength": 1}},
    "required": ["message"],
    "additionalProperties": False,
}

TOOL_WITH_NOT = {
    "name": "bad",
    "type": "object",
    "properties": {"q": {"not": {"type": "string"}}},
    "additionalProperties": False,
}

EXPECTED_PUBLIC_NAMES = {
    "convert",
    "convert_for_tool_calling",
    "convert_with_report",
    "convert_with_report_for_tool_calling",
    "to_json_schema",
    "validate",
    "validate_with_warnings",
    "AssumedClosedRecord",
    "ConversionResult",
    "FallbackRecord",
    "PythonTypeRenameReport",
    "ValidationError",
    "ValidationWarning",
}


class TestPublicImports:
    def test_public_names_import(self) -> None:
        import cats

        assert cats.convert is convert
        assert cats.validate is validate

    def test_all_matches_public_surface(self) -> None:
        import cats

        assert set(cats.__all__) == EXPECTED_PUBLIC_NAMES


class TestConvert:
    def test_convert_clean_schema_emits_field_lines(self) -> None:
        text = convert(copy.deepcopy(CLEAN_TOOL))
        assert "echo" in text
        assert "message" in text
        assert "string" in text

    def test_convert_with_report_records_fallback(self) -> None:
        result = convert_with_report(copy.deepcopy(TOOL_WITH_NOT))
        assert isinstance(result, ConversionResult)
        assert result.fallback_count > 0
        assert len(result.fallbacks) > 0
        assert all(isinstance(fb, FallbackRecord) for fb in result.fallbacks)

    def test_tool_calling_helpers_match_explicit_flags(self) -> None:
        tool = {
            "name": "coords",
            "type": "dict",
            "properties": {"lat": {"type": "float"}},
            "required": ["lat"],
        }
        explicit = convert_with_report(
            copy.deepcopy(tool),
            assume_closed=True,
            map_python_types=True,
        )
        helper = convert_with_report_for_tool_calling(copy.deepcopy(tool))
        assert helper.cats_text == explicit.cats_text
        assert helper.fallback_count == explicit.fallback_count
        assert convert_for_tool_calling(copy.deepcopy(tool)) == explicit.cats_text

    def test_convert_rejects_bare_type_fragment(self) -> None:
        with pytest.raises(ValueError, match="tool definitions"):
            convert({"type": "integer", "minimum": 1})

    def test_two_tool_document_no_spurious_unused_defs_warnings(self) -> None:
        envelope = [
            {
                "name": "use_foo",
                "type": "object",
                "$defs": {
                    "Foo": {
                        "type": "object",
                        "properties": {"s": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    "Bar": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "additionalProperties": False,
                    },
                },
                "properties": {"x": {"$ref": "#/$defs/Foo"}},
                "additionalProperties": False,
            },
            {
                "name": "use_bar",
                "type": "object",
                "$defs": {
                    "Foo": {
                        "type": "object",
                        "properties": {"s": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    "Bar": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "additionalProperties": False,
                    },
                },
                "properties": {"y": {"$ref": "#/$defs/Bar"}},
                "additionalProperties": False,
            },
        ]
        result = convert_with_report(copy.deepcopy(envelope))
        unused = [
            w for w in result.warnings
            if "never referenced" in str(w).lower()
        ]
        assert unused == []

    def test_inverted_bounds_fall_back_with_validation_warning(self) -> None:
        tool = {
            "name": "t",
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 10, "maximum": 1}},
            "required": ["n"],
            "additionalProperties": False,
        }
        result = convert_with_report(copy.deepcopy(tool))
        assert result.cats_text.startswith("{")
        assert any(fb.location == "t" for fb in result.fallbacks)
        assert any(
            "tool 't'" in str(w) and "§6.2" in w.section
            for w in result.warnings
        )
        recovered = to_json_schema(result.cats_text)
        assert isinstance(recovered, list) and len(recovered) == 1
        assert recovered[0] == tool

    def test_to_json_schema_round_trips_clean_tool(self) -> None:
        schema = copy.deepcopy(CLEAN_TOOL)
        recovered = to_json_schema(convert(schema))
        assert isinstance(recovered, list) and len(recovered) == 1
        assert oracle.behaviorally_equivalent(
            schema,
            recovered[0],
            [{"message": "hi"}, {"message": ""}, {}, {"message": "x", "extra": 1}],
        )

    def test_to_json_schema_round_trips_fallback_tool_through_text(self) -> None:
        schema = copy.deepcopy(TOOL_WITH_NOT)
        text = convert(schema)
        assert text.startswith("{")
        recovered = to_json_schema(text)
        assert isinstance(recovered, list) and len(recovered) == 1
        assert oracle.behaviorally_equivalent(
            schema,
            recovered[0],
            [{"q": 1}, {"q": "s"}, {}],
        )
