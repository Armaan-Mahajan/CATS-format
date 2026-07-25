"""§7.7 normalization characterization across nesting positions (Parts A–B)."""

from __future__ import annotations

import copy

import cats
from from_json import _apply_input_normalizations, from_json_with_report
from nodes import AnyType, Document, Object, RawSchema, ToolBlock
from to_cats import to_cats


def _assume_closed_locations(schema: dict) -> set[str]:
    _, records, _ = _apply_input_normalizations(
        schema, assume_closed=True, map_python_types=False
    )
    return {record.location for record in records}


def _map_report(schema: dict):
    _, _, report = _apply_input_normalizations(
        schema, assume_closed=False, map_python_types=True
    )
    return report


def _closed_tool(**overrides: object) -> dict:
    tool = {
        "name": "t",
        "type": "object",
        "properties": {},
    }
    tool.update(overrides)
    return tool


# ---------------------------------------------------------------------------
# Part A: assume_closed across nesting positions
# ---------------------------------------------------------------------------


class TestAssumeClosedNesting:
    def test_array_element_object_regression(self) -> None:
        tool = _closed_tool(
            properties={
                "arr": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                }
            }
        )
        locs = _assume_closed_locations(tool)
        assert "/properties/arr/items" in locs
        result = from_json_with_report(tool, assume_closed=True)
        assert isinstance(result.ast, ToolBlock)

    def test_property_of_property_two_levels(self) -> None:
        tool = _closed_tool(
            properties={
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {
                            "type": "object",
                            "properties": {"n": {"type": "integer"}},
                        }
                    },
                }
            }
        )
        locs = _assume_closed_locations(tool)
        assert "/properties/outer" in locs
        assert "/properties/outer/properties/inner" in locs
        result = from_json_with_report(tool, assume_closed=True)
        assert isinstance(result.ast, ToolBlock)

    def test_anyof_branch_object(self) -> None:
        tool = _closed_tool(
            properties={
                "u": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"a": {"type": "string"}},
                        },
                        {"type": "string"},
                    ]
                }
            }
        )
        locs = _assume_closed_locations(tool)
        assert "/properties/u/anyOf/0" in locs
        result = from_json_with_report(tool, assume_closed=True)
        # Encoder: shaped object inside a Union (anyOf) branch is a non-encodable
        # position (§7.5) — normalization still closes the branch object.
        assert isinstance(result.ast, RawSchema)

    def test_defs_entry_referenced(self) -> None:
        tool = _closed_tool(
            properties={"home": {"$ref": "#/$defs/Address"}},
            **{
                "$defs": {
                    "Address": {
                        "type": "object",
                        "properties": {"street": {"type": "string"}},
                    }
                }
            },
        )
        locs = _assume_closed_locations(tool)
        assert "/$defs/Address" in locs
        result = from_json_with_report(tool, assume_closed=True)
        assert isinstance(result.ast, Document)
        assert isinstance(result.ast.tools[0], ToolBlock)

    def test_items_list_form_tuple_style(self) -> None:
        # Encoder rejects tuple-style `items` lists (§8.1); walker still normalizes.
        schema = {
            "name": "t",
            "type": "object",
            "properties": {
                "row": {
                    "type": "array",
                    "items": [
                        {"type": "object", "properties": {"a": {"type": "string"}}},
                        {"type": "integer"},
                    ],
                }
            },
        }
        locs = _assume_closed_locations(schema)
        assert "/properties/row/items/0" in locs
        result = from_json_with_report(schema, assume_closed=True)
        assert isinstance(result.ast, RawSchema)

    def test_nested_arrays_object_element(self) -> None:
        tool = _closed_tool(
            properties={
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"v": {"type": "number"}},
                        },
                    },
                }
            }
        )
        locs = _assume_closed_locations(tool)
        assert "/properties/matrix/items/items" in locs
        result = from_json_with_report(tool, assume_closed=True)
        # Encoder: shaped object at array<array<…>> element is non-encodable (§7.5).
        assert isinstance(result.ast, RawSchema)

    def test_pattern_properties_nested_object(self) -> None:
        # patternProperties triggers whole-schema fallback (§8.1); walker still visits.
        schema = {
            "name": "t",
            "type": "object",
            "patternProperties": {
                "^x": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                }
            },
        }
        locs = _assume_closed_locations(schema)
        assert "/patternProperties/^x" in locs
        result = from_json_with_report(schema, assume_closed=True)
        assert isinstance(result.ast, RawSchema)

    def test_additional_properties_schema_nested_object(self) -> None:
        schema = {
            "name": "t",
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {"k": {"type": "string"}},
            },
        }
        locs = _assume_closed_locations(schema)
        assert "/additionalProperties" in locs
        result = from_json_with_report(schema, assume_closed=True)
        assert isinstance(result.ast, RawSchema)

    def test_mixed_depth_record_count(self) -> None:
        tool = _closed_tool(
            properties={
                "mid": {
                    "type": "object",
                    "properties": {
                        "deep": {
                            "type": "object",
                            "properties": {"x": {"type": "string"}},
                        }
                    },
                }
            }
        )
        result = from_json_with_report(tool, assume_closed=True)
        assert len(result.assumed_closed) == 3
        locs = {record.location for record in result.assumed_closed}
        assert locs == {
            "/",
            "/properties/mid",
            "/properties/mid/properties/deep",
        }


# ---------------------------------------------------------------------------
# Part A: map_python_types across nesting positions
# ---------------------------------------------------------------------------


class TestMapPythonTypesNesting:
    def test_array_element_dict_renamed(self) -> None:
        tool = _closed_tool(
            additionalProperties=False,
            properties={
                "arr": {
                    "type": "array",
                    "items": {"type": "dict", "properties": {"x": {"type": "string"}}},
                }
            },
        )
        report = _map_report(tool)
        assert report.dict_count >= 1
        assert "/properties/arr/items" in report.touched_locations

    def test_property_of_property_dict_renamed(self) -> None:
        tool = _closed_tool(
            additionalProperties=False,
            properties={
                "outer": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "inner": {"type": "dict", "properties": {"n": {"type": "float"}}},
                    },
                }
            },
        )
        report = _map_report(tool)
        assert report.dict_count == 1
        assert report.float_count == 1
        assert "/properties/outer/properties/inner" in report.touched_locations
        assert "/properties/outer/properties/inner/properties/n" in report.touched_locations

    def test_anyof_branch_dict_renamed(self) -> None:
        tool = _closed_tool(
            additionalProperties=False,
            properties={
                "u": {
                    "anyOf": [
                        {"type": "dict", "properties": {"a": {"type": "string"}}},
                        {"type": "string"},
                    ]
                }
            },
        )
        report = _map_report(tool)
        assert report.dict_count == 1
        assert "/properties/u/anyOf/0" in report.touched_locations

    def test_defs_entry_dict_renamed(self) -> None:
        tool = _closed_tool(
            additionalProperties=False,
            properties={"home": {"$ref": "#/$defs/Address"}},
            **{
                "$defs": {
                    "Address": {
                        "type": "dict",
                        "properties": {"street": {"type": "string"}},
                    }
                }
            },
        )
        report = _map_report(tool)
        assert report.dict_count == 1
        assert "/$defs/Address" in report.touched_locations

    def test_items_list_form_dict_renamed(self) -> None:
        schema = {
            "name": "t",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "row": {
                    "type": "array",
                    "items": [
                        {"type": "dict", "properties": {"a": {"type": "string"}}},
                        {"type": "integer"},
                    ],
                }
            },
        }
        report = _map_report(schema)
        assert report.dict_count == 1
        assert "/properties/row/items/0" in report.touched_locations

    def test_nested_arrays_dict_renamed(self) -> None:
        tool = _closed_tool(
            additionalProperties=False,
            properties={
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "dict", "properties": {"v": {"type": "float"}}},
                    },
                }
            },
        )
        report = _map_report(tool)
        assert report.dict_count == 1
        assert report.float_count == 1
        assert "/properties/matrix/items/items" in report.touched_locations

    def test_pattern_properties_dict_renamed(self) -> None:
        schema = {
            "name": "t",
            "type": "object",
            "additionalProperties": False,
            "patternProperties": {
                "^x": {"type": "dict", "properties": {"n": {"type": "integer"}}},
            },
        }
        report = _map_report(schema)
        assert report.dict_count == 1
        assert "/patternProperties/^x" in report.touched_locations

    def test_additional_properties_schema_dict_renamed(self) -> None:
        schema = {
            "name": "t",
            "type": "object",
            "additionalProperties": {
                "type": "dict",
                "properties": {"k": {"type": "string"}},
            },
        }
        report = _map_report(schema)
        assert report.dict_count == 1
        assert "/additionalProperties" in report.touched_locations


# ---------------------------------------------------------------------------
# Part B: interaction, ordering, idempotence, isolation
# ---------------------------------------------------------------------------


class TestInteractionAndGuards:
    NESTED_DICT = {
        "name": "t",
        "type": "dict",
        "properties": {
            "outer": {
                "type": "dict",
                "properties": {"x": {"type": "string"}},
            }
        },
    }

    def test_both_flags_map_then_close_two_levels(self) -> None:
        map_only = from_json_with_report(self.NESTED_DICT, map_python_types=True)
        assume_only = from_json_with_report(self.NESTED_DICT, assume_closed=True)
        both = from_json_with_report(
            self.NESTED_DICT,
            map_python_types=True,
            assume_closed=True,
        )
        assert isinstance(map_only.ast, RawSchema)
        assert isinstance(assume_only.ast, RawSchema)
        assert isinstance(both.ast, ToolBlock)
        assert both.python_type_renames.dict_count == 2
        assert len(both.assumed_closed) == 2

    def test_nested_any_drops_type_and_encodes_as_cats_any(self) -> None:
        tool = {
            "name": "t",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "wrapper": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "inner": {"type": "any", "description": "free-form"},
                    },
                }
            },
        }
        result = from_json_with_report(tool, map_python_types=True)
        assert result.python_type_renames.any_count == 1
        assert "/properties/wrapper/properties/inner" in (
            result.python_type_renames.touched_locations
        )
        assert isinstance(result.ast, ToolBlock)
        inner_field = result.ast.fields[0].type
        assert isinstance(inner_field, Object)
        inner_inner = inner_field.fields[0].type
        assert isinstance(inner_inner, AnyType)
        assert to_cats(inner_inner) == "any"

    def test_idempotent_convert_with_flags(self) -> None:
        tool = _closed_tool(properties={"x": {"type": "string"}})
        first = cats.convert_with_report(
            copy.deepcopy(tool),
            assume_closed=True,
            map_python_types=True,
        )
        second = cats.convert_with_report(
            copy.deepcopy(tool),
            assume_closed=True,
            map_python_types=True,
        )
        assert first.cats_text == second.cats_text
        assert len(first.assumed_closed) == len(second.assumed_closed)
        assert first.python_type_renames == second.python_type_renames

    def test_isolation_flags_off_unchanged_and_input_not_mutated(self) -> None:
        original = _closed_tool(properties={"x": {"type": "string"}})
        snapshot = copy.deepcopy(original)
        baseline = cats.convert(copy.deepcopy(original))
        flagged = cats.convert_with_report(
            copy.deepcopy(original),
            assume_closed=True,
            map_python_types=True,
        )
        assert original == snapshot
        assert cats.convert(copy.deepcopy(original)) == baseline
        assert flagged.cats_text != baseline
