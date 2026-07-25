"""Round-trip tests for from_json.py, developed AGAINST the behavioral oracle.

Every in-scope construct gets a test that runs the schema through the full
round trip (JSON Schema -> from_json -> CATS AST -> to_json -> JSON Schema) and
asserts, via tests/oracle.py, that the recovered schema accepts/rejects the
same values as the original. The oracle is the arbiter of the fallback boundary:
a construct is in-scope only if it round-trips behaviorally clean.

Confidence: `assert_round_trip_preserves_meaning` returns a RoundTripReport.
Where Hypothesis generates few valid instances (const, enum, tight bounds), the
test hand-feeds discriminating instances via `extra_instances` — including ones
that MUST be rejected — so a green result reflects real coverage, not a pass on
one sampled value. `_assert_meaning` asserts the pass is not silently thin once
those extras are supplied.

Fallback: a tool containing an unencodable construct (e.g. `not`) must still
round-trip raw-in -> RawSchema -> raw-out, and must be RECORDED in the report.
"""

from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

import oracle
from from_json import (
    ConversionResult,
    FallbackRecord,
    from_json,
    from_json_with_report,
    load_schema,
)
from nodes import (
    Array,
    Boolean,
    Const,
    Document,
    Enum,
    Integer,
    Object,
    RawSchema,
    Reference,
    String,
    ToolBlock,
    Union,
)
from to_json import to_json


def _assert_meaning(schema, *, extra=(), discriminating=False):
    """Round-trip `schema` and assert meaning is preserved; return the report.

    Hypothesis generation is legitimately THIN for narrow schemas (a boolean, a
    const, a small enum, a tight bound have few valid instances); the oracle
    flags that as low confidence by design, so a thin flag alone is not a test
    failure. The real coverage comes from hand-fed `extra` instances. When
    `discriminating` is set, this asserts the fixed batch (probes + extra) the
    original schema sees contains BOTH an accepted and a rejected instance — so
    the round-trip check exercises both directions and is not vacuous.
    """
    if discriminating:
        validator = Draft202012Validator(schema)
        fixed = list(oracle.DEFAULT_PROBE_INSTANCES) + list(extra)
        verdicts = {validator.is_valid(i) for i in fixed}
        assert verdicts == {True, False}, (
            f"non-discriminating batch for {schema!r}: every fixed instance is "
            f"{'accepted' if True in verdicts else 'rejected'}; add a counter-instance"
        )
    return oracle.assert_round_trip_preserves_meaning(schema, extra_instances=extra)


# ---------------------------------------------------------------------------
# Bucket 1 — cleanly encoded primitives and their constraints
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_bare_string(self) -> None:
        assert isinstance(from_json({"type": "string"}), String)
        _assert_meaning({"type": "string"}, discriminating=True)

    def test_bare_integer(self) -> None:
        _assert_meaning({"type": "integer"}, discriminating=True)

    def test_bare_number(self) -> None:
        _assert_meaning({"type": "number"}, discriminating=True)

    def test_bare_boolean(self) -> None:
        _assert_meaning({"type": "boolean"}, discriminating=True)

    def test_bare_null(self) -> None:
        _assert_meaning({"type": "null"})

    def test_string_length_bounds(self) -> None:
        schema = {"type": "string", "minLength": 2, "maxLength": 5}
        _assert_meaning(schema, extra=["a", "ab", "abcde", "abcdef"], discriminating=True)

    def test_string_pattern(self) -> None:
        schema = {"type": "string", "pattern": "^[a-z]+$"}
        _assert_meaning(schema, extra=["abc", "ABC", "a1b", ""], discriminating=True)

    def test_string_format_is_carried(self) -> None:
        # format is non-validating in Draft202012Validator, so this checks the
        # keyword rides along without changing behavior.
        node = from_json({"type": "string", "format": "email"})
        assert isinstance(node, String) and node.format == "email"
        _assert_meaning({"type": "string", "format": "email"}, discriminating=True)

    def test_string_content_encoding_media(self) -> None:
        schema = {"type": "string", "contentEncoding": "base64", "contentMediaType": "application/pdf"}
        node = from_json(schema)
        assert isinstance(node, String) and node.encoding == "base64" and node.media == "application/pdf"
        _assert_meaning(schema, discriminating=True)

    def test_integer_inclusive_bounds(self) -> None:
        schema = {"type": "integer", "minimum": 1, "maximum": 10}
        _assert_meaning(schema, extra=[0, 1, 10, 11, 5], discriminating=True)

    def test_integer_exclusive_bounds(self) -> None:
        schema = {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 5}
        node = from_json(schema)
        assert isinstance(node, Integer) and node.exclusive_min and node.exclusive_max
        _assert_meaning(schema, extra=[0, 1, 4, 5], discriminating=True)

    def test_number_multiple_of(self) -> None:
        schema = {"type": "number", "multipleOf": 0.5}
        _assert_meaning(schema, extra=[0, 0.5, 1.0, 0.25], discriminating=True)


class TestNumericConstraintValidation:
    """Ill-formed numeric constraint values fall back to raw JSON (§6.2 / §7.5)."""

    def test_string_typed_bounds_fall_back(self) -> None:
        tool = {
            "name": "t",
            "type": "object",
            "properties": {
                "n": {"type": "integer", "minimum": "1", "exclusiveMaximum": "3"},
            },
            "required": ["n"],
            "additionalProperties": False,
        }
        result = from_json_with_report(tool)
        assert result.fallbacks
        assert isinstance(result.ast, RawSchema)

    def test_multiple_of_zero_falls_back(self) -> None:
        result = from_json_with_report({"type": "integer", "multipleOf": 0})
        assert result.fallbacks
        assert "multipleOf must be a positive number" in result.fallbacks[0].reason

    def test_multiple_of_negative_falls_back(self) -> None:
        result = from_json_with_report({"type": "number", "multipleOf": -1})
        assert result.fallbacks
        assert "multipleOf must be a positive number" in result.fallbacks[0].reason

    def test_normal_numeric_schema_still_encodes(self) -> None:
        schema = {"type": "integer", "minimum": 1, "exclusiveMaximum": 3}
        node = from_json(schema)
        assert isinstance(node, Integer)
        assert node.minimum == 1 and node.maximum == 3 and node.exclusive_max


def _canonical_schema(obj):
    if isinstance(obj, dict):
        return {key: _canonical_schema(obj[key]) for key in sorted(obj)}
    if isinstance(obj, list):
        return [_canonical_schema(item) for item in obj]
    return obj


class TestNumericStructuralRoundTrip:
    """AST JSON -> CATS AST -> JSON must match exactly for numeric bounds (STEP 1)."""

    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "integer", "minimum": 1, "maximum": 3},
            {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 5},
            {"type": "integer", "minimum": 1, "exclusiveMaximum": 3},
            {"type": "integer", "exclusiveMinimum": 0, "maximum": 5},
            {"type": "integer", "minimum": 1},
            {"type": "integer", "exclusiveMaximum": 3},
            {"type": "number", "minimum": 0, "maximum": 1, "multipleOf": 0.25},
            {"type": "integer", "format": "int64", "minimum": 1, "maximum": 3},
        ],
    )
    def test_json_round_trip_is_structurally_identical(self, schema: dict) -> None:
        tripped = to_json(from_json(schema))
        assert _canonical_schema(tripped) == _canonical_schema(schema)


class TestArrays:
    def test_typed_array(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}}
        _assert_meaning(schema, extra=[[], [1, 2], [1, "x"], "nope"], discriminating=True)

    def test_array_bounds_and_unique(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 1,
                  "maxItems": 3, "uniqueItems": True}
        node = from_json(schema)
        assert isinstance(node, Array) and node.unique and node.min_items == 1
        _assert_meaning(schema, extra=[[], ["a"], ["a", "a"], ["a", "b", "c", "d"]],
                        discriminating=True)

    def test_bare_array(self) -> None:
        node = from_json({"type": "array"})
        assert isinstance(node, Array) and node.element is None
        _assert_meaning({"type": "array"}, extra=[[], [1, "x", True], "no"], discriminating=True)

    def test_nested_array(self) -> None:
        schema = {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}
        _assert_meaning(schema, extra=[[[1], [2, 3]], [[1], ["x"]], [1]], discriminating=True)


class TestObjects:
    def test_closed_object_with_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        }
        node = from_json(schema)
        assert isinstance(node, Object) and node.fields[0].required
        _assert_meaning(
            schema,
            extra=[{"id": 1}, {"id": 1, "name": "x"}, {"name": "x"},
                   {"id": 1, "extra": 9}, {}],
            discriminating=True,
        )

    def test_nested_object(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "addr": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                }
            },
            "required": ["addr"],
            "additionalProperties": False,
        }
        _assert_meaning(
            schema,
            extra=[{"addr": {"city": "x"}}, {"addr": {}}, {"addr": {"city": 1}}, {}],
            discriminating=True,
        )

    def test_property_description_is_carried_onto_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string", "description": "a label"}},
            "additionalProperties": False,
        }
        node = from_json(schema)
        assert isinstance(node, Object) and node.fields[0].description == "a label"
        _assert_meaning(schema, extra=[{"x": "y"}, {"x": 1}, {"z": 1}])


# ---------------------------------------------------------------------------
# Bucket 2 — transformed constructs
# ---------------------------------------------------------------------------


class TestTransformed:
    def test_anyof_union(self) -> None:
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert isinstance(from_json(schema), Union)
        _assert_meaning(schema, extra=["x", 5, True, None, []], discriminating=True)

    def test_type_array_nullable_union(self) -> None:
        schema = {"type": ["string", "null"]}
        assert isinstance(from_json(schema), Union)
        _assert_meaning(schema, extra=["x", None, 5, []], discriminating=True)

    def test_openapi_nullable_true_becomes_type_null_union(self) -> None:
        schema = {"type": "string", "nullable": True}
        node = from_json(schema)
        assert isinstance(node, Union)
        _assert_meaning(schema, extra=["x", None, 5, []], discriminating=True)

    def test_openapi_nullable_on_tool_field_encodes(self) -> None:
        tool = {
            "name": "patch_user",
            "type": "object",
            "properties": {
                "nickname": {"type": "string", "nullable": True},
            },
            "additionalProperties": False,
        }
        result = from_json_with_report(tool)
        assert result.fallback_count == 0
        assert isinstance(result.ast, ToolBlock)
        nickname = result.ast.fields[0]
        assert isinstance(nickname.type, Union)
        _assert_meaning(
            tool,
            extra=[
                {"nickname": "ada"},
                {"nickname": None},
                {"nickname": 1},
                {},
            ],
            discriminating=True,
        )

    def test_openapi_nullable_false_is_dropped(self) -> None:
        schema = {"type": "string", "nullable": False}
        assert isinstance(from_json(schema), String)
        _assert_meaning(schema, extra=["x", None, 1], discriminating=True)

    def test_multi_value_enum(self) -> None:
        schema = {"type": "string", "enum": ["draft", "published", "archived"]}
        node = from_json(schema)
        assert isinstance(node, Enum) and node.base_type == "string"
        _assert_meaning(schema, extra=["draft", "archived", "other", 1, None],
                        discriminating=True)

    def test_integer_enum(self) -> None:
        schema = {"type": "integer", "enum": [1, 2, 3]}
        _assert_meaning(schema, extra=[1, 3, 4, "1", 2.5], discriminating=True)

    def test_const_single_value(self) -> None:
        node = from_json({"const": "automatic"})
        assert isinstance(node, Const)
        _assert_meaning({"const": "automatic"}, extra=["automatic", "manual", 1, None],
                        discriminating=True)

    def test_single_element_enum_becomes_const(self) -> None:
        assert isinstance(from_json({"enum": ["only"]}), Const)
        _assert_meaning({"enum": ["only"]}, extra=["only", "other", 0])

    def test_true_false_set_becomes_boolean(self) -> None:
        assert isinstance(from_json({"enum": [True, False]}), Boolean)
        _assert_meaning({"enum": [True, False]}, extra=[True, False, "true", 1, None],
                        discriminating=True)

    def test_empty_schema_is_any(self) -> None:
        # `any` accepts EVERY value, so there is nothing to reject — the round
        # trip is checked, but a discrimination assertion would be impossible.
        _assert_meaning({}, extra=["x", 1, None, [], {}])

    def test_description_only_schema_is_any(self) -> None:
        # Description is validation-inert; {} and {"description":..} both accept all.
        _assert_meaning({"description": "anything goes"}, extra=["x", 1, None, []])

    def test_enum_with_redundant_constraint_is_dropped(self) -> None:
        # Every member already satisfies minLength:1, so it drops (§5.6) and the
        # field round-trips as a plain enum — behavior preserved.
        schema = {"type": "string", "enum": ["red", "green"], "minLength": 1}
        assert isinstance(from_json(schema), Enum)
        _assert_meaning(schema, extra=["red", "green", "blue", ""], discriminating=True)

    def test_examples_and_deprecated_drop_without_fallback(self) -> None:
        tool = {
            "name": "note",
            "type": "object",
            "properties": {
                "x": {
                    "type": "string",
                    "examples": ["a"],
                    "deprecated": True,
                    "example": "a",
                },
            },
            "additionalProperties": False,
        }
        result = from_json_with_report(tool)
        assert result.fallback_count == 0
        out = to_json(result.ast)
        assert isinstance(out, dict)
        field_schema = out["properties"]["x"]
        assert "examples" not in field_schema
        assert "deprecated" not in field_schema
        assert "example" not in field_schema

    def test_mixed_type_integer_string_enum_falls_back(self) -> None:
        tool = {
            "name": "t",
            "type": "object",
            "properties": {"level": {"enum": [1, "high"]}},
            "additionalProperties": False,
        }
        result = from_json_with_report(tool)
        assert isinstance(result.ast, RawSchema)
        assert result.fallback_count == 1

    def test_mixed_type_string_null_enum_falls_back(self) -> None:
        tool = {
            "name": "t",
            "type": "object",
            "properties": {"tag": {"enum": ["a", None]}},
            "additionalProperties": False,
        }
        result = from_json_with_report(tool)
        assert isinstance(result.ast, RawSchema)

    def test_reserved_format_length_falls_back(self) -> None:
        tool = {
            "name": "t",
            "type": "object",
            "properties": {"x": {"type": "string", "format": "length"}},
            "additionalProperties": False,
        }
        result = from_json_with_report(tool)
        assert isinstance(result.ast, RawSchema)

    def test_normal_format_email_still_encodes(self) -> None:
        node = from_json({"type": "string", "format": "email"})
        assert isinstance(node, String) and node.format == "email"

    def test_typeless_schema_with_constraint_falls_back(self) -> None:
        result = from_json_with_report({"minimum": 0})
        assert isinstance(result.ast, RawSchema)


# ---------------------------------------------------------------------------
# Tool-level and document (envelope) round trips
# ---------------------------------------------------------------------------


def _per_tool_equivalent(orig_tool: dict, got_tool: dict) -> bool:
    instances = list(oracle.DEFAULT_PROBE_INSTANCES)
    instances += [
        {}, {"x": 1}, {"x": "s"}, {"extra": 1},
        {"id": 1}, {"id": 1, "home": {"street": "s"}},
    ]
    instances += oracle.generate_instances(orig_tool).instances
    return oracle.behaviorally_equivalent(orig_tool, got_tool, instances)


def _assert_envelope_round_trips(envelope: list) -> None:
    out = oracle.round_trip(envelope)
    assert isinstance(out, list) and len(out) == len(envelope)
    for orig, got in zip(envelope, out):
        assert _per_tool_equivalent(orig, got), (orig, got)


class TestToolAndDocument:
    def test_single_tool_schema_becomes_toolblock(self) -> None:
        tool = to_json(
            ToolBlock(
                name="get-user",
                description="Look up a user",
                fields=[from_field_id_required(), from_field_name()],
            )
        )
        assert isinstance(tool, dict) and "name" in tool
        node = from_json(tool)
        assert isinstance(node, ToolBlock) and node.name == "get-user"
        out = oracle.round_trip(tool)
        assert _per_tool_equivalent(tool, out)

    def test_multi_tool_envelope_with_shared_identical_def(self) -> None:
        # Two tools both reference Address; to_json embeds an identical copy in
        # each. from_json must merge them into one document-level Definition.
        doc = Document(
            defs=[
                from_definition_address(),
            ],
            tools=[
                ToolBlock(name="a", fields=[from_field_home()]),
                ToolBlock(name="b", fields=[from_field_home()]),
            ],
        )
        envelope = to_json(doc)
        assert isinstance(envelope, list) and len(envelope) == 2

        result = from_json_with_report(envelope)
        assert isinstance(result.ast, Document)
        assert result.fallback_count == 0
        assert result.ast.defs is not None and len(result.ast.defs) == 1
        assert result.ast.defs[0].name == "Address"
        _assert_envelope_round_trips(envelope)

    def test_conflicting_same_named_defs_fall_back_the_using_tools(self) -> None:
        # Two tools carry DIFFERENT "Address" definitions under the same name.
        # Decision: both using tools fall back (conservative, behavior-preserving).
        tool_a = {
            "name": "a", "type": "object",
            "properties": {"home": {"$ref": "#/$defs/Address"}},
            "additionalProperties": False,
            "$defs": {"Address": {
                "type": "object", "properties": {"street": {"type": "string"}},
                "additionalProperties": False}},
        }
        tool_b = {
            "name": "b", "type": "object",
            "properties": {"home": {"$ref": "#/$defs/Address"}},
            "additionalProperties": False,
            "$defs": {"Address": {
                "type": "object", "properties": {"zip": {"type": "integer"}},
                "additionalProperties": False}},
        }
        envelope = [tool_a, tool_b]
        result = from_json_with_report(envelope)
        assert isinstance(result.ast, Document)
        assert all(isinstance(t, RawSchema) for t in result.ast.tools)
        assert result.fallback_count == 2
        assert any("Address" in fb.reason for fb in result.fallbacks)
        _assert_envelope_round_trips(envelope)


# Small builders kept out of the test bodies for readability.
def from_field_id_required():
    from nodes import Field
    return Field(name="id", type=Integer(minimum=1), required=True)


def from_field_name():
    from nodes import Field
    return Field(name="name", type=String())


def from_field_home():
    from nodes import Field
    return Field(name="home", type=Reference(name="Address"))


def from_definition_address():
    from nodes import Definition, Field
    return Definition(
        name="Address",
        fields=[Field(name="street", type=String(), required=True)],
    )


# ---------------------------------------------------------------------------
# Tool-level fallback (the §7.5 boundary) — must still round-trip and be RECORDED
# ---------------------------------------------------------------------------


class TestFallback:
    def test_bare_unencodable_schema_falls_back_to_rawschema(self) -> None:
        schema = {"not": {"type": "string"}}
        result = from_json_with_report(schema)
        assert isinstance(result.ast, RawSchema)
        assert result.fallback_count == 1
        # raw in -> RawSchema -> raw out, behaviorally identical.
        assert oracle.round_trip(schema) == schema
        _assert_meaning(schema, extra=["x", 1, None, [], {}])

    def test_tool_containing_not_falls_back_whole(self) -> None:
        tool = {
            "name": "t", "type": "object",
            "properties": {"q": {"not": {"type": "string"}}},
            "additionalProperties": False,
        }
        result = from_json_with_report(tool)
        assert isinstance(result.ast, RawSchema)
        assert result.fallback_count == 1
        assert result.fallbacks[0].location == "t"
        assert oracle.round_trip(tool) == tool

    def test_open_object_falls_back(self) -> None:
        # additionalProperties not false -> closing it would change behavior, so
        # this reader falls back rather than perform the §7.4 lossy conversion.
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = from_json_with_report(schema)
        assert isinstance(result.ast, RawSchema)
        _assert_meaning(schema, extra=[{"x": "s"}, {"x": "s", "y": 1}, {}, {"y": 1}])

    def test_oneof_falls_back(self) -> None:
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        result = from_json_with_report(schema)
        assert isinstance(result.ast, RawSchema)
        assert oracle.round_trip(schema) == schema

    def test_allof_falls_back(self) -> None:
        schema = {"allOf": [{"type": "object"}, {"type": "object"}]}
        assert isinstance(from_json(schema), RawSchema)
        assert oracle.round_trip(schema) == schema

    def test_envelope_with_one_fallback_tool(self) -> None:
        good = {
            "name": "good", "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        bad = {
            "name": "bad", "type": "object",
            "properties": {"q": {"not": {"type": "string"}}},
            "additionalProperties": False,
        }
        envelope = [good, bad]
        result = from_json_with_report(envelope)
        assert isinstance(result.ast, Document)
        assert isinstance(result.ast.tools[0], ToolBlock)
        assert isinstance(result.ast.tools[1], RawSchema)
        assert result.fallback_count == 1
        _assert_envelope_round_trips(envelope)


# ---------------------------------------------------------------------------
# Shaped object in non-encodable positions (to_cats serialization constraint)
# ---------------------------------------------------------------------------


class TestShapedObjectFallback:
    def test_shaped_object_in_array_of_array_falls_back(self) -> None:
        schema = {
            "name": "t",
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                            "additionalProperties": False,
                        }
                    }
                }
            },
            "additionalProperties": False,
        }
        result = from_json_with_report(schema)
        assert isinstance(result.ast, RawSchema)
        assert result.fallback_count == 1
        assert "shaped object" in result.fallbacks[0].reason.lower()

    def test_shaped_object_in_union_branch_falls_back(self) -> None:
        schema = {
            "name": "t",
            "type": "object",
            "properties": {
                "data": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {"nested_x": {"type": "integer"}},
                            "required": ["nested_x"],
                            "additionalProperties": False,
                        }
                    ]
                }
            },
            "additionalProperties": False,
        }
        result = from_json_with_report(schema)
        assert isinstance(result.ast, RawSchema)
        assert result.fallback_count == 1

    def test_shaped_object_in_array_element_does_not_fall_back(self) -> None:
        schema = {
            "name": "t",
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                        "additionalProperties": False,
                    }
                }
            },
            "additionalProperties": False,
        }
        result = from_json_with_report(schema)
        assert isinstance(result.ast, ToolBlock)
        assert result.fallback_count == 0

    def test_bare_object_in_array_of_array_does_not_fall_back(self) -> None:
        # An empty Object (no fields, validates only {}) is fine anywhere —
        # it has no fields to lose on serialization.
        schema = {
            "name": "t",
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": False}
                    }
                }
            },
            "additionalProperties": False,
        }
        result = from_json_with_report(schema)
        assert isinstance(result.ast, ToolBlock)
        assert result.fallback_count == 0

    def test_shaped_object_fallback_preserves_meaning_via_raw_path(self) -> None:
        schema = {
            "name": "t",
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"x": {"type": "integer", "minimum": 1}},
                            "required": ["x"],
                            "additionalProperties": False,
                        }
                    }
                }
            },
            "additionalProperties": False,
        }
        # The tool falls back to RawSchema; meaning is preserved through text too (§7.5).
        result = from_json_with_report(schema)
        assert isinstance(result.ast, RawSchema)
        tripped = to_json(result.ast)
        # Verify the meaning is preserved: the fallback is verbatim.
        instances = [
            {"matrix": [[{"x": 1}]]},
            {"matrix": [[{"x": 0}]]},  # violates minimum:1
            {"matrix": [[]]},
            {},
        ]
        from jsonschema import Draft202012Validator
        orig = Draft202012Validator(schema)
        trip = Draft202012Validator(tripped)
        for inst in instances:
            assert orig.is_valid(inst) == trip.is_valid(inst), (
                f"meaning changed for {inst}: "
                f"orig={orig.is_valid(inst)}, tripped={trip.is_valid(inst)}"
            )


# ---------------------------------------------------------------------------
# Reporting surface
# ---------------------------------------------------------------------------


class TestLoadSchema:
    def test_json_text_with_lowercase_false_parses(self) -> None:
        text = '{"type":"object","additionalProperties":false}'
        assert load_schema(text)["additionalProperties"] is False

    def test_dict_with_string_false_normalizes(self) -> None:
        assert load_schema({"additionalProperties": "false"})["additionalProperties"] is False

    def test_dict_with_python_bool_unchanged(self) -> None:
        assert load_schema({"additionalProperties": False})["additionalProperties"] is False


class TestReportingSurface:
    def test_from_json_with_report_returns_conversion_result(self) -> None:
        result = from_json_with_report({"type": "string"})
        assert isinstance(result, ConversionResult)
        assert result.fallback_count == 0
        assert result.fallbacks == []

    def test_fallback_record_has_location_and_reason(self) -> None:
        result = from_json_with_report({"not": {}})
        assert result.fallback_count == 1
        record = result.fallbacks[0]
        assert isinstance(record, FallbackRecord)
        assert record.location is None  # bare schema, no tool name
        assert "not" in record.reason


# ---------------------------------------------------------------------------
# Provider tool-definition envelope unwrapping (§7.6)
# ---------------------------------------------------------------------------


def _full_pipeline(schema):
    """JSON -> from_json -> to_cats -> text -> parse -> validate -> to_json.

    The same end-to-end path the demo uses, including §7.5 raw JSON in CATS text.
    """
    from parser import parse_text
    from to_cats import to_cats
    from validate import validate

    ast = from_json(schema)
    if isinstance(ast, (Document, ToolBlock, RawSchema)):
        doc = parse_text(to_cats(ast))
        validate(doc)
        out = to_json(doc)
    else:
        out = to_json(ast)
    if isinstance(out, list) and len(out) == 1:
        out = out[0]
    return out


def _assert_inner_meaning_preserved(wrapped, inner, instances):
    """The full pipeline on `wrapped` must accept/reject `instances` the same way
    the inner parameter schema does."""
    out = _full_pipeline(wrapped)
    all_instances = list(instances) + oracle.generate_instances(inner).instances
    disagreements = oracle.behavioral_disagreements(inner, out, all_instances)
    assert not disagreements, f"meaning changed: {disagreements}"


# A realistic OpenAI Responses-style tool def (flat: name + parameters + strict).
OPENAI_FLAT = {
    "name": "assess_answer",
    "description": "Assess a student's answer and assign a score",
    "parameters": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
            "feedback": {"type": "string", "minLength": 1},
        },
        "required": ["score"],
        "additionalProperties": False,
    },
    "strict": True,
}

ANTHROPIC = {
    "name": "get_weather",
    "description": "Look up the weather",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
    "cache_control": {"type": "ephemeral"},
}

OPENAI_CHAT = {
    "type": "function",
    "function": {
        "name": "set_reminder",
        "description": "Create a reminder",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "minutes": {"type": "integer", "minimum": 1}},
            "required": ["text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class TestProviderEnvelope:
    def test_openai_flat_encodes_and_drops_strict(self) -> None:
        result = from_json_with_report(OPENAI_FLAT)
        assert isinstance(result.ast, ToolBlock)  # encoded, NOT fallen back
        assert result.fallback_count == 0
        assert result.ast.name == "assess_answer"
        assert result.ast.description == "Assess a student's answer and assign a score"
        # `strict` is dropped: it never reaches the AST or the output schema.
        out = to_json(result.ast)
        assert "strict" not in out
        _assert_inner_meaning_preserved(
            OPENAI_FLAT,
            OPENAI_FLAT["parameters"],
            [
                {"score": 0}, {"score": 100}, {"score": -1}, {"score": 101},
                {"feedback": "x"}, {"score": 50, "feedback": "ok"},
                {"score": 50, "extra": 1}, {},
            ],
        )

    def test_anthropic_input_schema_unwraps_and_encodes(self) -> None:
        result = from_json_with_report(ANTHROPIC)
        assert isinstance(result.ast, ToolBlock)
        assert result.fallback_count == 0
        assert result.ast.name == "get_weather"
        _assert_inner_meaning_preserved(
            ANTHROPIC,
            ANTHROPIC["input_schema"],
            [
                {"city": "x"}, {"city": ""}, {"city": "x", "units": "celsius"},
                {"city": "x", "units": "kelvin"}, {}, {"units": "celsius"},
            ],
        )

    def test_openai_chat_completions_nested_unwraps(self) -> None:
        result = from_json_with_report(OPENAI_CHAT)
        assert isinstance(result.ast, ToolBlock)
        assert result.fallback_count == 0
        assert result.ast.name == "set_reminder"
        _assert_inner_meaning_preserved(
            OPENAI_CHAT,
            OPENAI_CHAT["function"]["parameters"],
            [
                {"text": "x"}, {"text": "x", "minutes": 1}, {"text": "x", "minutes": 0},
                {"minutes": 5}, {},
            ],
        )

    def test_bare_schema_without_wrapper_is_unchanged(self) -> None:
        # Regression: a plain object schema (no wrapper) is still read as a bare
        # Object node, exactly as before — `properties` marks it as a schema.
        bare = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        result = from_json_with_report(bare)
        assert isinstance(result.ast, Object)
        assert result.fallback_count == 0

    def test_cats_output_envelope_not_misread_as_wrapper(self) -> None:
        # A CATS output tool schema (name + type + properties) must still be read
        # as a tool, not unwrapped — it has top-level `properties`.
        tool = {
            "name": "t",
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        result = from_json_with_report(tool)
        assert isinstance(result.ast, ToolBlock)
        assert result.ast.name == "t"

    def test_wrapped_tool_with_inner_not_still_falls_back(self) -> None:
        # The wrapper is unwrapped; the inner schema's `not` triggers fallback on
        # its own merits, not because of the wrapper.
        wrapped = {
            "name": "weird",
            "description": "has an out-of-scope inner schema",
            "parameters": {
                "type": "object",
                "properties": {"q": {"not": {"type": "string"}}},
                "additionalProperties": False,
            },
            "strict": True,
        }
        result = from_json_with_report(wrapped)
        assert isinstance(result.ast, RawSchema)
        assert result.fallback_count == 1
        assert "not" in result.fallbacks[0].reason
        # The fallback carries the unwrapped parameter schema (the `not` survives
        # verbatim; envelope runtime flags like `strict` are gone). The lifted
        # name/description are validation-inert, so meaning matches the inner.
        carried = result.ast.schema
        assert "strict" not in carried
        assert carried["properties"] == wrapped["parameters"]["properties"]
        instances = [{"q": 1}, {"q": "s"}, {}, {"q": "s", "other": 1}]
        assert oracle.behaviorally_equivalent(wrapped["parameters"], carried, instances)


# ---------------------------------------------------------------------------
# Legacy `definitions` -> `$defs` normalization (§7.6)
# ---------------------------------------------------------------------------

_LEGACY_ADDRESS_DEF = {
    "type": "object",
    "properties": {"street": {"type": "string"}},
    "required": ["street"],
    "additionalProperties": False,
}

_TOOL_WITH_LEGACY_DEFINITIONS = {
    "name": "get-user",
    "type": "object",
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "home": {"$ref": "#/definitions/Address"},
    },
    "required": ["id"],
    "additionalProperties": False,
    "definitions": {"Address": _LEGACY_ADDRESS_DEF},
}

_TOOL_WITH_MODERN_DEFS = {
    "name": "get-user",
    "type": "object",
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "home": {"$ref": "#/$defs/Address"},
    },
    "required": ["id"],
    "additionalProperties": False,
    "$defs": {"Address": _LEGACY_ADDRESS_DEF},
}


def _assert_single_tool_round_trips(tool: dict, *, extra=(), discriminating=False) -> None:
    """Round-trip one tool schema; `to_json` emits a one-element list for Documents."""
    tool = copy.deepcopy(tool)
    if discriminating:
        validator = Draft202012Validator(tool)
        fixed = list(oracle.DEFAULT_PROBE_INSTANCES) + list(extra)
        verdicts = {validator.is_valid(i) for i in fixed}
        assert verdicts == {True, False}, (
            f"non-discriminating batch for {tool!r}: every fixed instance is "
            f"{'accepted' if True in verdicts else 'rejected'}; add a counter-instance"
        )
    out = oracle.round_trip(tool)
    assert isinstance(out, list) and len(out) == 1
    assert _per_tool_equivalent(tool, out[0])


class TestLegacyDefinitions:
    def test_definitions_and_legacy_ref_encode(self) -> None:
        tool = copy.deepcopy(_TOOL_WITH_LEGACY_DEFINITIONS)
        result = from_json_with_report(tool)
        assert result.fallback_count == 0
        assert isinstance(result.ast, Document)
        assert len(result.ast.tools) == 1
        assert isinstance(result.ast.tools[0], ToolBlock)
        assert result.ast.defs is not None and len(result.ast.defs) == 1
        assert result.ast.defs[0].name == "Address"
        home = result.ast.tools[0].fields[1]
        assert isinstance(home.type, Reference) and home.type.name == "Address"
        _assert_single_tool_round_trips(
            tool,
            extra=[
                {"id": 1, "home": {"street": "Main"}},
                {"id": 0},
                {"id": 1},
                {"id": 1, "home": {}},
                {"id": 1, "home": {"street": "s", "extra": 1}},
            ],
            discriminating=True,
        )

    def test_modern_defs_produces_identical_ast(self) -> None:
        legacy_ast = from_json(copy.deepcopy(_TOOL_WITH_LEGACY_DEFINITIONS))
        modern_ast = from_json(copy.deepcopy(_TOOL_WITH_MODERN_DEFS))
        assert legacy_ast == modern_ast

    def test_both_definitions_and_defs_fall_back(self) -> None:
        conflict = {
            **_TOOL_WITH_LEGACY_DEFINITIONS,
            "$defs": {"Other": {"type": "string"}},
        }
        result = from_json_with_report(conflict)
        assert isinstance(result.ast, RawSchema)
        assert result.fallback_count == 1
        assert "definitions" in result.fallbacks[0].reason
        assert "$defs" in result.fallbacks[0].reason
        assert "definitions" in result.ast.schema
        assert "$defs" in result.ast.schema
        assert oracle.round_trip(copy.deepcopy(conflict)) == conflict

    def test_only_modern_defs_unchanged(self) -> None:
        tool = copy.deepcopy(_TOOL_WITH_MODERN_DEFS)
        result = from_json_with_report(tool)
        assert result.fallback_count == 0
        assert isinstance(result.ast, Document)
        assert result.ast.defs is not None and result.ast.defs[0].name == "Address"
        _assert_single_tool_round_trips(tool, extra=[{"id": 1, "home": {"street": "s"}}])
