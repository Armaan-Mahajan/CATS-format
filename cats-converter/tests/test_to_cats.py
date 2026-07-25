"""Tests for to_cats.py — the CATS AST -> text serializer.

Three layers:
  1. AST round-trip: for each node type, parse_text(to_cats(ast)) reproduces a
     semantically identical AST. Trees are built directly so each test pins one
     construct; the field/type is wrapped in a tool so the text is a real
     document the lexer + parser accept.
  2. Full-pipeline oracle: JSON -> from_json -> to_cats -> text -> parse ->
     validate -> to_json preserves MEANING for representative in-scope schemas.
  3. RawSchema fallback (§7.5): verbatim JSON in the document round-trips
     through text.
"""

from __future__ import annotations

import oracle
from from_json import from_json_with_report
from nodes import (
    NO_DEFAULT,
    AnyType,
    Array,
    Boolean,
    Const,
    Definition,
    Document,
    Enum,
    Field,
    Integer,
    Null,
    Number,
    Object,
    RawSchema,
    Reference,
    String,
    ToolBlock,
    Union,
)
from parser import parse_text
from to_cats import to_cats
from to_json import to_json
from validate import validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def roundtrip_type(type_node):
    """Wrap a type in a one-field tool, serialize, re-parse, return the type."""
    doc = Document(tools=[ToolBlock(name="t", fields=[Field(name="f", type=type_node)])])
    reparsed = parse_text(to_cats(doc))
    return reparsed.tools[0].fields[0].type


def roundtrip_field(field: Field) -> Field:
    doc = Document(tools=[ToolBlock(name="t", fields=[field])])
    reparsed = parse_text(to_cats(doc))
    return reparsed.tools[0].fields[0]


def roundtrip_document(doc: Document) -> Document:
    return parse_text(to_cats(doc))


# ---------------------------------------------------------------------------
# 1. AST round-trip, per node type
# ---------------------------------------------------------------------------


class TestPrimitiveRoundTrip:
    def test_string(self) -> None:
        assert roundtrip_type(String()) == String()

    def test_integer(self) -> None:
        assert roundtrip_type(Integer()) == Integer()

    def test_number(self) -> None:
        assert roundtrip_type(Number()) == Number()

    def test_boolean(self) -> None:
        assert roundtrip_type(Boolean()) == Boolean()

    def test_null(self) -> None:
        assert roundtrip_type(Null()) == Null()

    def test_any(self) -> None:
        assert roundtrip_type(AnyType()) == AnyType()

    def test_string_with_format(self) -> None:
        assert roundtrip_type(String(format="email")) == String(format="email")


class TestConstrainedRoundTrip:
    def test_string_length_and_pattern(self) -> None:
        node = String(min_length=1, max_length=20, pattern="^[a-z]+$")
        assert roundtrip_type(node) == node

    def test_string_open_lower_length(self) -> None:
        node = String(max_length=5)
        assert roundtrip_type(node) == node

    def test_string_encoding_and_media(self) -> None:
        node = String(encoding="base64", media="application/pdf")
        assert roundtrip_type(node) == node

    def test_integer_inclusive_bounds(self) -> None:
        node = Integer(minimum=1, maximum=100)
        assert roundtrip_type(node) == node

    def test_integer_exclusive_bounds(self) -> None:
        node = Integer(minimum=0, maximum=5, exclusive_min=True, exclusive_max=True)
        assert roundtrip_type(node) == node

    def test_integer_open_upper_bound(self) -> None:
        node = Integer(minimum=1)
        assert roundtrip_type(node) == node

    def test_open_upper_emits_inclusive_bracket_per_spec(self) -> None:
        node = Integer(minimum=1, exclusive_max=True, maximum=None)
        assert to_cats(node) == "integer[1,]"

    def test_open_lower_emits_inclusive_bracket_per_spec(self) -> None:
        node = Integer(maximum=10, exclusive_min=True, minimum=None)
        assert to_cats(node) == "integer[,10]"

    def test_closed_exclusive_bounds_unchanged(self) -> None:
        node = Integer(minimum=1, maximum=3, exclusive_max=True)
        assert to_cats(node) == "integer[1,3)"

    def test_number_multiple_of(self) -> None:
        node = Number(multiple_of=0.5)
        assert roundtrip_type(node) == node

    def test_integer_bounds_and_multiple_of(self) -> None:
        node = Integer(minimum=1, maximum=100, multiple_of=5)
        assert roundtrip_type(node) == node


class TestArrayRoundTrip:
    def test_bare_array(self) -> None:
        assert roundtrip_type(Array()) == Array()

    def test_typed_array(self) -> None:
        node = Array(element=String())
        assert roundtrip_type(node) == node

    def test_array_bounds_and_unique(self) -> None:
        node = Array(element=String(), min_items=1, max_items=3, unique=True)
        assert roundtrip_type(node) == node

    def test_nested_array(self) -> None:
        node = Array(element=Array(element=Integer()))
        assert roundtrip_type(node) == node


class TestObjectRoundTrip:
    def test_bare_object(self) -> None:
        assert roundtrip_type(Object()) == Object()

    def test_object_with_fields(self) -> None:
        node = Object(
            fields=[
                Field(name="street", type=String(), required=True),
                Field(name="zip", type=Integer()),
            ]
        )
        assert roundtrip_type(node) == node

    def test_nested_object(self) -> None:
        inner = Object(fields=[Field(name="city", type=String(), required=True)])
        node = Object(fields=[Field(name="addr", type=inner, required=True)])
        assert roundtrip_type(node) == node

    def test_array_of_objects(self) -> None:
        element = Object(fields=[Field(name="name", type=String(), required=True)])
        node = Array(element=element)
        assert roundtrip_type(node) == node


class TestUnionEnumConstRoundTrip:
    def test_type_union(self) -> None:
        node = Union(branches=[String(), Null()])
        assert roundtrip_type(node) == node

    def test_union_three_branches(self) -> None:
        node = Union(branches=[String(), Integer(), Null()])
        assert roundtrip_type(node) == node

    def test_string_enum(self) -> None:
        node = Enum(values=["draft", "published", "archived"], base_type="string")
        assert roundtrip_type(node) == node

    def test_integer_enum(self) -> None:
        node = Enum(values=[1, 2, 3], base_type="integer")
        assert roundtrip_type(node) == node

    def test_enum_member_colliding_with_type_word_is_quoted(self) -> None:
        # "string" as a value must round-trip as a value, not the string TYPE.
        node = Enum(values=["string", "number"], base_type="string")
        assert roundtrip_type(node) == node

    def test_enum_member_looking_like_number_is_quoted(self) -> None:
        node = Enum(values=["1", "2"], base_type="string")
        assert roundtrip_type(node) == node

    def test_string_const(self) -> None:
        assert roundtrip_type(Const(value="automatic")) == Const(value="automatic")

    def test_integer_const(self) -> None:
        assert roundtrip_type(Const(value=5)) == Const(value=5)


class TestReferenceAndDefs:
    def test_reference(self) -> None:
        node = Reference(name="Address")
        assert roundtrip_type(node) == node

    def test_document_with_defs(self) -> None:
        doc = Document(
            defs=[
                Definition(
                    name="Address",
                    fields=[
                        Field(name="street", type=String(), required=True),
                        Field(name="city", type=String()),
                    ],
                )
            ],
            tools=[
                ToolBlock(
                    name="get-user",
                    description="Look up a user",
                    fields=[
                        Field(name="id", type=Integer(minimum=1), required=True),
                        Field(name="home", type=Reference(name="Address")),
                    ],
                )
            ],
        )
        assert roundtrip_document(doc) == doc


# ---------------------------------------------------------------------------
# Field-level features: required, defaults, descriptions, quoting
# ---------------------------------------------------------------------------


class TestFieldFeatures:
    def test_required_marker(self) -> None:
        field = Field(name="id", type=Integer(), required=True)
        assert roundtrip_field(field) == field

    def test_string_default(self) -> None:
        field = Field(name="mode", type=String(), default="auto")
        assert roundtrip_field(field) == field

    def test_null_default(self) -> None:
        field = Field(name="x", type=String(), default=None)
        assert roundtrip_field(field) == field

    def test_numeric_default(self) -> None:
        field = Field(name="count", type=Integer(), default=5)
        assert roundtrip_field(field) == field

    def test_boolean_default(self) -> None:
        field = Field(name="flag", type=Boolean(), default=True)
        assert roundtrip_field(field) == field

    def test_object_default_empty(self) -> None:
        field = Field(name="config", type=Object(), default={})
        assert roundtrip_field(field) == field

    def test_array_default_empty(self) -> None:
        field = Field(name="tags", type=Array(element=String()), default=[])
        assert roundtrip_field(field) == field

    def test_description(self) -> None:
        field = Field(name="x", type=String(), description="a label")
        assert roundtrip_field(field) == field

    def test_description_with_hash_is_quoted(self) -> None:
        field = Field(name="x", type=String(), description="issue #42 here")
        assert roundtrip_field(field) == field

    def test_hyphenated_name_stays_bare(self) -> None:
        field = Field(name="model-name", type=String())
        got = roundtrip_field(field)
        assert got == field
        assert "model-name" in to_cats(
            Document(tools=[ToolBlock(name="t", fields=[field])])
        )

    def test_field_name_colliding_with_type_word_is_quoted(self) -> None:
        field = Field(name="string", type=Integer())
        assert roundtrip_field(field) == field


# ---------------------------------------------------------------------------
# 2. Full-pipeline oracle (JSON -> CATS text -> JSON preserves meaning)
# ---------------------------------------------------------------------------


def _pipeline_round_trip(schema):
    """JSON -> from_json -> to_cats -> text -> parse -> validate -> to_json."""
    result = from_json_with_report(schema)
    ast = result.ast
    if isinstance(ast, (Document, ToolBlock, RawSchema)):
        reparsed = parse_text(to_cats(ast))
        validate(reparsed)
        out = to_json(reparsed)
    else:
        out = to_json(ast)
    return out


def _assert_pipeline_preserves_meaning(schema, *, extra=()):
    saved = oracle.round_trip
    oracle.round_trip = _pipeline_round_trip
    try:
        return oracle.assert_round_trip_preserves_meaning(schema, extra_instances=extra)
    finally:
        oracle.round_trip = saved


TOOL_SCHEMA = {
    "name": "get-user",
    "type": "object",
    "properties": {
        "id": {"type": "integer", "minimum": 1, "maximum": 100},
        "name": {"type": "string", "minLength": 1, "maxLength": 50},
        "status": {"type": "string", "enum": ["active", "inactive"]},
        "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "nickname": {"type": ["string", "null"]},
    },
    "required": ["id"],
    "additionalProperties": False,
}

NESTED_TOOL_SCHEMA = {
    "name": "create-order",
    "type": "object",
    "properties": {
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "zip": {"type": "string", "pattern": "^[0-9]{5}$"},
            },
            "required": ["street"],
            "additionalProperties": False,
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["address"],
    "additionalProperties": False,
}

DOC_WITH_DEFS = [
    {
        "name": "a",
        "type": "object",
        "properties": {"home": {"$ref": "#/$defs/Address"}},
        "additionalProperties": False,
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"street": {"type": "string"}},
                "required": ["street"],
                "additionalProperties": False,
            }
        },
    }
]


class TestFullPipelineOracle:
    def test_tool_schema(self) -> None:
        _assert_pipeline_preserves_meaning(
            TOOL_SCHEMA,
            extra=[
                {"id": 1}, {"id": 0}, {"id": 1, "status": "active"},
                {"id": 1, "status": "other"}, {"id": 1, "tags": ["a", "a"]},
                {"id": 1, "nickname": None}, {"name": "x"}, {"id": 1, "extra": 9},
            ],
        )

    def test_nested_tool_schema(self) -> None:
        _assert_pipeline_preserves_meaning(
            NESTED_TOOL_SCHEMA,
            extra=[
                {"address": {"street": "x"}},
                {"address": {"street": "x", "zip": "12345"}},
                {"address": {"street": "x", "zip": "bad"}},
                {"address": {}},
                {"items": [{"sku": "a"}]},
            ],
        )

    def test_document_with_defs(self) -> None:
        # The list envelope re-parses to a list; compare list-to-list directly.
        out = _pipeline_round_trip(DOC_WITH_DEFS)
        assert isinstance(out, list) and len(out) == 1
        instances = [
            {"home": {"street": "x"}},
            {"home": {}},
            {"home": {"street": 1}},
            {},
        ]
        assert oracle.behaviorally_equivalent(DOC_WITH_DEFS[0], out[0], instances)


# ---------------------------------------------------------------------------
# 3. RawSchema verbatim JSON in CATS text (§7.5)
# ---------------------------------------------------------------------------

FALLBACK_TOOL = {
    "name": "t",
    "type": "object",
    "properties": {"q": {"not": {"type": "string"}}},
    "additionalProperties": False,
}


class TestRawSchemaTextRoundTrip:
    def test_fallback_tool_emits_single_line_json_at_column_zero(self) -> None:
        text = to_cats(RawSchema(schema=FALLBACK_TOOL))
        assert text.startswith("{")
        assert "\n" not in text
        assert "not" in text
        assert "FALLBACK" not in text

    def test_fallback_tool_emits_json_object_at_column_zero(self) -> None:
        text = to_cats(RawSchema(schema=FALLBACK_TOOL))
        assert text.startswith("{")
        assert "not" in text
        assert "FALLBACK" not in text

    def test_fallback_round_trips_through_text_pipeline(self) -> None:
        result = from_json_with_report(FALLBACK_TOOL)
        assert isinstance(result.ast, RawSchema)
        assert result.fallback_count == 1
        _assert_pipeline_preserves_meaning(
            FALLBACK_TOOL,
            extra=[{"q": 1}, {"q": "s"}, {}],
        )

    def test_mixed_document_round_trips_cats_and_raw_tools(self) -> None:
        envelope = [
            {
                "name": "good",
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "additionalProperties": False,
            },
            FALLBACK_TOOL,
        ]
        result = from_json_with_report(envelope)
        assert isinstance(result.ast, Document)
        assert isinstance(result.ast.tools[0], ToolBlock)
        assert isinstance(result.ast.tools[1], RawSchema)
        text = to_cats(result.ast)
        assert "good" in text
        assert text.strip().split("\n\n")[-1].startswith("{")

        reparsed = parse_text(text)
        assert isinstance(reparsed.tools[0], ToolBlock)
        assert isinstance(reparsed.tools[1], RawSchema)
        assert reparsed.tools[1].schema == FALLBACK_TOOL

        out = to_json(reparsed)
        assert isinstance(out, list) and len(out) == 2
        assert oracle.behaviorally_equivalent(
            envelope[0], out[0], [{"x": "a"}, {}, {"x": "a", "y": 1}]
        )
        assert oracle.behaviorally_equivalent(
            envelope[1], out[1], [{"q": 1}, {"q": "s"}, {}]
        )
