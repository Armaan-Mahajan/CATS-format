"""Unit tests for the CATS -> JSON Schema serializer (cats-converter/to_json.py).

The forward direction is mechanical (§7.2): one correct output per node. These
tests cover, per the task:
  - one assertion per node type (structural ==)
  - the §7.2 keyword emission ORDER on multi-keyword schemas
  - `required` built in field-declaration order; `additionalProperties: false`
  - the canonicalizations the serializer owns (true|false -> boolean; closed obj)
  - a `$ref` for a Reference and a full Document with `$defs`
  - a json.dumps/json.loads round-trip (the dict is JSON-legal)
  - draft 2020-12 meta-validation of representative outputs via `jsonschema`

Trees are built directly from node classes so each test targets one mapping; a
couple of cases also go through the real parser to pin the parser/serializer seam.
"""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator

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
from to_json import count_definition_inlinings, to_json, to_json_string


# ---------------------------------------------------------------------------
# One test per node type
# ---------------------------------------------------------------------------


class TestPrimitiveNodes:
    def test_bare_string(self) -> None:
        assert to_json(String()) == {"type": "string"}

    def test_string_with_all_constraints(self) -> None:
        node = String(
            format="email",
            min_length=1,
            max_length=20,
            pattern="^[a-z]+$",
            encoding="base64",
            media="text/plain",
        )
        assert to_json(node) == {
            "type": "string",
            "format": "email",
            "minLength": 1,
            "maxLength": 20,
            "pattern": "^[a-z]+$",
            "contentEncoding": "base64",
            "contentMediaType": "text/plain",
        }

    def test_integer_inclusive_bounds(self) -> None:
        assert to_json(Integer(minimum=1, maximum=100)) == {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
        }

    def test_integer_exclusive_bounds_use_2020_12_numeric_form(self) -> None:
        # Draft 2020-12: exclusive* carry the numeric value, not a boolean, and
        # replace the inclusive keyword on that side.
        node = Integer(minimum=0, maximum=1, exclusive_min=True, exclusive_max=True)
        assert to_json(node) == {
            "type": "integer",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 1,
        }

    def test_number_multiple_of_and_format(self) -> None:
        assert to_json(Number(multiple_of=0.01, format="double")) == {
            "type": "number",
            "format": "double",
            "multipleOf": 0.01,
        }

    def test_boolean(self) -> None:
        assert to_json(Boolean()) == {"type": "boolean"}

    def test_null(self) -> None:
        assert to_json(Null()) == {"type": "null"}

    def test_any_no_description_is_empty_schema(self) -> None:
        assert to_json(AnyType()) == {}

    def test_any_with_description_via_field_has_no_type(self) -> None:
        field = Field(name="x", type=AnyType(), description="anything goes")
        assert to_json(field) == {"description": "anything goes"}


class TestArrayNode:
    def test_parameterized_array(self) -> None:
        assert to_json(Array(element=String())) == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_bare_array_has_no_items(self) -> None:
        assert to_json(Array()) == {"type": "array"}

    def test_array_with_bounds_and_unique(self) -> None:
        node = Array(element=Integer(), min_items=1, max_items=10, unique=True)
        assert to_json(node) == {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
            "maxItems": 10,
            "uniqueItems": True,
        }

    def test_unique_false_is_omitted(self) -> None:
        assert "uniqueItems" not in to_json(Array(element=String(), unique=False))


class TestObjectNode:
    def test_object_with_properties_required_and_closed(self) -> None:
        node = Object(
            fields=[
                Field(name="b", type=String(), required=True),
                Field(name="a", type=Integer()),
            ]
        )
        assert to_json(node) == {
            "type": "object",
            "properties": {"b": {"type": "string"}, "a": {"type": "integer"}},
            "required": ["b"],
            "additionalProperties": False,
        }

    def test_bare_object_emits_empty_properties_and_closed(self) -> None:
        assert to_json(Object()) == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def test_no_required_array_when_none_required(self) -> None:
        node = Object(fields=[Field(name="a", type=String())])
        assert "required" not in to_json(node)


class TestCompositionNodes:
    def test_reference_emits_ref_pointer(self) -> None:
        assert to_json(Reference(name="Address")) == {"$ref": "#/$defs/Address"}

    def test_union_emits_anyof(self) -> None:
        node = Union(branches=[String(), Null()])
        assert to_json(node) == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    def test_string_enum_emits_type_and_enum(self) -> None:
        node = Enum(values=["a", "b", "c"], base_type="string")
        assert to_json(node) == {"type": "string", "enum": ["a", "b", "c"]}

    def test_integer_enum_emits_type_and_enum(self) -> None:
        node = Enum(values=[1, 2, 3], base_type="integer")
        assert to_json(node) == {"type": "integer", "enum": [1, 2, 3]}

    def test_const_string(self) -> None:
        assert to_json(Const(value="automatic")) == {"const": "automatic"}

    def test_const_number(self) -> None:
        assert to_json(Const(value=200)) == {"const": 200}

    def test_raw_schema_emitted_verbatim(self) -> None:
        raw = {"allOf": [{"type": "string"}, {"minLength": 2}]}
        assert to_json(RawSchema(schema=raw)) == raw

    def test_raw_schema_is_deep_copied_not_aliased(self) -> None:
        raw = {"not": {"type": "string"}}
        node = RawSchema(schema=raw)
        out = to_json(node)
        out["not"]["type"] = "integer"  # mutate the OUTPUT
        assert node.schema == {"not": {"type": "string"}}  # tree unchanged


# ---------------------------------------------------------------------------
# Field default and description
# ---------------------------------------------------------------------------


class TestFieldDefaultsAndDescription:
    def test_no_default_emits_no_default_keyword(self) -> None:
        assert "default" not in to_json(Field(name="x", type=String()))

    def test_default_value_emitted(self) -> None:
        field = Field(name="x", type=Integer(), default=5)
        assert to_json(field) == {"type": "integer", "default": 5}

    def test_explicit_null_default_is_emitted(self) -> None:
        # A stored None is an explicit null default (distinct from NO_DEFAULT).
        field = Field(name="x", type=String(), default=None)
        out = to_json(field)
        assert "default" in out
        assert out["default"] is None

    def test_object_and_array_json_defaults(self) -> None:
        assert to_json(Field(name="c", type=Object(), default={}))["default"] == {}
        assert to_json(Field(name="x", type=Array(element=Integer()), default=[1, 2]))[
            "default"
        ] == [1, 2]

    def test_description_emitted_last(self) -> None:
        field = Field(name="x", type=String(), description="a label")
        assert to_json(field) == {"type": "string", "description": "a label"}


# ---------------------------------------------------------------------------
# §7.2 keyword emission ORDER
# ---------------------------------------------------------------------------


class TestKeywordOrder:
    def test_constrained_string_key_order(self) -> None:
        node = String(
            format="email",
            min_length=1,
            max_length=20,
            pattern="^x$",
            encoding="base64",
            media="text/plain",
        )
        assert list(to_json(node).keys()) == [
            "type",
            "format",
            "minLength",
            "maxLength",
            "pattern",
            "contentEncoding",
            "contentMediaType",
        ]

    def test_field_default_sits_after_format_before_constraints(self) -> None:
        # type, format, default, then validation constraints, then description.
        field = Field(
            name="x",
            type=String(format="email", min_length=3),
            default="a@b.com",
            description="addr",
        )
        assert list(to_json(field).keys()) == [
            "type",
            "format",
            "default",
            "minLength",
            "description",
        ]

    def test_numeric_lower_before_upper_with_multiple_of(self) -> None:
        node = Integer(minimum=0, maximum=10, exclusive_max=True, multiple_of=2, format="int32")
        assert list(to_json(node).keys()) == [
            "type",
            "format",
            "minimum",
            "exclusiveMaximum",
            "multipleOf",
        ]

    def test_object_structural_key_order(self) -> None:
        node = Object(fields=[Field(name="a", type=String(), required=True)])
        out = to_json(node)
        assert list(out.keys()) == [
            "type",
            "properties",
            "required",
            "additionalProperties",
        ]

    def test_array_default_before_items(self) -> None:
        field = Field(name="xs", type=Array(element=String(), min_items=1), default=[])
        assert list(to_json(field).keys()) == ["type", "default", "items", "minItems"]


# ---------------------------------------------------------------------------
# Required-array declaration order
# ---------------------------------------------------------------------------


class TestRequiredOrder:
    def test_required_follows_declaration_order(self) -> None:
        node = Object(
            fields=[
                Field(name="first", type=String(), required=True),
                Field(name="second", type=String()),
                Field(name="third", type=String(), required=True),
                Field(name="fourth", type=String(), required=True),
            ]
        )
        assert to_json(node)["required"] == ["first", "third", "fourth"]


# ---------------------------------------------------------------------------
# Canonicalizations the serializer owns
# ---------------------------------------------------------------------------


class TestCanonicalizations:
    def test_true_false_enum_becomes_boolean(self) -> None:
        assert to_json(Enum(values=[True, False], base_type="boolean")) == {
            "type": "boolean"
        }

    def test_false_true_order_still_boolean(self) -> None:
        assert to_json(Enum(values=[False, True])) == {"type": "boolean"}

    def test_zero_one_integer_enum_is_not_boolean(self) -> None:
        # {0, 1} must NOT be mistaken for {True, False} despite Python's 0==False.
        assert to_json(Enum(values=[0, 1], base_type="integer")) == {
            "type": "integer",
            "enum": [0, 1],
        }

    def test_object_always_closed(self) -> None:
        assert to_json(Object())["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Document envelope
# ---------------------------------------------------------------------------


class TestDocument:
    def test_document_serializes_to_a_list_in_declaration_order(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(name="alpha", fields=[Field(name="x", type=String())]),
                ToolBlock(name="beta", fields=[Field(name="y", type=String())]),
                ToolBlock(name="gamma", fields=[Field(name="z", type=String())]),
            ]
        )
        out = to_json(doc)
        assert isinstance(out, list)
        assert len(out) == 3
        assert [tool["name"] for tool in out] == ["alpha", "beta", "gamma"]

    def test_name_is_first_key_in_a_tool_schema(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    description="d",
                    fields=[Field(name="x", type=String(), required=True)],
                )
            ]
        )
        assert list(to_json(doc)[0].keys()) == [
            "name",
            "description",
            "type",
            "properties",
            "required",
            "additionalProperties",
        ]

    def test_tool_embeds_referenced_definition_in_its_own_defs(self) -> None:
        doc = Document(
            defs=[Definition(name="Address", fields=[Field(name="street", type=String(), required=True)])],
            tools=[
                ToolBlock(
                    name="get-user",
                    description="Look up a user",
                    fields=[Field(name="home", type=Reference(name="Address"))],
                )
            ],
        )
        assert to_json(doc) == [
            {
                "name": "get-user",
                "description": "Look up a user",
                "$defs": {
                    "Address": {
                        "type": "object",
                        "properties": {"street": {"type": "string"}},
                        "required": ["street"],
                        "additionalProperties": False,
                    }
                },
                "type": "object",
                "properties": {"home": {"$ref": "#/$defs/Address"}},
                "additionalProperties": False,
            }
        ]

    def test_embedded_defs_emitted_last(self) -> None:
        doc = Document(
            defs=[Definition(name="A", fields=[Field(name="x", type=String())])],
            tools=[
                ToolBlock(
                    name="t",
                    description="d",
                    fields=[Field(name="a", type=Reference(name="A"))],
                )
            ],
        )
        assert list(to_json(doc)[0].keys()) == [
            "name",
            "description",
            "$defs",
            "type",
            "properties",
            "additionalProperties",
        ]

    def test_tool_envelope_key_order_name_description_defs_body(self) -> None:
        doc = Document(
            defs=[Definition(name="Addr", fields=[Field(name="s", type=String(), required=True)])],
            tools=[
                ToolBlock(
                    name="get-user",
                    description="Look up",
                    fields=[Field(name="home", type=Reference(name="Addr"))],
                )
            ],
        )
        assert list(to_json(doc)[0].keys()) == [
            "name",
            "description",
            "$defs",
            "type",
            "properties",
            "additionalProperties",
        ]

    def test_fallback_tool_json_is_byte_identical(self) -> None:
        raw = {
            "z": 1,
            "name": "fallback-first",
            "type": "object",
            "additionalProperties": False,
        }
        doc = Document(tools=[RawSchema(schema=raw)])
        assert to_json(doc)[0] == raw

    def test_tool_referencing_nothing_emits_no_defs(self) -> None:
        doc = Document(
            defs=[Definition(name="Unused", fields=[Field(name="x", type=String())])],
            tools=[ToolBlock(name="t", fields=[Field(name="y", type=String())])],
        )
        assert "$defs" not in to_json(doc)[0]

    def test_only_referenced_definitions_are_embedded(self) -> None:
        # Three definitions, the tool uses exactly one -> only that one embeds.
        doc = Document(
            defs=[
                Definition(name="A", fields=[Field(name="x", type=String())]),
                Definition(name="B", fields=[Field(name="x", type=String())]),
                Definition(name="C", fields=[Field(name="x", type=String())]),
            ],
            tools=[ToolBlock(name="t", fields=[Field(name="b", type=Reference(name="B"))])],
        )
        assert set(to_json(doc)[0]["$defs"].keys()) == {"B"}

    def test_transitive_references_are_embedded(self) -> None:
        # tool -> A -> B must embed BOTH A and B.
        doc = Document(
            defs=[
                Definition(name="A", fields=[Field(name="b", type=Reference(name="B"))]),
                Definition(name="B", fields=[Field(name="x", type=String())]),
            ],
            tools=[ToolBlock(name="t", fields=[Field(name="a", type=Reference(name="A"))])],
        )
        assert set(to_json(doc)[0]["$defs"].keys()) == {"A", "B"}

    def test_reference_inside_array_and_union_is_followed(self) -> None:
        doc = Document(
            defs=[
                Definition(name="A", fields=[Field(name="x", type=String())]),
                Definition(name="B", fields=[Field(name="x", type=String())]),
            ],
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        Field(name="xs", type=Array(element=Reference(name="A"))),
                        Field(name="u", type=Union(branches=[Reference(name="B"), Null()])),
                    ],
                )
            ],
        )
        assert set(to_json(doc)[0]["$defs"].keys()) == {"A", "B"}

    def test_reference_cycle_terminates_and_embeds_both(self) -> None:
        # A -> B -> A: must not loop forever, and embeds both A and B.
        doc = Document(
            defs=[
                Definition(name="A", fields=[Field(name="b", type=Reference(name="B"))]),
                Definition(name="B", fields=[Field(name="a", type=Reference(name="A"))]),
            ],
            tools=[ToolBlock(name="t", fields=[Field(name="a", type=Reference(name="A"))])],
        )
        assert set(to_json(doc)[0]["$defs"].keys()) == {"A", "B"}

    def test_embedded_defs_follow_document_declaration_order(self) -> None:
        doc = Document(
            defs=[
                Definition(name="A", fields=[Field(name="x", type=String())]),
                Definition(name="B", fields=[Field(name="x", type=String())]),
            ],
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        Field(name="b", type=Reference(name="B")),
                        Field(name="a", type=Reference(name="A")),
                    ],
                )
            ],
        )
        # Embedding order follows the document's $defs declaration order (A, B),
        # not the order the references were encountered (B, A).
        assert list(to_json(doc)[0]["$defs"].keys()) == ["A", "B"]

    def test_single_toolblock_serializes_to_one_dict_not_a_list(self) -> None:
        # The list is only the Document wrapper; a lone ToolBlock is one dict.
        tool = ToolBlock(name="solo", fields=[Field(name="x", type=String())])
        out = to_json(tool)
        assert isinstance(out, dict)
        assert out["name"] == "solo"

    def test_each_list_item_validates_standalone(self) -> None:
        # The real usage pattern: pull ONE tool's schema out, with no wrapper,
        # and its embedded $defs must make the $ref resolve.
        doc = Document(
            defs=[Definition(name="Address", fields=[Field(name="city", type=String(), required=True)])],
            tools=[
                ToolBlock(
                    name="get-user",
                    fields=[Field(name="home", type=Reference(name="Address"), required=True)],
                ),
                ToolBlock(name="ping", fields=[Field(name="msg", type=String())]),
            ],
        )
        out = to_json(doc)
        assert len(out) == 2
        for tool_schema in out:
            assert "name" in tool_schema
            Draft202012Validator.check_schema(tool_schema)  # each is a legal schema

        get_user = out[0]
        validator = Draft202012Validator(get_user)
        validator.validate({"home": {"city": "Paris"}})  # $ref resolves standalone
        assert not validator.is_valid({"home": {}})  # city required inside Address


class TestCountDefinitionInlinings:
    def test_two_tools_sharing_one_definition_counts_two(self) -> None:
        doc = Document(
            defs=[Definition(name="Shared", fields=[Field(name="x", type=String())])],
            tools=[
                ToolBlock(name="t1", fields=[Field(name="a", type=Reference(name="Shared"))]),
                ToolBlock(name="t2", fields=[Field(name="b", type=Reference(name="Shared"))]),
            ],
        )
        assert count_definition_inlinings(doc) == 2

    def test_unreferenced_definitions_count_zero(self) -> None:
        doc = Document(
            defs=[Definition(name="Unused", fields=[Field(name="x", type=String())])],
            tools=[ToolBlock(name="t", fields=[Field(name="y", type=String())])],
        )
        assert count_definition_inlinings(doc) == 0

    def test_transitive_inlinings_counted_per_tool(self) -> None:
        # One tool, A -> B: two definitions embedded -> two inlinings.
        doc = Document(
            defs=[
                Definition(name="A", fields=[Field(name="b", type=Reference(name="B"))]),
                Definition(name="B", fields=[Field(name="x", type=String())]),
            ],
            tools=[ToolBlock(name="t", fields=[Field(name="a", type=Reference(name="A"))])],
        )
        assert count_definition_inlinings(doc) == 2


# ---------------------------------------------------------------------------
# JSON-legality and string serialization
# ---------------------------------------------------------------------------


class TestJsonLegality:
    def test_dumps_loads_round_trip_preserves_structure(self) -> None:
        doc = parse_text(
            "$defs\n"
            "  Address\n"
            "    street* string:length[1,20]\n"
            "get-user # look up a user\n"
            "  id* integer[1,100)\n"
            "  home $Address\n"
            "  tags array<string>[1,]:unique =[]\n"
            "  status active|archived\n"
        )
        out = to_json(doc)
        assert json.loads(json.dumps(out)) == out

    def test_to_json_string_is_valid_json_and_keeps_order(self) -> None:
        node = String(format="email", min_length=1)
        text = to_json_string(node)
        assert json.loads(text) == {"type": "string", "format": "email", "minLength": 1}
        # indent=2, insertion order kept: "type" appears before "format".
        assert text.index('"type"') < text.index('"format"') < text.index('"minLength"')


# ---------------------------------------------------------------------------
# Draft 2020-12 meta-validation via the jsonschema library
# ---------------------------------------------------------------------------


class TestDraft202012MetaValidation:
    def test_constrained_object_is_a_valid_2020_12_schema(self) -> None:
        node = Object(
            fields=[
                Field(name="email", type=String(format="email", min_length=3), required=True),
                Field(name="age", type=Integer(minimum=0, maximum=120)),
                Field(name="tags", type=Array(element=String(), min_items=1, unique=True)),
                Field(name="role", type=Enum(values=["admin", "user"], base_type="string")),
            ]
        )
        schema = to_json(node)
        Draft202012Validator.check_schema(schema)  # raises if malformed

    def test_self_contained_ref_schema_validates_instances(self) -> None:
        # The serializer's own per-tool output must carry its $defs so a $ref
        # resolves with no document wrapper. Check schema legality + behavior.
        doc = Document(
            defs=[Definition(name="Address", fields=[Field(name="city", type=String(), required=True)])],
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="home", type=Reference(name="Address"), required=True)],
                )
            ],
        )
        schema = to_json(doc)[0]
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate({"home": {"city": "Paris"}})
        assert not validator.is_valid({"home": {}})  # city is required
        assert not validator.is_valid({"home": {"city": "Paris"}, "extra": 1})  # closed

    def test_union_and_enum_schemas_are_valid(self) -> None:
        Draft202012Validator.check_schema(to_json(Union(branches=[String(), Integer(), Null()])))
        Draft202012Validator.check_schema(to_json(Enum(values=[1, 2, 3], base_type="integer")))

    def test_parsed_tool_parameter_schema_is_valid(self) -> None:
        doc = parse_text(
            "create-event\n"
            "  title* string:length[1,100]\n"
            "  attendees array<string>[1,]:unique\n"
            "  priority 1|2|3\n"
            "  recurring boolean =false\n"
        )
        tool_schema = to_json(doc)[0]
        Draft202012Validator.check_schema(tool_schema)
        validator = Draft202012Validator(tool_schema)
        validator.validate({"title": "Standup", "attendees": ["a@b.com"], "priority": 2})
