"""Unit tests for the CATS parser (cats-converter/parser.py).

The parser turns the lexer's flat token list into a `Document` AST. These tests
follow the suite asked for:
  - a full tool block with a nested object
  - a `$defs` block plus a `$`-reference to it
  - each type form (array / union / enum / const / reference / nullable)
  - the annotation chain landing in the right NAMED fields (§6)
  - default + description parsing, including a quoted description (§4.5/§4.6)
  - and — importantly — that the parser BUILDS a tree for grammatically-derivable
    but semantically-illegal input (numeric bound on string, required + default,
    mixed union, out-of-order annotations) rather than rejecting it.

Everything goes through the real lexer (no hand-built token lists) so the tests
also pin the lexer/parser seam.
"""

from __future__ import annotations

import pytest

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
    Reference,
    String,
    ToolBlock,
    Union,
)
from parser import ParseError, parse_text


def only_tool(text: str) -> ToolBlock:
    """Parse `text` and return its single tool block (a common shape here)."""
    doc = parse_text(text)
    assert isinstance(doc, Document)
    assert len(doc.tools) == 1
    return doc.tools[0]


def one_field_type(field_line: str):
    """Wrap a single field line in a tool and return that field's parsed type.

    `field_line` is the body line without indentation, e.g. `x array<string>`.
    """
    tool = only_tool("tool\n  " + field_line + "\n")
    assert len(tool.fields) == 1
    return tool.fields[0].type


# ---------------------------------------------------------------------------
# Document structure (§3)
# ---------------------------------------------------------------------------


class TestDocumentStructure:
    def test_full_tool_block_with_nested_object(self) -> None:
        text = (
            "get-weather # Look up the weather\n"
            "  location* object\n"
            "    city* string\n"
            "    zip string\n"
            "  units string|null\n"
        )
        tool = only_tool(text)

        assert tool.name == "get-weather"
        assert tool.description == "Look up the weather"
        assert [f.name for f in tool.fields] == ["location", "units"]

        location = tool.fields[0]
        assert location.required is True
        assert isinstance(location.type, Object)
        assert [f.name for f in location.type.fields] == ["city", "zip"]
        assert location.type.fields[0].required is True
        assert location.type.fields[1].required is False

        units = tool.fields[1]
        assert isinstance(units.type, Union)
        assert [type(b) for b in units.type.branches] == [String, Null]

    def test_defs_block_and_reference(self) -> None:
        text = (
            "$defs\n"
            "  Address\n"
            "    street* string\n"
            "    city string\n"
            "get-user\n"
            "  home $Address\n"
        )
        doc = parse_text(text)

        assert len(doc.defs) == 1
        address = doc.defs[0]
        assert isinstance(address, Definition)
        assert address.name == "Address"
        assert [f.name for f in address.fields] == ["street", "city"]

        assert len(doc.tools) == 1
        home = doc.tools[0].fields[0]
        assert isinstance(home.type, Reference)
        assert home.type.name == "Address"  # leading '$' dropped

    def test_multiple_tool_blocks(self) -> None:
        doc = parse_text("a\n  x string\nb\n  y integer\n")
        assert [t.name for t in doc.tools] == ["a", "b"]

    def test_quoted_tool_name(self) -> None:
        tool = only_tool('"my.tool"\n  x string\n')
        assert tool.name == "my.tool"  # quotes stripped by the lexer

    def test_empty_document_builds_permissively(self) -> None:
        # Zero tool blocks violates §3.1, but that is validate.py's call.
        doc = parse_text("")
        assert doc == Document(tools=[], defs=None)

    def test_parameterless_tool_builds(self) -> None:
        # A header with no indented body -> empty field list, not an error.
        tool = only_tool("ping\n")
        assert tool.fields == []


# ---------------------------------------------------------------------------
# Type forms (§5)
# ---------------------------------------------------------------------------


class TestTypeForms:
    def test_each_primitive_word(self) -> None:
        assert isinstance(one_field_type("a string"), String)
        assert isinstance(one_field_type("a integer"), Integer)
        assert isinstance(one_field_type("a number"), Number)
        assert isinstance(one_field_type("a boolean"), Boolean)
        assert isinstance(one_field_type("a null"), Null)
        assert isinstance(one_field_type("a any"), AnyType)

    def test_integer_and_number_stay_distinct(self) -> None:
        assert type(one_field_type("a integer")) is Integer
        assert type(one_field_type("a number")) is Number

    def test_parameterized_array(self) -> None:
        t = one_field_type("a array<string>")
        assert isinstance(t, Array)
        assert isinstance(t.element, String)

    def test_bare_array_has_no_element(self) -> None:
        t = one_field_type("a array")
        assert isinstance(t, Array)
        assert t.element is None

    def test_nested_array(self) -> None:
        t = one_field_type("a array<array<integer>>")
        assert isinstance(t, Array)
        assert isinstance(t.element, Array)
        assert isinstance(t.element.element, Integer)

    def test_array_of_reference(self) -> None:
        t = one_field_type("a array<$Person>")
        assert isinstance(t, Array)
        assert isinstance(t.element, Reference)
        assert t.element.name == "Person"

    def test_type_union(self) -> None:
        t = one_field_type("a string|integer")
        assert isinstance(t, Union)
        assert [type(b) for b in t.branches] == [String, Integer]

    def test_nullable_union(self) -> None:
        t = one_field_type("a string|null")
        assert isinstance(t, Union)
        assert [type(b) for b in t.branches] == [String, Null]

    def test_string_enum(self) -> None:
        t = one_field_type("a public|private|default")
        assert isinstance(t, Enum)
        assert t.values == ["public", "private", "default"]
        assert t.base_type == "string"

    def test_integer_enum(self) -> None:
        t = one_field_type("a 1|2|3|5")
        assert isinstance(t, Enum)
        assert t.values == [1, 2, 3, 5]
        assert t.base_type == "integer"

    def test_number_enum(self) -> None:
        t = one_field_type("a -0.5|0|0.5")
        assert isinstance(t, Enum)
        assert t.values == [-0.5, 0, 0.5]
        assert t.base_type == "number"

    def test_single_value_enum_quoted(self) -> None:
        t = one_field_type('a "automatic"')
        assert isinstance(t, Const)
        assert t.value == "automatic"

    def test_single_value_enum_bare(self) -> None:
        t = one_field_type("a active")
        assert isinstance(t, Const)
        assert t.value == "active"

    def test_single_value_enum_number(self) -> None:
        t = one_field_type("a 200")
        assert isinstance(t, Const)
        assert t.value == 200

    def test_boolean_single_value_enum(self) -> None:
        # `true` is not a type word, so a lone `true` is a boolean VALUE -> Const.
        t = one_field_type("a true")
        assert isinstance(t, Const)
        assert t.value is True

    def test_quoted_value_colliding_with_type_word(self) -> None:
        # "string" quoted is a value (§2.6), not the string type.
        t = one_field_type('a "string"')
        assert isinstance(t, Const)
        assert t.value == "string"

    def test_reference_in_union(self) -> None:
        t = one_field_type("a $User|$Guest")
        assert isinstance(t, Union)
        assert [type(b) for b in t.branches] == [Reference, Reference]
        assert [b.name for b in t.branches] == ["User", "Guest"]


# ---------------------------------------------------------------------------
# Annotation chain -> named fields (§6)
# ---------------------------------------------------------------------------


class TestAnnotations:
    def test_string_format(self) -> None:
        t = one_field_type("a string:date-time")
        assert isinstance(t, String)
        assert t.format == "date-time"

    def test_numeric_format_passthrough(self) -> None:
        t = one_field_type("a integer:int64")
        assert isinstance(t, Integer)
        assert t.format == "int64"

    def test_string_length_regex_encoding_media(self) -> None:
        t = one_field_type(
            'a string:length[1,20]:regex["^[a-z]+$"]:encoding[base64]:media["application/pdf"]'
        )
        assert isinstance(t, String)
        assert t.min_length == 1
        assert t.max_length == 20
        assert t.pattern == "^[a-z]+$"
        assert t.encoding == "base64"
        assert t.media == "application/pdf"

    def test_inclusive_numeric_bounds(self) -> None:
        t = one_field_type("a integer[1,100]")
        assert isinstance(t, Integer)
        assert (t.minimum, t.maximum) == (1, 100)
        assert t.exclusive_min is False
        assert t.exclusive_max is False

    def test_exclusive_numeric_bounds(self) -> None:
        t = one_field_type("a number(0,1)")
        assert isinstance(t, Number)
        assert (t.minimum, t.maximum) == (0, 1)
        assert t.exclusive_min is True
        assert t.exclusive_max is True

    def test_mixed_bracket_bounds(self) -> None:
        t = one_field_type("a integer[0,100)")
        assert (t.minimum, t.maximum) == (0, 100)
        assert t.exclusive_min is False
        assert t.exclusive_max is True

    def test_open_upper_bound_with_multiple_of(self) -> None:
        t = one_field_type("a integer[1,)%5")
        assert t.minimum == 1
        assert t.maximum is None
        assert t.multiple_of == 5

    def test_open_lower_bound(self) -> None:
        t = one_field_type("a integer[,100]")
        assert t.minimum is None
        assert t.maximum == 100

    def test_multiple_of_without_bounds(self) -> None:
        t = one_field_type("a number%0.01")
        assert isinstance(t, Number)
        assert t.multiple_of == 0.01

    def test_array_bounds(self) -> None:
        t = one_field_type("a array<string>[1,10]")
        assert isinstance(t, Array)
        assert t.min_items == 1
        assert t.max_items == 10

    def test_array_bounds_and_unique(self) -> None:
        t = one_field_type("a array<string>[1,]:unique")
        assert t.min_items == 1
        assert t.max_items is None
        assert t.unique is True

    def test_array_unique_only(self) -> None:
        t = one_field_type("a array<string>:unique")
        assert t.unique is True

    def test_annotation_binds_to_union_branch(self) -> None:
        # `count integer[0,100]|null` bounds only the integer branch (§4.4/§5.5).
        t = one_field_type("count integer[0,100]|null")
        assert isinstance(t, Union)
        integer_branch, null_branch = t.branches
        assert isinstance(integer_branch, Integer)
        assert (integer_branch.minimum, integer_branch.maximum) == (0, 100)
        assert isinstance(null_branch, Null)


# ---------------------------------------------------------------------------
# Defaults and descriptions (§4.5, §4.6)
# ---------------------------------------------------------------------------


class TestDefaultsAndDescriptions:
    def test_no_default_uses_sentinel(self) -> None:
        field = only_tool("tool\n  x string\n").fields[0]
        assert field.default is NO_DEFAULT
        assert field.description is None

    def test_number_default_and_description(self) -> None:
        field = only_tool("tool\n  count integer =5 # how many\n").fields[0]
        assert field.default == 5
        assert field.description == "how many"

    def test_string_default(self) -> None:
        field = only_tool('tool\n  name string ="bob"\n').fields[0]
        assert field.default == "bob"

    def test_boolean_default(self) -> None:
        field = only_tool("tool\n  flag boolean =true\n").fields[0]
        assert field.default is True

    def test_null_default_is_none_not_sentinel(self) -> None:
        # `=null` is the null VALUE (None), distinct from "no default" (§4.5).
        field = only_tool("tool\n  opt string =null\n").fields[0]
        assert field.default is None
        assert field.default is not NO_DEFAULT

    def test_object_json_default(self) -> None:
        field = only_tool("tool\n  cfg object ={}\n").fields[0]
        assert field.default == {}

    def test_array_json_default(self) -> None:
        field = only_tool("tool\n  xs array<integer> =[1,2,3]\n").fields[0]
        assert field.default == [1, 2, 3]

    def test_bare_enum_member_default(self) -> None:
        field = only_tool("tool\n  vis public|private =public\n").fields[0]
        assert isinstance(field.type, Enum)
        assert field.default == "public"

    def test_quoted_description_strips_quotes_without_decoding(self) -> None:
        # The lexer hands the description over raw, quotes included; the parser
        # strips the outer pair and does NOT decode escapes (§4.6 contract).
        field = only_tool('tool\n  w string # "weight in kg (#2 priority)"\n').fields[0]
        assert field.description == "weight in kg (#2 priority)"

    def test_quoted_description_keeps_backslash(self) -> None:
        field = only_tool('tool\n  p string # "path C:\\temp #1"\n').fields[0]
        assert field.description == "path C:\\temp #1"

    def test_description_on_definition_header(self) -> None:
        doc = parse_text("$defs\n  Foo # a shape\n    x integer\nt\n  y $Foo\n")
        assert doc.defs[0].description == "a shape"


# ---------------------------------------------------------------------------
# §7.5 raw JSON Schema tools vs CATS tool blocks
# ---------------------------------------------------------------------------


class TestRawJsonToolBlocks:
    def test_raw_json_object_parses_as_rawschema(self) -> None:
        raw = '{"name":"t","type":"object","properties":{"q":{"not":{"type":"string"}}},"additionalProperties":false}'
        doc = parse_text(raw)
        assert len(doc.tools) == 1
        from nodes import RawSchema

        assert isinstance(doc.tools[0], RawSchema)
        assert doc.tools[0].schema["name"] == "t"

    def test_mixed_cats_and_raw_tools_in_one_document(self) -> None:
        text = (
            "good\n"
            "  x string\n"
            "\n"
            '{"name":"bad","type":"object","properties":{"q":{"not":{"type":"string"}}},'
            '"additionalProperties":false}'
        )
        doc = parse_text(text)
        from nodes import RawSchema, ToolBlock

        assert len(doc.tools) == 2
        assert isinstance(doc.tools[0], ToolBlock)
        assert doc.tools[0].name == "good"
        assert isinstance(doc.tools[1], RawSchema)

    def test_brace_at_column_zero_is_raw_json_not_cats_tool_block(self) -> None:
        """CATS tool blocks begin with a name; `{` marks a raw JSON tool (§7.5)."""
        from nodes import RawSchema, ToolBlock

        raw_doc = parse_text(
            '{"name":"t","type":"object","additionalProperties":false}'
        )
        assert isinstance(raw_doc.tools[0], RawSchema)
        cats_doc = parse_text("t\n  x string\n")
        assert isinstance(cats_doc.tools[0], ToolBlock)

    def test_raw_tool_with_brace_in_string_round_trips(self) -> None:
        from nodes import RawSchema

        raw_line = (
            '{"name":"t","description":"uses {curly} syntax","type":"object",'
            '"additionalProperties":false}'
        )
        doc = parse_text(raw_line)
        assert isinstance(doc.tools[0], RawSchema)
        assert doc.tools[0].schema["description"] == "uses {curly} syntax"

    def test_multiline_raw_tool_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_text('{"name":"t"}\n{"name":"u"}')

    def test_mixed_document_byte_round_trip(self) -> None:
        from nodes import RawSchema, ToolBlock
        from to_cats import to_cats

        raw_line = (
            '{"name":"bad","type":"object","properties":{"q":{"not":{"type":"string"}}},'
            '"additionalProperties":false}'
        )
        text = "good\n  x string\n\n" + raw_line
        doc = parse_text(text)
        assert isinstance(doc.tools[0], ToolBlock)
        assert isinstance(doc.tools[1], RawSchema)
        assert to_cats(doc) == text


# ---------------------------------------------------------------------------
# Permissive: build grammatical-but-illegal input, do NOT reject (the core rule)
# ---------------------------------------------------------------------------


class TestHomelessAnnotations:
    """§6.5 pairings with no legal base-type home must raise ParseError (not drop)."""

    @pytest.mark.parametrize(
        "field_line,annotation_fragment",
        [
            ("x string[0,100]", "numeric bound"),
            ("x string%5", "%divisor"),
            ("x integer:length[1,10]", ":length"),
            ("x number:regex[\"x\"]", ":regex"),
            ("x boolean:encoding[base64]", ":encoding"),
            ("x null:media[\"text/plain\"]", "annotation"),
            ("x integer:unique", ":unique"),
            ("x string:unique", ":unique"),
            ("x array<string>[1,10]:length[0,1]", ":length"),
            ("x object[0,1]", "numeric bound"),
            ("x $Foo[1,2]", "numeric bound"),
        ],
    )
    def test_illegal_annotation_raises(
        self, field_line: str, annotation_fragment: str
    ) -> None:
        with pytest.raises(ParseError) as exc:
            parse_text("tool\n  " + field_line + "\n")
        assert annotation_fragment in exc.value.message
        assert "§6.5" in exc.value.message

    def test_numeric_bound_on_string_error_location(self) -> None:
        with pytest.raises(ParseError) as exc:
            parse_text("tool\n  x string[0,100]\n")
        assert exc.value.line == 2
        assert exc.value.col == 11  # column of '['


class TestFormatOnEveryBaseType:
    """§6.1: :format must land on every base type, including array/object/reference."""

    @pytest.mark.parametrize(
        "field_line,node_type,format_value",
        [
            ("a string:email", String, "email"),
            ("a integer:int64", Integer, "int64"),
            ("a number:double", Number, "double"),
            ("a boolean:flag", Boolean, "flag"),
            ("a array<string>:list", Array, "list"),
            ("a object:rec", Object, "rec"),
        ],
    )
    def test_format_on_primitive_and_array_object(
        self, field_line: str, node_type: type, format_value: str
    ) -> None:
        t = one_field_type(field_line)
        assert isinstance(t, node_type)
        assert t.format == format_value

    def test_format_on_reference(self) -> None:
        t = one_field_type("a $Person:card")
        assert isinstance(t, Reference)
        assert t.format == "card"


class TestNullAndAnyForbidAnnotations:
    """§5.2 / §6.5: `null` and `any` admit no annotation chain."""

    @pytest.mark.parametrize(
        "field_line",
        [
            "x null:date-time",
            "x any:email",
            "x any[0,10]",
        ],
    )
    def test_annotation_on_null_or_any_raises(self, field_line: str) -> None:
        with pytest.raises(ParseError) as exc:
            parse_text("tool\n  " + field_line + "\n")
        assert "§6.5" in exc.value.message

    def test_bare_null_and_any_still_parse(self) -> None:
        assert isinstance(one_field_type("x null"), Null)
        assert isinstance(one_field_type("x any"), AnyType)


class TestDefsHeaderDescription:
    def test_defs_header_description_raises(self) -> None:
        with pytest.raises(ParseError) as exc:
            parse_text("$defs # shared shapes\n  Foo\n    x string\n")
        assert "§3.2" in exc.value.message


class TestPermissiveBuildsIllegal:
    def test_required_and_default_coexist(self) -> None:
        # §4.1 forbids both; the parser represents both and lets validate object.
        field = only_tool("tool\n  x* string =5\n").fields[0]
        assert field.required is True
        assert field.default == 5

    def test_mixed_type_value_union_builds(self) -> None:
        # §5.6 forbids mixing a type with a value. Built as a Union carrying a
        # Const branch, so validate.py can spot the mix.
        t = one_field_type("a string|published")
        assert isinstance(t, Union)
        assert isinstance(t.branches[0], String)
        assert isinstance(t.branches[1], Const)
        assert t.branches[1].value == "published"

    def test_out_of_order_annotations_build(self) -> None:
        # §6.5 fixes annotation order; out of order still parses (order is not
        # retained once it lands in named fields).
        t = one_field_type("a integer%5[1,10]")
        assert isinstance(t, Integer)
        assert t.minimum == 1
        assert t.maximum == 10
        assert t.multiple_of == 5


# ---------------------------------------------------------------------------
# Structural errors that SHOULD raise (syntax, not semantics)
# ---------------------------------------------------------------------------


class TestStructuralErrors:
    def test_field_with_no_type(self) -> None:
        with pytest.raises(ParseError):
            parse_text("tool\n  x\n")

    def test_unterminated_array(self) -> None:
        with pytest.raises(ParseError):
            parse_text("tool\n  x array<string\n")

    def test_default_with_no_value(self) -> None:
        with pytest.raises(ParseError):
            parse_text("tool\n  x string =\n")

    def test_nested_block_under_scalar(self) -> None:
        # An INDENT where no block can open: a string field cannot hold fields.
        with pytest.raises(ParseError):
            parse_text("tool\n  x string\n    y integer\n")

    def test_bounds_missing_comma(self) -> None:
        with pytest.raises(ParseError):
            parse_text("tool\n  x integer[1]\n")

    def test_dangling_pipe(self) -> None:
        with pytest.raises(ParseError):
            parse_text("tool\n  x string|\n")

    def test_error_carries_location(self) -> None:
        with pytest.raises(ParseError) as exc:
            parse_text("tool\n  x\n")
        assert exc.value.line == 2


# ---------------------------------------------------------------------------
# array<object> nested block (§4.7)
# ---------------------------------------------------------------------------


class TestArrayOfObjects:
    def test_array_of_object_attaches_nested_block(self) -> None:
        text = (
            "tool\n"
            "  people array<object>\n"
            "    name* string\n"
            "    age integer\n"
        )
        field = only_tool(text).fields[0]
        assert isinstance(field.type, Array)
        assert isinstance(field.type.element, Object)
        assert [f.name for f in field.type.element.fields] == ["name", "age"]
