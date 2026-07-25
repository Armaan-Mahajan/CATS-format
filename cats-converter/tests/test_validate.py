"""Unit tests for the CATS semantic validator (cats-converter/validate.py).

validate(document) walks a finished AST and returns a list[ValidationError];
an empty list means the tree is legal. These tests cover:
  - one violation per rule (1-8), each yielding exactly that error
  - a clean tree yielding no errors
  - collect-all: three distinct violations returned together in one call
  - the no-mutation guard (the tree is identical after validation)
  - the carve-outs that must NOT error (true|false set, out-of-order chain)

Trees are built directly from the node classes so each test targets one rule
without depending on parser behavior; a couple of tests round-trip through the
real parser to pin the parser/validator seam.
"""

from __future__ import annotations

import copy

from nodes import (
    NO_DEFAULT,
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
from parser import parse_text
from validate import ValidationError, ValidationWarning, validate, validate_with_warnings


def sections(errors: list[ValidationError]) -> list[str]:
    return [e.section for e in errors]


def clean_document() -> Document:
    """A fully legal document touching several rules' happy paths."""
    return Document(
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
                    Field(name="id", type=Integer(minimum=1, maximum=100), required=True),
                    Field(name="home", type=Reference(name="Address")),
                    Field(name="tags", type=Array(element=String(), min_items=0, max_items=5)),
                    Field(name="status", type=Union(branches=[String(), Null()])),
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Clean tree
# ---------------------------------------------------------------------------


class TestCleanTree:
    def test_clean_document_has_no_errors(self) -> None:
        assert validate(clean_document()) == []

    def test_clean_document_has_no_errors_or_warnings(self) -> None:
        errors, warnings = validate_with_warnings(clean_document())
        assert errors == []
        assert warnings == []

    def test_clean_parsed_document_has_no_errors(self) -> None:
        doc = parse_text(
            "$defs\n"
            "  Address\n"
            "    street* string\n"
            "get-user\n"
            "  id* integer[1,100]\n"
            "  home $Address\n"
        )
        assert validate(doc) == []


# ---------------------------------------------------------------------------
# One test per rule
# ---------------------------------------------------------------------------


class TestRule1RequiredDefault:
    def test_required_with_default_is_error(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="x", type=String(), required=True, default="hi")],
                )
            ]
        )
        errors = validate(doc)
        assert sections(errors) == ["§4.1"]

    def test_required_with_null_default_is_error(self) -> None:
        # A literal null IS a real default (distinct from NO_DEFAULT), so it still
        # collides with the required marker.
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="x", type=String(), required=True, default=None)],
                )
            ]
        )
        assert sections(validate(doc)) == ["§4.1"]

    def test_required_without_default_is_clean(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="x", type=String(), required=True)],
                )
            ]
        )
        assert validate(doc) == []

    def test_optional_with_default_is_clean(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="x", type=String(), default="hi")],
                )
            ]
        )
        assert validate(doc) == []


class TestRule2ReferenceResolution:
    def test_dangling_reference_is_error(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(name="t", fields=[Field(name="a", type=Reference(name="Missing"))])
            ]
        )
        errors = validate(doc)
        assert sections(errors) == ["§5.7"]
        assert "Missing" in errors[0].message

    def test_resolved_reference_is_clean(self) -> None:
        doc = Document(
            defs=[Definition(name="Foo", fields=[Field(name="x", type=String())])],
            tools=[ToolBlock(name="t", fields=[Field(name="a", type=Reference(name="Foo"))])],
        )
        assert validate(doc) == []

    def test_reference_nested_in_array_and_union_is_checked(self) -> None:
        # A dangling reference must be found however deep it sits.
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        Field(name="a", type=Array(element=Reference(name="Gone"))),
                        Field(name="b", type=Union(branches=[Reference(name="AlsoGone"), Null()])),
                    ],
                )
            ]
        )
        errors = validate(doc)
        assert sections(errors) == ["§5.7", "§5.7"]


class TestRule3DocumentCardinality:
    def test_zero_tools_is_error(self) -> None:
        doc = Document(
            defs=[Definition(name="Foo", fields=[Field(name="x", type=String())])],
            tools=[],
        )
        assert sections(validate(doc)) == ["§3.1"]


class TestRule4DefsCardinality:
    def test_no_defs_block_is_not_flagged(self) -> None:
        # defs=None means no $defs header — distinct from present-but-empty [].
        doc = Document(
            tools=[ToolBlock(name="t", fields=[Field(name="x", type=String())])],
            defs=None,
        )
        assert validate(doc) == []

    def test_empty_defs_list_is_error(self) -> None:
        doc = Document(
            tools=[ToolBlock(name="t", fields=[Field(name="x", type=String())])],
            defs=[],
        )
        errors = validate(doc)
        assert sections(errors) == ["§3.2"]
        assert errors[0].message == "$defs block is present but contains no definitions"

    def test_parsed_empty_defs_block_is_error(self) -> None:
        doc = parse_text("$defs\nmy-tool\n  x string\n")
        assert doc.defs == []
        assert sections(validate(doc)) == ["§3.2"]


class TestRule5DefinitionName:
    def test_hyphenated_definition_name_is_error(self) -> None:
        doc = Document(
            defs=[Definition(name="my-def", fields=[Field(name="x", type=String())])],
            tools=[ToolBlock(name="t", fields=[Field(name="x", type=String())])],
        )
        assert sections(validate(doc)) == ["§3.2"]

    def test_dotted_definition_name_is_error(self) -> None:
        doc = Document(
            defs=[Definition(name="my.def", fields=[Field(name="x", type=String())])],
            tools=[ToolBlock(name="t", fields=[Field(name="x", type=String())])],
        )
        assert sections(validate(doc)) == ["§3.2"]

    def test_pascal_case_definition_name_is_clean(self) -> None:
        doc = Document(
            defs=[Definition(name="MyDef", fields=[Field(name="x", type=String())])],
            tools=[ToolBlock(name="t", fields=[Field(name="x", type=String())])],
        )
        assert validate(doc) == []


class TestRule6UnionHomogeneity:
    def test_mixed_type_value_union_is_error(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        Field(name="a", type=Union(branches=[String(), Const(value="published")]))
                    ],
                )
            ]
        )
        assert sections(validate(doc)) == ["§5.6"]

    def test_all_type_union_is_clean(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="a", type=Union(branches=[String(), Integer(), Null()]))],
                )
            ]
        )
        assert validate(doc) == []


class TestRule7EnumConsistency:
    def test_mixed_enum_is_error(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(name="t", fields=[Field(name="a", type=Enum(values=["x", 1]))])
            ]
        )
        assert sections(validate(doc)) == ["§5.6"]

    def test_string_enum_is_clean(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(name="t", fields=[Field(name="a", type=Enum(values=["a", "b", "c"]))])
            ]
        )
        assert validate(doc) == []

    def test_numeric_enum_mixing_int_and_float_is_clean(self) -> None:
        # int + float are both "numeric" (a number enum, §5.6) — not a mix.
        doc = Document(
            tools=[
                ToolBlock(name="t", fields=[Field(name="a", type=Enum(values=[-0.5, 0, 0.5]))])
            ]
        )
        assert validate(doc) == []


class TestRule8Bounds:
    def test_numeric_lower_exceeds_upper_is_error(self) -> None:
        doc = Document(
            tools=[ToolBlock(name="t", fields=[Field(name="a", type=Integer(minimum=10, maximum=1))])]
        )
        assert sections(validate(doc)) == ["§6.2"]

    def test_string_length_lower_exceeds_upper_is_error(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(name="t", fields=[Field(name="a", type=String(min_length=5, max_length=2))])
            ]
        )
        assert sections(validate(doc)) == ["§6.3"]

    def test_array_items_lower_exceeds_upper_is_error(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="a", type=Array(element=String(), min_items=9, max_items=3))],
                )
            ]
        )
        assert sections(validate(doc)) == ["§6.4"]

    def test_equal_bounds_are_clean(self) -> None:
        doc = Document(
            tools=[ToolBlock(name="t", fields=[Field(name="a", type=Integer(minimum=5, maximum=5))])]
        )
        assert validate(doc) == []

    def test_open_bound_imposes_no_ordering(self) -> None:
        doc = Document(
            tools=[ToolBlock(name="t", fields=[Field(name="a", type=Integer(minimum=10, maximum=None))])]
        )
        assert validate(doc) == []


# ---------------------------------------------------------------------------
# Collect-all
# ---------------------------------------------------------------------------


class TestCollectAll:
    def test_three_distinct_violations_returned_together(self) -> None:
        doc = Document(
            # bad definition name (rule 5)
            defs=[Definition(name="bad-name", fields=[Field(name="x", type=String())])],
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        # required + default (rule 1)
                        Field(name="a", type=String(), required=True, default="x"),
                        # dangling reference (rule 2)
                        Field(name="b", type=Reference(name="Nope")),
                    ],
                )
            ],
        )
        errors = validate(doc)
        assert len(errors) == 3
        assert set(sections(errors)) == {"§3.2", "§4.1", "§5.7"}

    def test_multiple_bound_violations_all_reported(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        Field(name="a", type=Integer(minimum=10, maximum=1)),
                        Field(name="b", type=String(min_length=5, max_length=2)),
                    ],
                )
            ]
        )
        assert len(validate(doc)) == 2


# ---------------------------------------------------------------------------
# No-mutation guard
# ---------------------------------------------------------------------------


class TestNoMutation:
    def test_validation_does_not_change_the_tree(self) -> None:
        doc = clean_document()
        before = copy.deepcopy(doc)
        validate(doc)
        assert doc == before

    def test_true_false_set_is_not_rewritten_to_boolean(self) -> None:
        # The tree represents true|false as an Enum of two boolean values. The
        # serializer canonicalizes that to boolean later; the VALIDATOR must
        # leave it exactly as it found it (and must not error on it).
        enum = Enum(values=[True, False], base_type="boolean")
        doc = Document(tools=[ToolBlock(name="t", fields=[Field(name="flag", type=enum)])])
        before = copy.deepcopy(doc)

        errors = validate(doc)

        assert errors == []
        assert doc == before
        still = doc.tools[0].fields[0].type
        assert isinstance(still, Enum)
        assert still.values == [True, False]


# ---------------------------------------------------------------------------
# Carve-outs that must NOT error
# ---------------------------------------------------------------------------


class TestCarveOuts:
    def test_true_false_enum_is_not_an_error(self) -> None:
        doc = Document(
            tools=[ToolBlock(name="t", fields=[Field(name="a", type=Enum(values=[True, False]))])]
        )
        assert validate(doc) == []

    def test_out_of_order_annotation_chain_is_not_an_error(self) -> None:
        # Order is not retained in the tree; an integer carrying bounds + %div in
        # any source order is a plain valid Integer node here.
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="a", type=Integer(minimum=1, maximum=10, multiple_of=5))],
                )
            ]
        )
        assert validate(doc) == []

    def test_bare_object_is_not_an_error(self) -> None:
        # Open-vs-closed object is a serializer concern, not a legality one.
        doc = Document(tools=[ToolBlock(name="t", fields=[Field(name="a", type=Object())])])
        assert validate(doc) == []


# ---------------------------------------------------------------------------
# Duplicate names (rules 9–10 + tool-name warning)
# ---------------------------------------------------------------------------


class TestDuplicateNames:
    def test_duplicate_field_names_in_tool_error(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        Field(name="x", type=String()),
                        Field(name="x", type=Integer()),
                    ],
                )
            ]
        )
        errors = validate(doc)
        assert len(errors) == 1
        assert errors[0].section == "§4.2"
        assert "duplicate field name 'x'" in errors[0].message

    def test_duplicate_field_names_in_nested_object_error(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        Field(
                            name="outer",
                            type=Object(
                                fields=[
                                    Field(name="a", type=String()),
                                    Field(name="a", type=Integer()),
                                ]
                            ),
                        )
                    ],
                )
            ]
        )
        errors = validate(doc)
        assert any(e.section == "§4.2" for e in errors)

    def test_duplicate_definition_names_error(self) -> None:
        doc = Document(
            defs=[
                Definition(name="Foo", fields=[Field(name="x", type=String())]),
                Definition(name="Foo", fields=[Field(name="y", type=Integer())]),
            ],
            tools=[ToolBlock(name="t", fields=[Field(name="a", type=String())])],
        )
        errors = validate(doc)
        assert len(errors) == 1
        assert errors[0].section == "§3.2"
        assert "duplicate definition name 'Foo'" in errors[0].message

    def test_duplicate_tool_names_warning(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(name="echo", fields=[Field(name="x", type=String())]),
                ToolBlock(name="echo", fields=[Field(name="y", type=Integer())]),
            ]
        )
        errors, warnings = validate_with_warnings(doc)
        assert errors == []
        assert len(warnings) == 1
        assert warnings[0].section == "§3.5"
        assert "duplicate tool name 'echo'" in warnings[0].message


# ---------------------------------------------------------------------------
# Unused $defs definitions (warnings, not errors)
# ---------------------------------------------------------------------------


class TestUnusedDefinitions:
    def test_unused_definition_yields_warning_not_error(self) -> None:
        doc = Document(
            defs=[
                Definition(name="Used", fields=[Field(name="x", type=String())]),
                Definition(name="Orphan", fields=[Field(name="y", type=Integer())]),
            ],
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="a", type=Reference(name="Used"))],
                )
            ],
        )
        errors, warnings = validate_with_warnings(doc)
        assert errors == []
        assert len(warnings) == 1
        assert "Orphan" in warnings[0].message
        assert warnings[0].section == "§3.2"

    def test_definition_used_by_a_tool_has_no_warning(self) -> None:
        doc = Document(
            defs=[Definition(name="Addr", fields=[Field(name="s", type=String())])],
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="home", type=Reference(name="Addr"))],
                )
            ],
        )
        errors, warnings = validate_with_warnings(doc)
        assert errors == []
        assert warnings == []

    def test_transitive_use_through_another_definition_has_no_warning(self) -> None:
        # Tool -> A; A's fields reference B only. Both A and B are used.
        doc = Document(
            defs=[
                Definition(
                    name="A",
                    fields=[Field(name="nested", type=Reference(name="B"))],
                ),
                Definition(name="B", fields=[Field(name="z", type=String())]),
            ],
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="root", type=Reference(name="A"))],
                )
            ],
        )
        errors, warnings = validate_with_warnings(doc)
        assert errors == []
        assert warnings == []
