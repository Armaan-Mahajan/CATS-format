"""Tests for primer.py — manifest detection and calibrated primer assembly."""

from __future__ import annotations

import copy

import pytest

import primer
from nodes import (
    Array,
    Const,
    Definition,
    Document,
    Enum,
    Field,
    Integer,
    Null,
    Object,
    RawSchema,
    Reference,
    String,
    ToolBlock,
    Union,
)
from parser import ParseError
from primer import (
    OUTPUT_CONTRACT,
    Manifest,
    PrimerResult,
    _CORE,
    _CORE_INTRO,
    _CORE_RULE_ALL_REQUIRED,
    _CORE_RULE_MIXED,
    _CLAUSE_BOUNDS_BASE,
    _CLAUSE_BOUNDS_BOTH_STYLES,
    _CLAUSE_BOUNDS_EXCL_ONLY,
    _CLAUSE_ENUM_MULTI,
    _CLAUSE_ENUM_SINGLE,
    _CLAUSE_FALLBACK,
    _CLAUSE_OPEN_BOUND,
    build_manifest,
    build_output_contract,
    build_system_prompt,
    generate_primer_from_cats,
    generate_primer_from_json,
)

CLEAN_TOOL = {
    "name": "echo",
    "type": "object",
    "properties": {"message": {"type": "string", "minLength": 1}},
    "required": ["message"],
    "additionalProperties": False,
}

TOOL_WITH_EXCLUSIVE_BOUNDS = {
    "name": "rate",
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "exclusiveMaximum": 100},
    },
    "required": ["score"],
    "additionalProperties": False,
}

TOOL_WITH_ENUM = {
    "name": "search",
    "type": "object",
    "properties": {
        "sort": {"type": "string", "enum": ["relevance", "price", "newest"]},
    },
    "required": ["sort"],
    "additionalProperties": False,
}

TOOL_WITH_NOT = {
    "name": "bad",
    "type": "object",
    "properties": {"q": {"not": {"type": "string"}}},
    "additionalProperties": False,
}


def _empty_manifest(**overrides: bool) -> Manifest:
    base = Manifest(
        bounds_inclusive=False,
        bounds_exclusive=False,
        bounds_open=False,
        has_multiple_of=False,
        has_string_length=False,
        string_length_open=False,
        has_regex=False,
        has_encoding_media=False,
        has_typed_array=False,
        has_array_bounds=False,
        has_unique=False,
        has_type_union=False,
        enum_multi=False,
        enum_single=False,
        has_defs_reference=False,
        has_parameterless_tool=False,
        has_default_value=False,
        required_uniformity="all_optional",
        has_fallback=False,
        all_fallback=False,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _doc_with_type(type_node, *, defs=None) -> Document:
    return Document(
        defs=defs,
        tools=[ToolBlock(name="t", fields=[Field(name="x", type=type_node)])],
    )


class TestBuildManifest:
    def test_plain_string_fields_all_false(self) -> None:
        doc = _doc_with_type(String())
        assert build_manifest(doc) == _empty_manifest()

    def test_string_pattern_sets_has_regex(self) -> None:
        doc = _doc_with_type(String(pattern=r"\d+"))
        assert build_manifest(doc) == _empty_manifest(has_regex=True)

    def test_integer_inclusive_bounds(self) -> None:
        doc = _doc_with_type(Integer(minimum=1, maximum=10))
        assert build_manifest(doc) == _empty_manifest(
            bounds_inclusive=True,
        )

    def test_integer_mixed_exclusive_upper(self) -> None:
        doc = _doc_with_type(Integer(minimum=0, maximum=100, exclusive_max=True))
        assert build_manifest(doc) == _empty_manifest(
            bounds_inclusive=True,
            bounds_exclusive=True,
        )

    def test_integer_open_lower_only(self) -> None:
        doc = _doc_with_type(Integer(minimum=1))
        assert build_manifest(doc) == _empty_manifest(
            bounds_inclusive=True,
            bounds_open=True,
        )

    def test_enum_multi_not_single(self) -> None:
        doc = _doc_with_type(Enum(values=["a", "b"], base_type="string"))
        assert build_manifest(doc) == _empty_manifest(enum_multi=True)

    def test_const_single_not_multi(self) -> None:
        doc = _doc_with_type(Const(value="x"))
        assert build_manifest(doc) == _empty_manifest(enum_single=True)

    def test_union_sets_has_type_union(self) -> None:
        doc = _doc_with_type(Union(branches=[String(), Null()]))
        assert build_manifest(doc) == _empty_manifest(has_type_union=True)

    def test_header_only_tool_sets_has_parameterless_tool(self) -> None:
        doc = Document(tools=[ToolBlock(name="ping", description="Check connectivity")])
        assert build_manifest(doc) == _empty_manifest(has_parameterless_tool=True)

    def test_tool_with_fields_does_not_set_parameterless(self) -> None:
        doc = _doc_with_type(String())
        assert build_manifest(doc).has_parameterless_tool is False

    def test_field_with_default_sets_has_default_value(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="limit", type=Integer(), default=10)],
                )
            ]
        )
        assert build_manifest(doc) == _empty_manifest(has_default_value=True)

    def test_field_without_default_does_not_set_has_default_value(self) -> None:
        doc = _doc_with_type(String())
        assert build_manifest(doc).has_default_value is False

    def test_raw_schema_tool_sets_has_fallback(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(name="ok", fields=[Field(name="x", type=String())]),
                RawSchema(schema={"name": "raw", "type": "object"}),
            ]
        )
        assert build_manifest(doc) == _empty_manifest(has_fallback=True)

    def test_all_raw_schema_tools_sets_all_fallback(self) -> None:
        doc = Document(
            tools=[
                RawSchema(schema={"name": "a", "type": "object"}),
                RawSchema(schema={"name": "b", "type": "object"}),
            ]
        )
        assert build_manifest(doc) == _empty_manifest(has_fallback=True, all_fallback=True)

    def test_nested_exclusive_min_detected(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        Field(
                            name="outer",
                            type=Object(
                                fields=[
                                    Field(
                                        name="inner",
                                        type=Object(
                                            fields=[
                                                Field(
                                                    name="n",
                                                    type=Integer(
                                                        minimum=0,
                                                        exclusive_min=True,
                                                    ),
                                                )
                                            ]
                                        ),
                                    )
                                ]
                            ),
                        )
                    ],
                )
            ]
        )
        assert build_manifest(doc) == _empty_manifest(
            bounds_exclusive=True,
            bounds_open=True,
        )


class TestAssemblePrimer:
    def test_minimal_document_is_core_intro_only(self) -> None:
        doc = _doc_with_type(String())
        text = primer._assemble_primer(build_manifest(doc))
        assert text == _CORE_INTRO
        assert _CORE_RULE_MIXED not in text
        assert _CORE_RULE_ALL_REQUIRED not in text

    def test_exclusive_only_bounds_clause(self) -> None:
        # Both endpoints exclusive: (0,1) — no inclusive bracket anywhere.
        doc = _doc_with_type(Integer(minimum=0, maximum=1, exclusive_min=True, exclusive_max=True))
        text = primer._assemble_primer(build_manifest(doc))
        assert _CLAUSE_BOUNDS_BASE in text
        assert _CLAUSE_BOUNDS_EXCL_ONLY in text
        assert _CLAUSE_BOUNDS_BOTH_STYLES not in text
        assert _CLAUSE_OPEN_BOUND not in text

    def test_mixed_inclusive_lower_exclusive_upper_emits_both_styles(self) -> None:
        # [1,50) has an inclusive lower bracket and an exclusive upper paren —
        # both styles are present, so the contrast sentence must appear.
        doc = _doc_with_type(Integer(minimum=1, maximum=50, exclusive_max=True))
        text = primer._assemble_primer(build_manifest(doc))
        assert _CLAUSE_BOUNDS_BASE in text
        assert _CLAUSE_BOUNDS_BOTH_STYLES in text
        assert _CLAUSE_BOUNDS_EXCL_ONLY not in text

    def test_mixed_inclusive_and_exclusive_bounds_clause(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[
                        Field(name="a", type=Integer(minimum=0, maximum=100)),
                        Field(
                            name="b",
                            type=Integer(minimum=0, maximum=100, exclusive_max=True),
                        ),
                    ],
                )
            ]
        )
        text = primer._assemble_primer(build_manifest(doc))
        assert _CLAUSE_BOUNDS_BOTH_STYLES in text

    def test_open_bound_clause(self) -> None:
        doc = _doc_with_type(Integer(minimum=1))
        text = primer._assemble_primer(build_manifest(doc))
        assert _CLAUSE_OPEN_BOUND in text

    def test_enum_multi_clause_only(self) -> None:
        doc = _doc_with_type(Enum(values=["a", "b"], base_type="string"))
        text = primer._assemble_primer(build_manifest(doc))
        assert _CLAUSE_ENUM_MULTI in text
        assert _CLAUSE_ENUM_SINGLE not in text

    def test_parameterless_tool_clause(self) -> None:
        doc = Document(tools=[ToolBlock(name="ping")])
        text = primer._assemble_primer(build_manifest(doc))
        assert primer._CLAUSE_PARAMETERLESS_TOOL in text

    def test_parameterless_clause_absent_when_tool_has_fields(self) -> None:
        doc = _doc_with_type(String())
        text = primer._assemble_primer(build_manifest(doc))
        assert primer._CLAUSE_PARAMETERLESS_TOOL not in text

    def test_default_value_clause_only_when_present(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="t",
                    fields=[Field(name="limit", type=Integer(), default=10)],
                )
            ]
        )
        text = primer._assemble_primer(build_manifest(doc))
        assert primer._CLAUSE_DEFAULT_VALUE in text

    def test_default_value_clause_absent_without_defaults(self) -> None:
        doc = _doc_with_type(String())
        text = primer._assemble_primer(build_manifest(doc))
        assert primer._CLAUSE_DEFAULT_VALUE not in text

    def test_full_grammar_contains_every_clause(self) -> None:
        text = primer._assemble_primer(build_manifest(_doc_with_type(String())), full_grammar=True)
        for name in (
            "_CLAUSE_DEFAULT_VALUE",
            "_CLAUSE_BOUNDS_BASE",
            "_CLAUSE_BOUNDS_BOTH_STYLES",
            "_CLAUSE_OPEN_BOUND",
            "_CLAUSE_MULTIPLE_OF",
            "_CLAUSE_STRING_LENGTH",
            "_CLAUSE_STRING_REGEX",
            "_CLAUSE_ENCODING_MEDIA",
            "_CLAUSE_ARRAY_TYPED",
            "_CLAUSE_ARRAY_BOUNDS",
            "_CLAUSE_ARRAY_UNIQUE",
            "_CLAUSE_TYPE_UNION",
            "_CLAUSE_ENUM_MULTI",
            "_CLAUSE_ENUM_SINGLE",
            "_CLAUSE_DEFS_REF",
            "_CLAUSE_PARAMETERLESS_TOOL",
            "_CLAUSE_FALLBACK",
        ):
            assert getattr(primer, name) in text


class TestRequiredUniformity:
    """Regression guard: mixed case must keep exact historical core + contract text."""

    def test_all_required_core_and_contract(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="echo",
                    fields=[Field(name="message", type=String(), required=True)],
                )
            ]
        )
        manifest = build_manifest(doc)
        assert manifest.required_uniformity == "all_required"
        text = primer._assemble_primer(manifest)
        assert _CORE_RULE_ALL_REQUIRED in text
        assert _CORE_RULE_MIXED not in text
        contract = build_output_contract(manifest.required_uniformity)
        assert "Include every required parameter." in contract
        assert "optional parameters only when" not in contract
        result = PrimerResult(
            primer_text=text,
            manifest=manifest,
            cats_text="echo\n  message* string",
            all_fallback=False,
        )
        assert build_output_contract(manifest.required_uniformity) in build_system_prompt(result)

    def test_all_optional_core_and_contract(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="search",
                    fields=[
                        Field(name="query", type=String(), required=False),
                        Field(name="limit", type=Integer(), required=False),
                    ],
                )
            ]
        )
        manifest = build_manifest(doc)
        assert manifest.required_uniformity == "all_optional"
        text = primer._assemble_primer(manifest)
        assert text == _CORE_INTRO
        assert _CORE_RULE_MIXED not in text
        assert _CORE_RULE_ALL_REQUIRED not in text
        contract = build_output_contract(manifest.required_uniformity)
        assert "Include any parameters you have values for." in contract
        assert "Include every required parameter" not in contract

    def test_mixed_keeps_exact_historical_core_and_contract(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="search",
                    fields=[
                        Field(name="query", type=String(), required=True),
                        Field(name="limit", type=Integer(), required=False),
                    ],
                )
            ]
        )
        manifest = build_manifest(doc)
        assert manifest.required_uniformity == "mixed"
        assert primer._assemble_primer(manifest) == _CORE
        assert build_output_contract(manifest.required_uniformity) == OUTPUT_CONTRACT

    def test_mixed_across_two_tools(self) -> None:
        doc = Document(
            tools=[
                ToolBlock(
                    name="a",
                    fields=[Field(name="x", type=String(), required=True)],
                ),
                ToolBlock(
                    name="b",
                    fields=[Field(name="y", type=String(), required=False)],
                ),
            ]
        )
        assert build_manifest(doc).required_uniformity == "mixed"
        assert primer._assemble_primer(build_manifest(doc)) == _CORE

    def test_zero_parameter_tool_is_all_optional(self) -> None:
        doc = Document(tools=[ToolBlock(name="ping", description="Check connectivity")])
        manifest = build_manifest(doc)
        assert manifest.required_uniformity == "all_optional"
        assert primer._assemble_primer(manifest).startswith(_CORE_INTRO)
        assert _CORE_RULE_MIXED not in primer._assemble_primer(manifest)


class TestAllFallback:
    def test_all_raw_schema_document(self) -> None:
        doc = Document(
            tools=[RawSchema(schema={"name": "only", "type": "object"})]
        )
        result = PrimerResult(
            primer_text="",
            manifest=build_manifest(doc),
            cats_text=None,
            all_fallback=True,
        )
        assert result.all_fallback is True
        assert result.primer_text == ""
        assert result.cats_text is None

    def test_generate_primer_from_json_all_fallback(self) -> None:
        result = generate_primer_from_json(copy.deepcopy(TOOL_WITH_NOT))
        assert result.all_fallback is True
        assert result.primer_text == ""
        assert result.cats_text is None


class TestEquivalence:
    def test_clean_tool_json_and_cats_paths_match(self) -> None:
        json_result = generate_primer_from_json(copy.deepcopy(CLEAN_TOOL))
        cats_result = generate_primer_from_cats(json_result.cats_text)
        assert json_result.manifest == cats_result.manifest
        assert json_result.primer_text == cats_result.primer_text

    def test_exclusive_bounds_json_and_cats_paths_match(self) -> None:
        json_result = generate_primer_from_json(copy.deepcopy(TOOL_WITH_EXCLUSIVE_BOUNDS))
        cats_result = generate_primer_from_cats(json_result.cats_text)
        assert json_result.manifest == cats_result.manifest
        assert json_result.primer_text == cats_result.primer_text

    def test_enum_json_and_cats_paths_match(self) -> None:
        json_result = generate_primer_from_json(copy.deepcopy(TOOL_WITH_ENUM))
        cats_result = generate_primer_from_cats(json_result.cats_text)
        assert json_result.manifest == cats_result.manifest
        assert json_result.primer_text == cats_result.primer_text


class TestInputPaths:
    def test_generate_primer_from_json_smoke(self) -> None:
        result = generate_primer_from_json(copy.deepcopy(CLEAN_TOOL))
        assert isinstance(result, PrimerResult)
        assert result.cats_text
        assert result.primer_text
        assert not result.all_fallback

    def test_generate_primer_from_json_defaults_match_part1_flags(self) -> None:
        import cats

        tool = {
            "name": "coords",
            "type": "dict",
            "properties": {"lat": {"type": "float"}, "lon": {"type": "float"}},
            "required": ["lat", "lon"],
        }
        default_result = generate_primer_from_json(copy.deepcopy(tool))
        explicit = cats.convert_with_report(
            copy.deepcopy(tool),
            assume_closed=True,
            map_python_types=True,
        )
        assert default_result.cats_text == explicit.cats_text
        raw = cats.convert_with_report(
            copy.deepcopy(tool),
            assume_closed=False,
            map_python_types=False,
        )
        assert default_result.cats_text != raw.cats_text

    def test_generate_primer_from_cats_smoke(self) -> None:
        cats_text = generate_primer_from_json(copy.deepcopy(CLEAN_TOOL)).cats_text
        assert cats_text is not None
        result = generate_primer_from_cats(cats_text)
        assert isinstance(result, PrimerResult)
        assert result.cats_text == cats_text

    def test_malformed_cats_raises_parse_error(self) -> None:
        # A {-opening block is parsed as a raw JSON Schema tool (§7.5);
        # invalid JSON inside it raises ParseError at the structural level.
        with pytest.raises(ParseError):
            generate_primer_from_cats('{"unclosed": ')

    def test_output_contract_is_exported(self) -> None:
        assert "arguments" in OUTPUT_CONTRACT
        assert "tool_name" in OUTPUT_CONTRACT


class TestBuildSystemPrompt:
    def test_assembles_three_sections_in_order(self) -> None:
        result = generate_primer_from_json(copy.deepcopy(TOOL_WITH_ENUM))
        prompt = build_system_prompt(result)
        assert prompt.startswith(result.primer_text)
        assert "## Available Tools" not in prompt
        assert "## Calling a Tool" not in prompt
        assert prompt.count("\n---\n") == 2
        parts = prompt.split("\n---\n")
        assert len(parts) == 3
        assert parts[0].strip() == result.primer_text.strip()
        assert f"```\n{result.cats_text}\n```" in parts[1]
        assert build_output_contract(result.manifest.required_uniformity) in parts[2]

    def test_cats_text_is_fenced(self) -> None:
        result = generate_primer_from_json(copy.deepcopy(TOOL_WITH_ENUM))
        prompt = build_system_prompt(result)
        assert f"```\n{result.cats_text}\n```" in prompt

    def test_output_contract_is_present(self) -> None:
        result = generate_primer_from_json(copy.deepcopy(TOOL_WITH_ENUM))
        prompt = build_system_prompt(result)
        assert build_output_contract(result.manifest.required_uniformity) in prompt

    def test_all_fallback_raises(self) -> None:
        result = generate_primer_from_json(copy.deepcopy(TOOL_WITH_NOT))
        assert result.all_fallback is True
        with pytest.raises(ValueError):
            build_system_prompt(result)
