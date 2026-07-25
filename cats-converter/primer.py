"""
primer.py — calibrated CATS primer generation for system prompts.

Walks a CATS document AST, records which notation features appear, and assembles
only the primer clauses a model needs to read that document. Sibling to cats.py;
nothing in the converter imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

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
    Node,
    Null,
    Number,
    Object,
    RawSchema,
    Reference,
    String,
    ToolBlock,
    TypeNode,
    Union,
)
from parser import parse_text

import cats

RequiredUniformity = Literal["all_required", "all_optional", "mixed"]

__all__ = [
    "Manifest",
    "PrimerResult",
    "OUTPUT_CONTRACT",
    "RequiredUniformity",
    "build_manifest",
    "build_output_contract",
    "build_system_prompt",
    "generate_primer_from_json",
    "generate_primer_from_cats",
]

# Fixed core intro (always emitted)
_CORE_INTRO = """\
The following tools are described in CATS, a compact notation for tool definitions. Read each tool like a typed function signature.

- Each tool starts with its name on its own line, optionally followed by `#` and a description. The indented lines below are its parameters.
- Each parameter line reads left to right: the parameter name, then its type, then, optionally, a description introduced by `#`."""

# Required/optional rule variants (calibrated on whole-prompt parameter uniformity)
_CORE_RULE_MIXED = (
    "A parameter is **required** only if its name is immediately followed by `*`. "
    "A parameter with no `*` is optional."
)
_CORE_RULE_ALL_REQUIRED = "All fields of the below tools are required (indicated by `*`)."

# Mixed case — full core as historically emitted (regression reference / export)
_CORE = f"{_CORE_INTRO}\n- {_CORE_RULE_MIXED}"

# Default values on parameter lines
_CLAUSE_DEFAULT_VALUE = "`=value` sets a default; omit the parameter to use it."

# Numeric bounds — assembled from up to three independent fragments
_CLAUSE_BOUNDS_BASE = "Numeric ranges use interval notation: `integer[1,100]` accepts 1 to 100."
_CLAUSE_BOUNDS_EXCL_ONLY = "A parenthesis excludes that endpoint, so `(0,1)` is between 0 and 1 with neither included."
_CLAUSE_BOUNDS_BOTH_STYLES = "A square bracket includes the endpoint and a parenthesis excludes it, so `[0,100)` allows 0 but not 100."
_CLAUSE_OPEN_BOUND = "A missing number means that side is unbounded."

# multipleOf
_CLAUSE_MULTIPLE_OF = "A `%` after a number means the value must be a multiple of what follows it: `integer%5` accepts multiples of 5, `number%0.01` rounds to hundredths."

# String annotations
_CLAUSE_STRING_LENGTH = "`:length[1,20]` after a string constrains its length: at least 1 character, at most 20."
_CLAUSE_STRING_REGEX = "`:regex[\"...\"]` after a string means the value must match that regular expression."
_CLAUSE_ENCODING_MEDIA = "`:encoding[base64]` means the string is encoded that way; `:media[\"application/pdf\"]` gives its MIME type."

# Array
_CLAUSE_ARRAY_TYPED = "`array<T>` elements all have type T (T can be nested)."
_CLAUSE_ARRAY_BOUNDS = "Brackets after an array, like `array<string>[1,10]`, constrain how many elements it has — here, 1 to 10. (This is a count of elements, not a range of values.)"
_CLAUSE_ARRAY_UNIQUE = "`:unique` on an array means all its elements must be distinct."

# Unions and enums
_CLAUSE_TYPE_UNION = "A `|` between types means any one of them: `string|array<string>` accepts either. `X|null` marks a nullable field."
_CLAUSE_ENUM_MULTI = "A `|` between values restricts the field to exactly those: `sort relevance|price|newest` allows only those three."
_CLAUSE_ENUM_SINGLE = "A quoted value like `mode \"automatic\"` means the field must equal exactly that."

# $defs references
_CLAUSE_DEFS_REF = "A type written `$Name` (for example `home_address $Address`) refers to a reusable shape defined under the `$defs` block at the top of the document. Look there for that shape's fields."

# Fallback (mixed document)
_CLAUSE_FALLBACK = "Some tools below are written as raw JSON Schema (a block starting with `{`) instead of CATS. Read those exactly as you normally read JSON Schema tool definitions. Every tool is either fully CATS or fully JSON Schema — the two are never mixed inside one tool."

# Header-only tools (zero parameters)
_CLAUSE_PARAMETERLESS_TOOL = (
    'A tool shown with a name but no parameter lines takes no arguments. Call it with '
    'an empty arguments object: `{"name": "<tool_name>", "arguments": {}}`.'
)

_OUTPUT_CONTRACT_PREFIX = """\
To call a tool, respond with only a single fenced JSON block in this exact shape, and nothing else:

```json
{"name": "<tool_name>", "arguments": { ... }}
```

Arguments are ordinary JSON matching the tool's parameters."""

_OUTPUT_CONTRACT_SUFFIX = (
    " If no tool applies, respond in plain text with no JSON block."
)

_OUTPUT_CONTRACT_ARGS_MIXED = (
    " Include every required parameter; optional parameters only when you have a value for them."
)
_OUTPUT_CONTRACT_ARGS_ALL_REQUIRED = " Include every required parameter."
_OUTPUT_CONTRACT_ARGS_ALL_OPTIONAL = " Include any parameters you have values for."


def build_output_contract(uniformity: RequiredUniformity) -> str:
    """Assemble the output contract for a prompt's required/optional uniformity."""
    if uniformity == "all_required":
        args_clause = _OUTPUT_CONTRACT_ARGS_ALL_REQUIRED
    elif uniformity == "all_optional":
        args_clause = _OUTPUT_CONTRACT_ARGS_ALL_OPTIONAL
    else:
        args_clause = _OUTPUT_CONTRACT_ARGS_MIXED
    return _OUTPUT_CONTRACT_PREFIX + args_clause + _OUTPUT_CONTRACT_SUFFIX


# Mixed-case contract — module-level constant for backward-compatible imports
OUTPUT_CONTRACT = build_output_contract("mixed")


@dataclass
class Manifest:
    # --- Numeric bounds (Integer / Number) ---
    bounds_inclusive: bool
    bounds_exclusive: bool
    bounds_open: bool
    has_multiple_of: bool

    # --- String annotations ---
    has_string_length: bool
    string_length_open: bool
    has_regex: bool
    has_encoding_media: bool

    # --- Array ---
    has_typed_array: bool
    has_array_bounds: bool
    has_unique: bool

    # --- Unions and enums ---
    has_type_union: bool
    enum_multi: bool
    enum_single: bool

    # --- $defs references ---
    has_defs_reference: bool

    # --- Header-only tools ---
    has_parameterless_tool: bool

    # --- Default values ---
    has_default_value: bool

    # --- Whole-prompt required/optional uniformity (tool parameter lines only) ---
    required_uniformity: RequiredUniformity

    # --- Fallback ---
    has_fallback: bool
    all_fallback: bool


@dataclass
class PrimerResult:
    primer_text: str
    manifest: Manifest
    cats_text: Optional[str]
    all_fallback: bool


@dataclass
class _ManifestAcc:
    bounds_inclusive: bool = False
    bounds_exclusive: bool = False
    bounds_open: bool = False
    has_multiple_of: bool = False
    has_string_length: bool = False
    string_length_open: bool = False
    has_regex: bool = False
    has_encoding_media: bool = False
    has_typed_array: bool = False
    has_array_bounds: bool = False
    has_unique: bool = False
    has_type_union: bool = False
    enum_multi: bool = False
    enum_single: bool = False
    has_defs_reference: bool = False
    has_parameterless_tool: bool = False
    has_default_value: bool = False
    has_fallback: bool = False


def _visit_numeric_bounds(
    acc: _ManifestAcc,
    *,
    minimum: Optional[float],
    maximum: Optional[float],
    exclusive_min: bool,
    exclusive_max: bool,
) -> None:
    if minimum is not None and not exclusive_min:
        acc.bounds_inclusive = True
    if maximum is not None and not exclusive_max:
        acc.bounds_inclusive = True
    if minimum is not None and exclusive_min:
        acc.bounds_exclusive = True
    if maximum is not None and exclusive_max:
        acc.bounds_exclusive = True
    if (minimum is None) != (maximum is None) and (minimum is not None or maximum is not None):
        acc.bounds_open = True


def _visit_string_length(
    acc: _ManifestAcc,
    *,
    min_length: Optional[int],
    max_length: Optional[int],
) -> None:
    if min_length is not None or max_length is not None:
        acc.has_string_length = True
    if (min_length is None) != (max_length is None) and (
        min_length is not None or max_length is not None
    ):
        acc.string_length_open = True


def _visit_type(node: TypeNode, acc: _ManifestAcc) -> None:
    if isinstance(node, (String, Integer, Number, Boolean, Null, AnyType)):
        if isinstance(node, String):
            _visit_string_length(acc, min_length=node.min_length, max_length=node.max_length)
            if node.pattern is not None:
                acc.has_regex = True
            if node.encoding is not None or node.media is not None:
                acc.has_encoding_media = True
        elif isinstance(node, (Integer, Number)):
            _visit_numeric_bounds(
                acc,
                minimum=node.minimum,
                maximum=node.maximum,
                exclusive_min=node.exclusive_min,
                exclusive_max=node.exclusive_max,
            )
            if node.multiple_of is not None:
                acc.has_multiple_of = True
        return

    if isinstance(node, Array):
        if node.element is not None:
            acc.has_typed_array = True
            _visit_type(node.element, acc)
        if node.min_items is not None or node.max_items is not None:
            acc.has_array_bounds = True
        if node.unique:
            acc.has_unique = True
        return

    if isinstance(node, Object):
        for field in node.fields:
            _visit_field(field, acc)
        return

    if isinstance(node, Union):
        acc.has_type_union = True
        for branch in node.branches:
            _visit_type(branch, acc)
        return

    if isinstance(node, Enum):
        acc.enum_multi = True
        return

    if isinstance(node, Const):
        acc.enum_single = True
        return

    if isinstance(node, Reference):
        acc.has_defs_reference = True
        return


def _visit_field(field: Field, acc: _ManifestAcc) -> None:
    if field.default is not NO_DEFAULT:
        acc.has_default_value = True
    _visit_type(field.type, acc)


def _visit_definition(defn: Definition, acc: _ManifestAcc) -> None:
    for field in defn.fields:
        _visit_field(field, acc)


def _visit_tool(tool: ToolBlock, acc: _ManifestAcc) -> None:
    if not tool.fields:
        acc.has_parameterless_tool = True
    for field in tool.fields:
        _visit_field(field, acc)


def _iter_parameter_fields(field: Field) -> list[Field]:
    """All parameter-line fields under a tool field, including nested object members."""
    fields = [field]
    if isinstance(field.type, Object):
        for nested in field.type.fields:
            fields.extend(_iter_parameter_fields(nested))
    return fields


def _tool_parameter_fields(tool: ToolBlock) -> list[Field]:
    out: list[Field] = []
    for field in tool.fields:
        out.extend(_iter_parameter_fields(field))
    return out


def _required_uniformity(document: Document) -> RequiredUniformity:
    """Classify required/optional `*` usage across all tool parameter lines.

    Zero-parameter prompts are treated as ``all_optional``: no ``*`` appears anywhere.
    """
    required_flags: list[bool] = []
    for tool in document.tools:
        if isinstance(tool, ToolBlock):
            required_flags.extend(f.required for f in _tool_parameter_fields(tool))

    if not required_flags:
        return "all_optional"
    if all(required_flags):
        return "all_required"
    if not any(required_flags):
        return "all_optional"
    return "mixed"


def build_manifest(document: Document) -> Manifest:
    """Walk the AST and record which CATS features appear in ``document``."""
    acc = _ManifestAcc()

    if document.defs is not None:
        for defn in document.defs:
            _visit_definition(defn, acc)

    for tool in document.tools:
        if isinstance(tool, RawSchema):
            acc.has_fallback = True
        else:
            _visit_tool(tool, acc)

    all_fallback = len(document.tools) > 0 and all(
        isinstance(tool, RawSchema) for tool in document.tools
    )

    return Manifest(
        bounds_inclusive=acc.bounds_inclusive,
        bounds_exclusive=acc.bounds_exclusive,
        bounds_open=acc.bounds_open,
        has_multiple_of=acc.has_multiple_of,
        has_string_length=acc.has_string_length,
        string_length_open=acc.string_length_open,
        has_regex=acc.has_regex,
        has_encoding_media=acc.has_encoding_media,
        has_typed_array=acc.has_typed_array,
        has_array_bounds=acc.has_array_bounds,
        has_unique=acc.has_unique,
        has_type_union=acc.has_type_union,
        enum_multi=acc.enum_multi,
        enum_single=acc.enum_single,
        has_defs_reference=acc.has_defs_reference,
        has_parameterless_tool=acc.has_parameterless_tool,
        has_default_value=acc.has_default_value,
        required_uniformity=_required_uniformity(document),
        has_fallback=acc.has_fallback,
        all_fallback=all_fallback,
    )


def _full_manifest() -> Manifest:
    """Every feature flag True; ``all_fallback`` stays False (document-state only)."""
    return Manifest(
        bounds_inclusive=True,
        bounds_exclusive=True,
        bounds_open=True,
        has_multiple_of=True,
        has_string_length=True,
        string_length_open=True,
        has_regex=True,
        has_encoding_media=True,
        has_typed_array=True,
        has_array_bounds=True,
        has_unique=True,
        has_type_union=True,
        enum_multi=True,
        enum_single=True,
        has_defs_reference=True,
        has_parameterless_tool=True,
        has_default_value=True,
        required_uniformity="mixed",
        has_fallback=True,
        all_fallback=False,
    )


def _assemble_core(uniformity: RequiredUniformity) -> str:
    """Assemble the fixed core intro plus the calibrated required/optional rule."""
    if uniformity == "all_required":
        return f"{_CORE_INTRO}\n- {_CORE_RULE_ALL_REQUIRED}"
    if uniformity == "mixed":
        return f"{_CORE_INTRO}\n- {_CORE_RULE_MIXED}"
    return _CORE_INTRO


def _assemble_primer(manifest: Manifest, *, full_grammar: bool = False) -> str:
    """Assemble calibrated primer text from ``manifest`` (no tools, no output contract)."""
    if full_grammar:
        manifest = _full_manifest()

    sections: list[str] = [_assemble_core(manifest.required_uniformity)]
    optional: list[str] = []

    if manifest.has_default_value:
        optional.append(_CLAUSE_DEFAULT_VALUE)

    if manifest.bounds_inclusive or manifest.bounds_exclusive:
        bounds_parts = [_CLAUSE_BOUNDS_BASE]
        if manifest.bounds_inclusive and manifest.bounds_exclusive:
            bounds_parts.append(_CLAUSE_BOUNDS_BOTH_STYLES)
        elif manifest.bounds_exclusive:
            bounds_parts.append(_CLAUSE_BOUNDS_EXCL_ONLY)
        if manifest.bounds_open:
            bounds_parts.append(_CLAUSE_OPEN_BOUND)
        optional.append(" ".join(bounds_parts))

    if manifest.has_multiple_of:
        optional.append(_CLAUSE_MULTIPLE_OF)

    if manifest.has_string_length:
        length_parts = [_CLAUSE_STRING_LENGTH]
        if manifest.string_length_open:
            length_parts.append(_CLAUSE_OPEN_BOUND)
        optional.append(" ".join(length_parts))

    if manifest.has_regex:
        optional.append(_CLAUSE_STRING_REGEX)

    if manifest.has_encoding_media:
        optional.append(_CLAUSE_ENCODING_MEDIA)

    if manifest.has_typed_array:
        optional.append(_CLAUSE_ARRAY_TYPED)

    if manifest.has_array_bounds:
        optional.append(_CLAUSE_ARRAY_BOUNDS)

    if manifest.has_unique:
        optional.append(_CLAUSE_ARRAY_UNIQUE)

    if manifest.has_type_union:
        optional.append(_CLAUSE_TYPE_UNION)

    if manifest.enum_multi:
        optional.append(_CLAUSE_ENUM_MULTI)

    if manifest.enum_single:
        optional.append(_CLAUSE_ENUM_SINGLE)

    if manifest.has_defs_reference:
        optional.append(_CLAUSE_DEFS_REF)

    if manifest.has_parameterless_tool:
        optional.append(_CLAUSE_PARAMETERLESS_TOOL)

    if manifest.has_fallback:
        optional.append(_CLAUSE_FALLBACK)

    sections.extend(optional)
    return "\n\n".join(sections)


def _as_document(ast: Node) -> Document:
    if isinstance(ast, Document):
        return ast
    if isinstance(ast, (ToolBlock, RawSchema)):
        return Document(tools=[ast])
    raise TypeError(f"Cannot build primer from AST root {type(ast).__name__}")


def generate_primer_from_json(
    schema: dict[str, Any] | list[Any],
    *,
    full_grammar: bool = False,
    assume_closed: bool = True,
    map_python_types: bool = True,
) -> PrimerResult:
    """Build a calibrated primer from a JSON Schema tool definition.

    Uses :func:`cats.convert_with_report` so CATS text and the feature manifest
    match the Part 1 eval pipeline (``assume_closed`` and ``map_python_types``
    default to ``True``). Pass ``False`` for either to keep raw JSON Schema
    semantics on that axis.
    """
    conversion = cats.convert_with_report(
        schema,
        assume_closed=assume_closed,
        map_python_types=map_python_types,
    )
    document = parse_text(conversion.cats_text)
    manifest = build_manifest(document)

    if manifest.all_fallback:
        return PrimerResult("", manifest, None, True)

    effective = _full_manifest() if full_grammar else manifest
    return PrimerResult(
        _assemble_primer(effective, full_grammar=False),
        effective,
        conversion.cats_text,
        False,
    )


def generate_primer_from_cats(
    cats_text: str,
    *,
    full_grammar: bool = False,
) -> PrimerResult:
    """Build a calibrated primer from a CATS document string."""
    document = parse_text(cats_text)
    manifest = build_manifest(document)

    if manifest.all_fallback:
        return PrimerResult("", manifest, None, True)

    effective = _full_manifest() if full_grammar else manifest
    return PrimerResult(
        _assemble_primer(effective, full_grammar=False),
        effective,
        cats_text,
        False,
    )


_CATS_FENCE_LANG = ""  # plain fence (no language tag)


def assemble_prompt_sections(
    primer_text: str,
    fenced_tools: str,
    manifest: Manifest,
) -> str:
    """Join primer prose, tool block, and output contract with ``---`` separators."""
    return "\n\n".join(
        [
            primer_text,
            "---",
            fenced_tools,
            "---",
            build_output_contract(manifest.required_uniformity),
        ]
    )


def build_system_prompt(result: PrimerResult) -> str:
    """Assemble the full system prompt from a PrimerResult: primer, fenced CATS
    tool block, and the output contract, separated by markdown section breaks.

    Raises ValueError if called on an all-fallback result, where CATS does not
    apply and there is no primer or CATS text to assemble (use the provider's
    native tool-calling channel instead).
    """
    if result.all_fallback:
        raise ValueError(
            "build_system_prompt called on an all-fallback result; CATS does "
            "not apply here. Use the native tool-calling channel instead."
        )

    fenced_tools = f"```{_CATS_FENCE_LANG}\n{result.cats_text}\n```"

    return assemble_prompt_sections(result.primer_text, fenced_tools, result.manifest)
