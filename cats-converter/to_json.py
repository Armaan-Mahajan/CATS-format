"""
to_json.py — the CATS -> JSON Schema serializer (the forward, mechanical direction).

Fourth stage of the pipeline:

    text --[lexer]--> tokens --[parser]--> AST --[validate]--> AST --[to_json]--> JSON Schema

Input  : a (validated) `Document` AST, or any single node from nodes.py.
Output : a native Python dict that is JSON Schema draft 2020-12 (NOT a string).

This direction is MECHANICAL (spec §7.2): every node has exactly one correct
JSON Schema form. There are no judgment calls in the per-schema mapping — §7.2's
table fixes each construct and §7.2's final paragraph fixes the keyword order.
The serializer's only "decisions" are the three canonicalizations §7.2/§7.4
assign to the output side (not to the validator):

  - a two-member true|false enum emits `{"type": "boolean"}`, not an enum (§5.6);
  - every object emits `"additionalProperties": false` — open objects normalize
    to closed (§5.4 / §7.4);
  - a pipe-union always emits `anyOf`, never `oneOf` (§5.5 / §7.4).

KEYWORD ORDER (§7.2): within every emitted schema object, keys appear in one
fixed canonical order — type, const/enum, format, default, properties, required,
additionalProperties, items, anyOf, $ref, the validation constraints, then
finally description. Python dicts preserve insertion order, so the output order
is the insertion order; `_canonical` enforces the §7.2 sequence as a safety net
(used where a Field splices its `default`/`description` into a type's schema).
We never sort alphabetically (that would destroy the §7.2 order). Encoded tool
schemas use the §7.2.1 envelope order: `name`, `description`, `$defs`, then the
parameter-schema body. Fallen-back tools (RawSchema) are emitted verbatim with
no key reordering (§7.5).

DOCUMENT ENVELOPE (spec §7.2): the atomic output unit is ONE TOOL — a
self-contained JSON Schema object carrying its `name`, its parameter schema
(type/properties/required/additionalProperties:false), and a local `$defs` with
ONLY the definitions that tool references (transitively). A document with
multiple tools serializes to a JSON ARRAY of these tool schemas, in declaration
order:

    [
      { "name": "<tool name>", "type": "object", "properties": {...},
        "required": [...], "additionalProperties": false,
        "$defs": { "<DefName>": <object schema>, ... } },  # only refs this tool uses
      ...
    ]

Each tool embeds only the definitions it actually references — found by walking
the tool's type tree for Reference nodes and following those references
TRANSITIVELY into the referenced definitions (with a cycle guard). A tool that
references nothing emits no `$defs` key. The per-construct mapping is unchanged:
`"$ref": "#/$defs/Name"` (§7.2) resolves against the tool's embedded `$defs`, so
the definition travels with the tool. Each tool schema is independently valid
because tool-calling APIs send tool schemas individually; a document-level shared
`$defs` would dangle on extraction.

`count_definition_inlinings` reports the duplication this self-containment
introduces (a shared definition embedded into each tool that uses it) for later
token-cost measurement.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Optional

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
    Union,
)

# Canonical keyword order (§7.2). Every constructed schema's top-level keys are
# emitted in this sequence. The numeric-bound keys interleave lower-before-upper
# so that whichever of minimum/exclusiveMinimum is present precedes whichever of
# maximum/exclusiveMaximum is present.
_KEY_ORDER: tuple[str, ...] = (
    # `name` and `$defs` appear only on encoded tool envelopes (§7.2.1), which
    # are assembled in fixed order without `_canonical`. Inner schema objects use
    # the sequence below; `description` sorts last among them (§7.2).
    "name",
    "type",
    "const",
    "enum",
    "format",
    "default",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "anyOf",
    "$ref",
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "contentEncoding",
    "contentMediaType",
    "minItems",
    "maxItems",
    "uniqueItems",
    "description",
    # Tool envelope only (§7.2.1); not used by `_canonical` on inner schemas.
    "$defs",
)
_KEY_RANK: dict[str, int] = {key: i for i, key in enumerate(_KEY_ORDER)}


def _canonical(schema: dict[str, Any]) -> dict[str, Any]:
    """Reorder a constructed schema's TOP-LEVEL keys into the §7.2 sequence.

    Only the keys of `schema` itself are reordered; nested schemas and the
    contents of `default` values are left untouched (the sort is not recursive),
    so property maps and opaque default JSON keep their own order. An unknown key
    sorts to the end, and Python's stable sort preserves its relative position.
    """
    return {
        key: schema[key]
        for key in sorted(schema, key=lambda k: _KEY_RANK.get(k, len(_KEY_ORDER)))
    }


def _is_true_false_enum(enum: Enum) -> bool:
    """True iff this Enum is exactly the two boolean members True and False.

    Such a set canonicalizes to the `boolean` type, not an enum (§5.6). The
    `isinstance(v, bool)` guard matters because in Python `True == 1` and
    `False == 0`, so a {0, 1} integer enum would otherwise masquerade as boolean.
    """
    return (
        len(enum.values) == 2
        and all(isinstance(v, bool) for v in enum.values)
        and set(enum.values) == {True, False}
    )


def _infer_enum_type(enum: Enum) -> Optional[str]:
    """The JSON Schema `type` to emit beside an enum's `enum` array (§7.2).

    Prefers the node's recorded `base_type`; falls back to inferring from the
    member values so a hand-built Enum still emits a type. Returns None only for
    a genuinely mixed set (which the validator rejects under §5.6), in which case
    the caller emits `enum` without a `type`.
    """
    if enum.base_type is not None:
        return enum.base_type

    values = enum.values
    if values and all(isinstance(v, bool) for v in values):
        return "boolean"
    if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "integer"
    if values and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    ):
        return "number"
    if values and all(isinstance(v, str) for v in values):
        return "string"
    return None


def _serialize_string(node: String) -> dict[str, Any]:
    """String (§5.2) -> {"type":"string"} plus its present §6.3 constraints."""
    schema: dict[str, Any] = {"type": "string"}
    if node.format is not None:
        schema["format"] = node.format
    if node.min_length is not None:
        schema["minLength"] = node.min_length
    if node.max_length is not None:
        schema["maxLength"] = node.max_length
    if node.pattern is not None:
        schema["pattern"] = node.pattern
    if node.encoding is not None:
        schema["contentEncoding"] = node.encoding
    if node.media is not None:
        schema["contentMediaType"] = node.media
    return _canonical(schema)


def _serialize_numeric(node: Integer | Number, type_name: str) -> dict[str, Any]:
    """Integer/Number (§5.2) -> {"type":...} plus §6.2 numeric constraints.

    Draft 2020-12 makes `exclusiveMinimum`/`exclusiveMaximum` take the numeric
    bound value (not the draft-04 boolean), so an exclusive bound emits the
    `exclusive*` keyword INSTEAD OF the inclusive one carrying the same value.
    """
    schema: dict[str, Any] = {"type": type_name}
    if node.format is not None:
        schema["format"] = node.format
    if node.minimum is not None:
        if node.exclusive_min:
            schema["exclusiveMinimum"] = node.minimum
        else:
            schema["minimum"] = node.minimum
    if node.maximum is not None:
        if node.exclusive_max:
            schema["exclusiveMaximum"] = node.maximum
        else:
            schema["maximum"] = node.maximum
    if node.multiple_of is not None:
        schema["multipleOf"] = node.multiple_of
    return _canonical(schema)


def _serialize_array(node: Array) -> dict[str, Any]:
    """Array (§5.3) -> {"type":"array"} plus items and §6.4 constraints.

    A bare array (element is None) emits no `items` (§7.2). `uniqueItems` is
    emitted only when True; its absence is JSON Schema's default of false (§6.4).
    """
    schema: dict[str, Any] = {"type": "array"}
    if node.format is not None:
        schema["format"] = node.format
    if node.element is not None:
        schema["items"] = to_json(node.element)
    if node.min_items is not None:
        schema["minItems"] = node.min_items
    if node.max_items is not None:
        schema["maxItems"] = node.max_items
    if node.unique:
        schema["uniqueItems"] = True
    return _canonical(schema)


def _serialize_object_fields(
    fields: list[Field], description: Optional[str]
) -> dict[str, Any]:
    """Build a closed object schema from a list of field lines (§5.4 / §7.2).

    Shared by Object, ToolBlock, and Definition — all three are object schemas
    that differ only in where they sit. Emits `properties` (always, possibly
    empty), `required` (only when non-empty, in field-declaration order), and
    `additionalProperties: false` unconditionally (§5.4). A description, when
    given, is the tool/definition/object prose (§4.6).
    """
    properties: dict[str, Any] = {fld.name: to_json(fld) for fld in fields}
    required: list[str] = [fld.name for fld in fields if fld.required]

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    schema["additionalProperties"] = False
    if description is not None:
        schema["description"] = description
    return _canonical(schema)


def _serialize_enum(node: Enum) -> dict[str, Any]:
    """Enum (§5.6) -> {"type": <inferred>, "enum": [...]} (§7.2).

    The two-member true|false set canonicalizes to `boolean` (§5.6). Otherwise an
    explicit `type` is emitted alongside `enum` (always, per §7.2) when the base
    type is determinable; a genuinely mixed enum (validator-rejected) emits bare
    `enum`.
    """
    if _is_true_false_enum(node):
        return {"type": "boolean"}

    schema: dict[str, Any] = {}
    inferred = _infer_enum_type(node)
    if inferred is not None:
        schema["type"] = inferred
    schema["enum"] = list(node.values)
    return _canonical(schema)


def _serialize_field(node: Field) -> dict[str, Any]:
    """A field line -> its type's subschema plus `default`/`description` (§4.5/§4.6).

    NO_DEFAULT means no default was given; any other value (including a literal
    None == JSON null) is a real default and is emitted. `_canonical` then slots
    `default` and `description` into their §7.2 positions within the type schema.
    """
    schema = dict(to_json(node.type))
    if node.default is not NO_DEFAULT:
        schema["default"] = node.default
    if node.description is not None:
        schema["description"] = node.description
    return _canonical(schema)


def _collect_reference_names(type_node: object, names: set[str]) -> None:
    """Add every `$defs` Reference name reachable inside one type node to `names`.

    Recurses through the only composite shapes that nest a type: an Array's
    element, a Union's branches, and an Object's field types. Primitives, enums,
    consts, and RawSchema carry no CATS reference (a `$ref` buried in RawSchema
    JSON is opaque per §7.5 and intentionally not followed).
    """
    if isinstance(type_node, Reference):
        names.add(type_node.name)
    elif isinstance(type_node, Array):
        if type_node.element is not None:
            _collect_reference_names(type_node.element, names)
    elif isinstance(type_node, Union):
        for branch in type_node.branches:
            _collect_reference_names(branch, names)
    elif isinstance(type_node, Object):
        for fld in type_node.fields:
            _collect_reference_names(fld.type, names)


def _referenced_definition_names(
    fields: list[Field], defs_by_name: dict[str, Definition]
) -> set[str]:
    """Names of every definition reachable from `fields`, TRANSITIVELY (§5.7).

    Seeds with the references found directly in the given fields, then expands
    through each referenced definition's own fields. `resolved` doubles as the
    cycle guard: a name already resolved is never re-expanded, so a reference
    cycle (A -> B -> A) terminates. Names with no matching definition (dangling
    references — the validator's concern) stay in the set but embed nothing.
    """
    pending: set[str] = set()
    for fld in fields:
        _collect_reference_names(fld.type, pending)

    resolved: set[str] = set()
    while pending:
        name = pending.pop()
        if name in resolved:
            continue
        resolved.add(name)
        definition = defs_by_name.get(name)
        if definition is None:
            continue  # dangling reference: nothing to walk into or embed
        for fld in definition.fields:
            _collect_reference_names(fld.type, pending)
    return resolved


def _serialize_document(node: Document) -> list[dict[str, Any]]:
    """A whole document -> a LIST of SELF-CONTAINED tool schemas (declaration order).

    Each tool is an object schema that embeds, under its own `$defs`, only the
    definitions it references (transitively); see the module header. There is no
    document-root `$defs`. A tool referencing nothing emits no `$defs` key. A
    tool that fell back (a RawSchema, §7.5) is emitted verbatim in place.
    """
    defs_by_name = {definition.name: definition for definition in (node.defs or [])}
    out: list[dict[str, Any]] = []
    for tool in node.tools:
        if isinstance(tool, RawSchema):
            out.append(copy.deepcopy(tool.schema))
        else:
            out.append(_serialize_tool(tool, defs_by_name))
    return out


def _serialize_tool(
    tool: ToolBlock, defs_by_name: dict[str, Definition]
) -> dict[str, Any]:
    """One self-contained tool schema (the atomic output unit; see header).

    Builds the tool's object schema, prepends its self-identifying `name`, and
    embeds — under a local `$defs` — only the definitions the tool references
    transitively (resolved against `defs_by_name`, in that map's declaration
    order). `defs_by_name` is empty when a ToolBlock is serialized on its own
    (no document context), so a standalone tool simply carries no `$defs`.
    """
    body = _serialize_object_fields(tool.fields, None)

    referenced = _referenced_definition_names(tool.fields, defs_by_name)
    embedded = {
        definition.name: to_json(definition)
        for definition in defs_by_name.values()
        if definition.name in referenced
    }

    # §7.2.1 tool envelope: name → description → $defs → parameter schema.
    schema: dict[str, Any] = {"name": tool.name}
    if tool.description is not None:
        schema["description"] = tool.description
    if embedded:
        schema["$defs"] = embedded
    schema.update(body)
    return schema


def count_definition_inlinings(document: Document) -> int:
    """Instrumentation: how many times definitions get embedded across all tools.

    Sums, over every tool, the count of distinct definitions that tool embeds
    (transitively referenced AND resolvable in `$defs`). A definition shared by
    two tools counts twice — that double-count IS the duplication this self-
    contained envelope introduces, which the eval can weigh against its token
    cost. Not part of the JSON Schema output; a measurement aid only.
    """
    defs_by_name = {definition.name: definition for definition in (document.defs or [])}
    total = 0
    for tool in document.tools:
        if isinstance(tool, RawSchema):
            continue  # a fallen-back tool embeds nothing through the CATS path (§7.5)
        referenced = _referenced_definition_names(tool.fields, defs_by_name)
        total += sum(1 for name in referenced if name in defs_by_name)
    return total


def to_json(node: Node) -> dict[str, Any] | list[dict[str, Any]]:
    """Serialize a CATS AST node to JSON Schema (draft 2020-12).

    Dispatches on node type per §7.2. A `Document` returns a LIST of
    self-contained tool schemas in declaration order (§7.2 envelope); every other
    node returns a single dict — a `ToolBlock` its self-contained tool schema, a
    `Definition` an object schema, a `Field` its subschema, or any single type
    node its subschema. The result is a fresh structure; the tree is not mutated.
    """
    if isinstance(node, Document):
        return _serialize_document(node)

    if isinstance(node, ToolBlock):
        # A lone tool has no document context, so no definitions to embed; it
        # still carries its own `name` so it is self-identifying when standalone.
        return _serialize_tool(node, {})

    if isinstance(node, Definition):
        # A definition is a plain object schema; its name is the $defs map key
        # the embedding tool gives it, so no `name` key here.
        return _serialize_object_fields(node.fields, node.description)

    if isinstance(node, Field):
        return _serialize_field(node)

    if isinstance(node, String):
        return _serialize_string(node)

    if isinstance(node, Integer):
        return _serialize_numeric(node, "integer")

    if isinstance(node, Number):
        return _serialize_numeric(node, "number")

    if isinstance(node, Boolean):
        boolean_schema: dict[str, Any] = {"type": "boolean"}
        if node.format is not None:
            boolean_schema["format"] = node.format
        return _canonical(boolean_schema)

    if isinstance(node, Null):
        null_schema: dict[str, Any] = {"type": "null"}
        if node.format is not None:
            null_schema["format"] = node.format
        return _canonical(null_schema)

    if isinstance(node, AnyType):
        # `any` emits the empty schema and NO `type` keyword (§5.2 / §7.2); a
        # description, when present, rides on the Field, producing {"description"}.
        any_schema: dict[str, Any] = {}
        if node.format is not None:
            any_schema["format"] = node.format
        return any_schema

    if isinstance(node, Array):
        return _serialize_array(node)

    if isinstance(node, Object):
        # Object carries no description of its own (that lives on the Field); a
        # stored :format is unusual but emitted rather than dropped (lossless).
        object_schema = _serialize_object_fields(node.fields, None)
        if node.format is not None:
            object_schema["format"] = node.format
            object_schema = _canonical(object_schema)
        return object_schema

    if isinstance(node, Reference):
        ref_schema: dict[str, Any] = {}
        if node.format is not None:
            ref_schema["format"] = node.format
        ref_schema["$ref"] = f"#/$defs/{node.name}"
        return _canonical(ref_schema)

    if isinstance(node, Union):
        return {"anyOf": [to_json(branch) for branch in node.branches]}

    if isinstance(node, Enum):
        return _serialize_enum(node)

    if isinstance(node, Const):
        return {"const": node.value}

    if isinstance(node, RawSchema):
        # §7.5 fallback: the carried JSON Schema is emitted verbatim. Deep-copied
        # so callers cannot mutate the tree through the returned dict, and never
        # reordered (its keyword order is not ours to canonicalize).
        return copy.deepcopy(node.schema)

    raise TypeError(f"cannot serialize node of type {type(node).__name__}")


def to_json_string(node: Node) -> str:
    """Serialize a node to a pretty-printed JSON string (insertion order kept).

    Thin wrapper over to_json(); uses indent=2 and never sort_keys, so the §7.2
    keyword order built into the dict survives into the text.
    """
    return json.dumps(to_json(node), indent=2)
