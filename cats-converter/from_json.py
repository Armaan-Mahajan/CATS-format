"""
from_json.py — the JSON Schema (draft 2020-12) -> CATS AST reader (reverse direction).

    JSON Schema  --[from_json]-->  AST  --[to_json]-->  JSON Schema

This is the inverse of to_json.py. It builds the nodes.py AST that the forward
serializer consumes, mirroring the §7.2.1 output envelope:

  - a LIST of tool schemas        -> a `Document`   (the multi-tool envelope)
  - a single tool schema (a dict
    carrying a top-level "name")   -> a `ToolBlock`  (or a one-tool `Document`
                                      when it embeds a "$defs")
  - any other dict (a bare
    subschema, e.g. {"type":...})  -> a single type node

THE FALLBACK BOUNDARY (decided empirically against the round-trip oracle).
A construct is IN-SCOPE only if it round-trips behaviorally clean. The forward
serializer (to_json) is mechanical and lossless for the nodes it emits, so the
reader encodes exactly the constructs it can reconstruct WITHOUT changing which
values validate. Everything else takes the §7.5 fallback path: the WHOLE tool
containing it becomes one `RawSchema` carrying the original schema verbatim
(raw in -> RawSchema -> raw out, behaviorally identical). Fallback is
all-or-nothing per tool — never inline raw JSON inside an otherwise-CATS tool.

DELIBERATE DEVIATION FROM §7.1 (flagged for review). §7.1/§7.4 prescribe some
BEHAVIOR-CHANGING forward conversions: open objects normalize to closed,
`oneOf` widening to `anyOf`, `propertyNames` dropped without fallback. Those
change the accepted value set, so the oracle would (correctly) reject them.
Because this reader's contract is behavior preservation, it does NOT perform
them — it falls back instead. `oneOf` is in `_FALLBACK_KEYWORDS` (§5.5/§7.4);
so do open objects and `propertyNames` — each triggers whole-tool fallback
rather than a lossy re-encoding.

IN-SCOPE (buckets 1-2 of §7.1):
  primitives (string/integer/number/boolean/null) and their constraints
  (minLength/maxLength, pattern, minimum/maximum + exclusive*, multipleOf,
  format, contentEncoding/contentMediaType); arrays (items, minItems/maxItems,
  uniqueItems); CLOSED objects (properties + required + additionalProperties
  false); `$ref` -> Reference; `anyOf` -> Union; `enum` (+type) -> Enum;
  `const`/single-value enum -> Const; the true/false set -> Boolean; bare
  `array` -> bare Array; the empty schema {} -> AnyType; and a simple
  `type: [..]` array -> Union (no sibling constraints).

NORMALIZED INPUTS (§7.6):
  Provider tool-definition envelopes are unwrapped to their inner parameter
  schema before processing (see `_unwrap_provider_envelope`): OpenAI Responses /
  Gemini (`parameters`, flat), OpenAI Chat Completions (`function.parameters`,
  nested), and Anthropic (`input_schema`). The envelope's name/description are
  lifted onto the tool; runtime flags (`strict`, `cache_control`, `x-*`) are
  dropped per §8.3.

  Legacy `definitions` blocks and `#/definitions/Name` $ref pointers are renamed
  to `$defs` / `#/$defs/Name` before schema processing (`_normalize_legacy_definitions`).
  If both `definitions` and `$defs` appear on the same object, the tool falls back.

  OpenAPI `nullable: true` is rewritten to a `null` type branch (§7.6,
  `_normalize_nullable`): `{"type": "X", "nullable": true}` becomes
  `{"type": ["X", "null"]}` when `type` is the only meaningful keyword, otherwise
  `anyOf` with a `{type: null}` branch.

Validation-inert metadata (§7.1, §8.2): `examples`, `deprecated`, `optional`, and OpenAPI
  singular `example` are silently dropped when the author does not fold them into
  description prose — they carry no validation semantics. Normalizing singular
  `example` into a one-element `examples` array is not implemented; dropping is
  the intentional disposition.

DEFERRED (bucket 3, §7.6 OpenAPI normalization) — NOT yet implemented, flagged:
  (none beyond the drops above)

  Legacy `definitions` -> `$defs` (+ `#/definitions/` $ref rewrite) IS implemented
  in `_normalize_legacy_definitions` (§7.6); both keys on the same object -> fallback.

FALLBACK REPORTING. `from_json_with_report` returns a `ConversionResult` whose
`fallbacks` list records every fallen-back tool and why — the fallback rate is a
headline eval metric. `from_json` is the thin AST-only entry the oracle calls.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass, field
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

JsonSchema = dict[str, Any]


# ---------------------------------------------------------------------------
# Fallback signalling and reporting
# ---------------------------------------------------------------------------

class _Unencodable(Exception):
    """Internal signal: this (sub)schema has no behavior-preserving CATS form.

    Raised wherever the reader meets a construct outside the in-scope set. It
    propagates up to the nearest TOOL (or the document root for a bare schema),
    where it is caught and turned into a whole-tool RawSchema fallback.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class FallbackRecord:
    """One tool that fell back to raw JSON Schema, and why (headline metric)."""

    location: Optional[str]   # tool name, or None for a bare/root schema
    reason: str


@dataclass(frozen=True)
class AssumedClosedRecord:
    """One object schema read as closed via the ``assume_closed`` option (§7.7.1)."""

    location: str


@dataclass
class PythonTypeRenameReport:
    """Counts and locations for ``map_python_types`` renames (§7.7.2)."""

    float_count: int = 0
    dict_count: int = 0
    tuple_count: int = 0
    any_count: int = 0
    touched_locations: list[str] = field(default_factory=list)


_PYTHON_TYPE_ALIASES: dict[str, str] = {
    "float": "number",
    "dict": "object",
    "tuple": "array",
}


@dataclass
class ConversionResult:
    """The AST plus the list of tools that fell back during conversion."""

    ast: Node
    fallbacks: list[FallbackRecord] = field(default_factory=list)
    assumed_closed: list[AssumedClosedRecord] = field(default_factory=list)
    python_type_renames: PythonTypeRenameReport = field(
        default_factory=PythonTypeRenameReport
    )

    @property
    def fallback_count(self) -> int:
        return len(self.fallbacks)


# ---------------------------------------------------------------------------
# Keyword classification
# ---------------------------------------------------------------------------

# Validation-inert metadata: safe to drop on input without changing which values
# validate. (`description`/`default` are additionally lifted onto a Field where
# one encloses the schema; here, at the type level, they are simply ignored.)
_IGNORABLE: frozenset[str] = frozenset({
    "title", "description", "default", "examples", "example", "deprecated",
    "readOnly", "writeOnly", "optional",
    "$schema", "$id", "$comment", "$vocabulary",
})

# Constructs with no behavior-preserving CATS form (§8.1) — and the §7.1
# behavior-changing conversions this reader declines to make (oneOf,
# propertyNames). Any of these in a (sub)schema fails the containing tool over.
# Reserved `format` spellings that collide with CATS named annotations (§6.1).
_RESERVED_FORMAT_VALUES: frozenset[str] = frozenset({
    "length", "regex", "encoding", "media", "unique",
})

_FALLBACK_KEYWORDS: frozenset[str] = frozenset({
    "allOf", "oneOf", "not", "if", "then", "else",
    "contains", "minContains", "maxContains",
    "prefixItems", "additionalItems", "unevaluatedItems",
    "dependentRequired", "dependentSchemas",
    "minProperties", "maxProperties", "propertyNames",
    "patternProperties", "unevaluatedProperties",
    "contentSchema", "$anchor", "$dynamicAnchor", "$dynamicRef", "discriminator",
})

# Enum sibling constraints the converter can test exactly against literal
# members (§5.6): drop when every member satisfies, else fall back.
_MEMBER_CHECKABLE: frozenset[str] = frozenset({
    "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
})


def _is_number(value: Any) -> bool:
    """A JSON number — int or float but NOT bool (Python bool subclasses int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _meaningful_keys(schema: JsonSchema) -> set[str]:
    """The schema's keys minus validation-inert metadata (§8.2)."""
    return set(schema) - _IGNORABLE


# ---------------------------------------------------------------------------
# Type-level reader (raises _Unencodable on any out-of-scope construct)
# ---------------------------------------------------------------------------

def _read_type(schema: Any) -> Node:
    """One JSON Schema (sub)schema -> one CATS type node, or raise _Unencodable."""
    if isinstance(schema, bool):
        # `true`/`false` boolean schemas: universal accept/reject, no structure (§8.1).
        raise _Unencodable("boolean schema (true/false) has no CATS form (§8.1)")
    if not isinstance(schema, dict):
        raise _Unencodable(f"schema is not an object: {type(schema).__name__}")

    keys = _meaningful_keys(schema)

    bad = keys & _FALLBACK_KEYWORDS
    if bad:
        raise _Unencodable(f"unencodable keyword(s) {sorted(bad)} (§7.4/§8.1)")

    if "$ref" in keys:
        return _read_ref(schema, keys)
    if "const" in keys:
        return _read_const(schema, keys)
    if "enum" in keys:
        return _read_enum(schema, keys)
    if "anyOf" in keys:
        return _read_anyof(schema, keys)
    if "type" in keys:
        return _read_typed(schema, keys)
    if not keys:
        # {} or a description-only schema -> the empty schema (§5.2).
        return AnyType()
    raise _Unencodable(
        f"typeless schema with keyword(s) {sorted(keys)}; type is not inferred (§7.4)"
    )


def _read_ref(schema: JsonSchema, keys: set[str]) -> Reference:
    if keys != {"$ref"}:
        raise _Unencodable("$ref combined with sibling keywords is not encoded (§5.7)")
    ref = schema["$ref"]
    prefix = "#/$defs/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise _Unencodable(f"only #/$defs/Name references are encoded, not {ref!r} (§5.7/§8.1)")
    name = ref[len(prefix):]
    if not name.isidentifier():
        raise _Unencodable(f"reference target {name!r} is not a bare identifier (§5.7)")
    return Reference(name=name)


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _value_has_json_type(value: Any, type_name: str) -> bool:
    name = _json_type_name(value)
    if type_name == "number":
        return name in ("number", "integer")
    if type_name == "integer":
        return name == "integer" or (isinstance(value, float) and value.is_integer())
    return name == type_name


def _read_const(schema: JsonSchema, keys: set[str]) -> Const:
    extra = keys - {"const", "type"}
    if extra:
        raise _Unencodable(f"const with sibling constraint(s) {sorted(extra)} (§5.6)")
    value = schema["const"]
    declared = schema.get("type")
    if isinstance(declared, str) and not _value_has_json_type(value, declared):
        raise _Unencodable("const value contradicts its declared type")
    return Const(value=value)


def _member_satisfies(value: Any, key: str, bound: Any) -> bool:
    """Does one literal enum member satisfy a member-checkable constraint (§5.6)?

    A constraint that does not apply to the member's JSON type is vacuously
    satisfied (e.g. `minLength` says nothing about an integer member).
    """
    if key == "minLength":
        return not isinstance(value, str) or len(value) >= bound
    if key == "maxLength":
        return not isinstance(value, str) or len(value) <= bound
    if not _is_number(value):
        return True
    if key == "minimum":
        return value >= bound
    if key == "maximum":
        return value <= bound
    if key == "exclusiveMinimum":
        return value > bound
    if key == "exclusiveMaximum":
        return value < bound
    if key == "multipleOf":
        try:
            return (value % bound) == 0
        except ZeroDivisionError:
            return False
    return False


def _read_format(schema: JsonSchema) -> Optional[str]:
    """Read `format`, falling back when it collides with a reserved annotation (§6.1)."""
    fmt = schema.get("format")
    if isinstance(fmt, str) and fmt in _RESERVED_FORMAT_VALUES:
        raise _Unencodable(
            "format value collides with a reserved annotation keyword (§6.1)"
        )
    return fmt if isinstance(fmt, str) else None


def _infer_base_type(values: list[Any]) -> Optional[str]:
    """The enum's inferred primitive type for §7.2 emission, or None if mixed."""
    if values and all(isinstance(v, bool) for v in values):
        return "boolean"
    if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "integer"
    if values and all(_is_number(v) for v in values):
        return "number"
    if values and all(isinstance(v, str) for v in values):
        return "string"
    return None


def _read_enum(schema: JsonSchema, keys: set[str]) -> Node:
    values = schema["enum"]
    if not isinstance(values, list) or not values:
        raise _Unencodable("enum must be a non-empty list")

    sibling = keys - {"enum", "type"}
    for key in list(sibling):
        if key in _MEMBER_CHECKABLE:
            bound = schema[key]
            if all(_member_satisfies(v, key, bound) for v in values):
                sibling.discard(key)  # provably redundant -> drop (§5.6, keyword-lossy)
            else:
                raise _Unencodable(
                    f"enum member fails '{key}'; dropping it would widen the value set (§5.6)"
                )
    if sibling:
        raise _Unencodable(
            f"enum with non-member-checkable constraint(s) {sorted(sibling)} (§5.6)"
        )

    if len(values) == 2 and all(isinstance(v, bool) for v in values) and set(values) == {True, False}:
        return Boolean()  # true/false set canonicalizes to boolean (§5.6)
    if len(values) == 1:
        return Const(value=values[0])  # single-value enum -> const (§5.6/§7.2)

    declared = schema.get("type")
    base = declared if isinstance(declared, str) else _infer_base_type(values)
    if base is None:
        raise _Unencodable(
            "mixed-type enum: members span more than one JSON type (§5.6/§8.1)"
        )
    return Enum(values=list(values), base_type=base)


def _read_anyof(schema: JsonSchema, keys: set[str]) -> Union:
    if keys != {"anyOf"}:
        raise _Unencodable("anyOf combined with sibling keywords is not encoded (§5.5)")
    branches = schema["anyOf"]
    if not isinstance(branches, list) or not branches:
        raise _Unencodable("anyOf must be a non-empty list")
    return Union(branches=[_read_type(b) for b in branches])


def _read_typed(schema: JsonSchema, keys: set[str]) -> Node:
    declared = schema["type"]
    if isinstance(declared, list):
        return _read_type_array(schema, keys, declared)
    if declared == "string":
        return _read_string(schema, keys)
    if declared in ("integer", "number"):
        return _read_numeric(schema, keys, declared)
    if declared == "boolean":
        extra = keys - {"type", "format"}
        if extra:
            raise _Unencodable(f"boolean with sibling constraint(s) {sorted(extra)}")
        return Boolean(format=_read_format(schema))
    if declared == "null":
        extra = keys - {"type", "format"}
        if extra:
            raise _Unencodable(f"null with sibling constraint(s) {sorted(extra)}")
        return Null(format=_read_format(schema))
    if declared == "object":
        return _read_object(schema, keys)
    if declared == "array":
        return _read_array(schema, keys)
    raise _Unencodable(f"unknown type {declared!r}")


def _read_type_array(schema: JsonSchema, keys: set[str], types: list[Any]) -> Node:
    if keys != {"type"}:
        raise _Unencodable("type-array combined with sibling constraints is not encoded")
    if not types:
        raise _Unencodable("empty type array")
    if len(types) == 1:
        return _read_type({"type": types[0]})
    return Union(branches=[_read_type({"type": t}) for t in types])


def _read_string(schema: JsonSchema, keys: set[str]) -> String:
    allowed = {"type", "format", "minLength", "maxLength", "pattern",
               "contentEncoding", "contentMediaType"}
    extra = keys - allowed
    if extra:
        raise _Unencodable(f"string with unencodable keyword(s) {sorted(extra)}")
    return String(
        format=_read_format(schema),
        min_length=schema.get("minLength"),
        max_length=schema.get("maxLength"),
        pattern=schema.get("pattern"),
        encoding=schema.get("contentEncoding"),
        media=schema.get("contentMediaType"),
    )


def _read_numeric_bound(schema: JsonSchema, key: str) -> float:
    """Read one numeric bound keyword; reject non-JSON-number values (§6.2)."""
    value = schema[key]
    if not _is_number(value):
        raise _Unencodable("non-numeric bound value (§6.2)")
    return value


def _read_multiple_of(schema: JsonSchema) -> float:
    """Read `multipleOf`; reject non-numbers and non-positive values (§6.2)."""
    value = schema["multipleOf"]
    if not _is_number(value) or value <= 0:
        raise _Unencodable("multipleOf must be a positive number (§6.2)")
    return value


def _read_numeric(schema: JsonSchema, keys: set[str], type_name: str) -> Node:
    allowed = {"type", "format", "minimum", "maximum",
               "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
    extra = keys - allowed
    if extra:
        raise _Unencodable(f"{type_name} with unencodable keyword(s) {sorted(extra)}")

    minimum: Optional[float] = None
    exclusive_min = False
    if "minimum" in keys and "exclusiveMinimum" in keys:
        raise _Unencodable("both minimum and exclusiveMinimum present (no single-bound CATS form)")
    if "exclusiveMinimum" in keys:
        minimum, exclusive_min = _read_numeric_bound(schema, "exclusiveMinimum"), True
    elif "minimum" in keys:
        minimum = _read_numeric_bound(schema, "minimum")

    maximum: Optional[float] = None
    exclusive_max = False
    if "maximum" in keys and "exclusiveMaximum" in keys:
        raise _Unencodable("both maximum and exclusiveMaximum present (no single-bound CATS form)")
    if "exclusiveMaximum" in keys:
        maximum, exclusive_max = _read_numeric_bound(schema, "exclusiveMaximum"), True
    elif "maximum" in keys:
        maximum = _read_numeric_bound(schema, "maximum")

    multiple_of: Optional[float] = None
    if "multipleOf" in keys:
        multiple_of = _read_multiple_of(schema)

    cls = Integer if type_name == "integer" else Number
    return cls(
        format=_read_format(schema),
        minimum=minimum,
        maximum=maximum,
        exclusive_min=exclusive_min,
        exclusive_max=exclusive_max,
        multiple_of=multiple_of,
    )


def _read_array(schema: JsonSchema, keys: set[str]) -> Array:
    allowed = {"type", "format", "items", "minItems", "maxItems", "uniqueItems"}
    extra = keys - allowed
    if extra:
        raise _Unencodable(f"array with unencodable keyword(s) {sorted(extra)}")
    element: Optional[Node] = None
    if "items" in schema:
        items = schema["items"]
        if isinstance(items, list):
            # A list `items` is the pre-2020-12 tuple form (§8.1) — not array<T>.
            raise _Unencodable("tuple-style array `items` list is not encoded (§8.1)")
        element = _read_type(items)
    return Array(
        element=element,
        format=_read_format(schema),
        min_items=schema.get("minItems"),
        max_items=schema.get("maxItems"),
        unique=bool(schema.get("uniqueItems", False)),
    )


def _read_object(schema: JsonSchema, keys: set[str]) -> Object:
    allowed = {"type", "format", "properties", "required", "additionalProperties"}
    extra = keys - allowed
    if extra:
        raise _Unencodable(f"object with unencodable keyword(s) {sorted(extra)}")
    if schema.get("additionalProperties") is not False:
        # Open/typed-open objects would have to be CLOSED to encode, which changes
        # behavior (§7.4); preserve meaning by falling back instead.
        raise _Unencodable("open object (additionalProperties is not false) (§7.4)")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise _Unencodable("`properties` must be an object")
    required = schema.get("required", [])
    required_set = set(required) if isinstance(required, list) else set()

    fields = [
        _read_field(name, prop, name in required_set)
        for name, prop in properties.items()
    ]
    return Object(fields=fields, format=_read_format(schema))


def _read_field(name: str, prop: Any, required: bool) -> Field:
    default: Any = NO_DEFAULT
    description: Optional[str] = None
    if isinstance(prop, dict):
        if "default" in prop:
            default = prop["default"]
        raw_desc = prop.get("description")
        if isinstance(raw_desc, str):
            description = raw_desc
    type_node = _read_type(prop)
    # §4.1: a required field must not also carry a default. The default is
    # validation-inert, so dropping it keeps the AST legal and meaning intact.
    if required and default is not NO_DEFAULT:
        default = NO_DEFAULT
    return Field(name=name, type=type_node, required=required, default=default, description=description)


# ---------------------------------------------------------------------------
# Tool / definition / document readers
# ---------------------------------------------------------------------------

def _read_tool_block(tool_schema: JsonSchema) -> ToolBlock:
    """A single tool schema (dict with "name") -> a ToolBlock, or raise _Unencodable."""
    body = {k: v for k, v in tool_schema.items() if k not in ("name", "$defs")}
    node = _read_type(body)
    if not isinstance(node, Object):
        raise _Unencodable("tool body is not a closed object")
    description = body.get("description") if isinstance(body.get("description"), str) else None

    # Check for shaped objects in positions to_cats cannot serialize (§7.5 fallback)
    for field in node.fields:
        if _has_shaped_object_in_non_encodable_position(field.type, is_field_direct_type=True):
            raise _Unencodable(
                "contains a shaped object in a position to_cats cannot serialize "
                "(nested block only allowed at field level or as array<object> element)"
            )

    return ToolBlock(name=tool_schema["name"], fields=node.fields, description=description)


def _read_definition(name: str, def_schema: Any) -> Definition:
    """One embedded $defs entry -> a Definition, or raise _Unencodable."""
    node = _read_type(def_schema)
    if not isinstance(node, Object):
        raise _Unencodable(f"definition {name!r} is not a closed object")
    description = None
    if isinstance(def_schema, dict) and isinstance(def_schema.get("description"), str):
        description = def_schema["description"]
    return Definition(name=name, fields=node.fields, description=description)


def _read_single_tool(tool_schema: JsonSchema, fallbacks: list[FallbackRecord]) -> Node:
    """A standalone tool schema -> ToolBlock, or a one-tool Document if it has $defs."""
    defs_block = tool_schema.get("$defs") or {}
    try:
        tool = _read_tool_block(tool_schema)
        definitions = [_read_definition(n, s) for n, s in defs_block.items()]
    except _Unencodable as exc:
        fallbacks.append(FallbackRecord(tool_schema.get("name"), exc.reason))
        return RawSchema(schema=copy.deepcopy(tool_schema))
    if defs_block:
        return Document(tools=[tool], defs=definitions)
    return tool


def _read_document(envelope: list[Any], fallbacks: list[FallbackRecord]) -> Document:
    """The list-of-tools envelope -> a Document, reconciling per-tool $defs (§7.2.1).

    Per-tool embedded definitions are merged into one document-level set when
    IDENTICAL across the tools that carry them. A name carried with DIFFERING
    schemas by two tools is a CONFLICT: every tool that uses that name falls back
    to raw JSON Schema (the conservative, behavior-preserving choice — flagged
    for review). A definition that is itself unencodable is treated the same way.
    """
    # Pass 1 — read each tool's block and remember its embedded defs.
    parsed: list[dict[str, Any]] = []
    for item in envelope:
        if isinstance(item, dict):
            item, conflict_reason = _prepare_tool_schema(item)
            if conflict_reason is not None:
                parsed.append({
                    "tool": None,
                    "raw": item,
                    "defs": {},
                    "reason": conflict_reason,
                })
                continue
        if not (isinstance(item, dict) and "name" in item):
            parsed.append({"tool": None, "raw": item,
                           "defs": {}, "reason": "envelope element is not a tool schema"})
            continue
        item_defs = item.get("$defs") or {}
        try:
            tool = _read_tool_block(item)
        except _Unencodable as exc:
            parsed.append({"tool": None, "raw": item, "defs": item_defs, "reason": exc.reason})
            continue
        parsed.append({"tool": tool, "raw": item, "defs": item_defs, "reason": None})

    # Pass 2 — gather definitions across tools; find name conflicts.
    name_to_schemas: dict[str, list[Any]] = defaultdict(list)
    for entry in parsed:
        if entry["tool"] is None:
            continue
        for dname, dschema in entry["defs"].items():
            if dschema not in name_to_schemas[dname]:
                name_to_schemas[dname].append(dschema)
    conflicted = {name for name, schemas in name_to_schemas.items() if len(schemas) > 1}

    definitions: list[Definition] = []
    unreadable: set[str] = set()
    for dname, schemas in name_to_schemas.items():
        if dname in conflicted:
            continue
        try:
            definitions.append(_read_definition(dname, schemas[0]))
        except _Unencodable:
            unreadable.add(dname)
    problem = conflicted | unreadable

    # Pass 3 — assemble; fall back any tool that uses a problem definition.
    tools: list[Node] = []
    for entry in parsed:
        if entry["tool"] is None:
            fallbacks.append(FallbackRecord(_safe_name(entry["raw"]), entry["reason"]))
            tools.append(RawSchema(schema=copy.deepcopy(entry["raw"])))
            continue
        used_problem = set(entry["defs"].keys()) & problem
        if used_problem:
            reason = f"references conflicting/unencodable definition(s) {sorted(used_problem)} across tools"
            fallbacks.append(FallbackRecord(_safe_name(entry["raw"]), reason))
            tools.append(RawSchema(schema=copy.deepcopy(entry["raw"])))
        else:
            tools.append(entry["tool"])

    return Document(tools=tools, defs=definitions or None)


def _safe_name(raw: Any) -> Optional[str]:
    return raw.get("name") if isinstance(raw, dict) else None


# ---------------------------------------------------------------------------
# Provider tool-definition envelope unwrapping (§7.6)
# ---------------------------------------------------------------------------

# The keys that hold the real parameter schema inside a provider wrapper. None
# is a JSON Schema draft 2020-12 keyword, so at the TOP LEVEL they unambiguously
# mark an API wrapper (not a schema, not the CATS output envelope).
_PARAM_SCHEMA_KEYS = ("parameters", "input_schema")


def _unwrap_provider_envelope(schema: Any) -> Optional[JsonSchema]:
    """If `schema` is a known provider tool-definition envelope, return the inner
    parameter schema with the envelope's name/description lifted onto it (and the
    envelope's runtime flags dropped); otherwise return None (§7.6).

    DETECTION RULE: a dict is a provider envelope iff it has NO top-level
    `properties` (a dict with `properties` IS a schema — or the CATS output
    envelope — never a wrapper) AND it carries a wrapper key whose value is an
    object: `function` (OpenAI Chat Completions, nested one level), `parameters`
    (OpenAI Responses / Gemini functionDeclarations, flat), or `input_schema`
    (Anthropic). Those keys are not JSON Schema keywords, so a top-level
    occurrence marks a wrapper; a property literally named one of them lives
    inside `properties`, not at the top level, and is therefore never misread.

    Runtime flags on the envelope (`strict`, `cache_control`, any `x-*` vendor
    extension) are dropped per §8.3 simply by not copying them — only the inner
    parameter schema plus the lifted name/description survive. Dropping them is
    not itself a fallback trigger.
    """
    if not isinstance(schema, dict):
        return None
    if "properties" in schema:
        return None

    # OpenAI Chat Completions nests the tool under "function"; unwrap one level.
    envelope = schema["function"] if isinstance(schema.get("function"), dict) else schema

    for key in _PARAM_SCHEMA_KEYS:
        inner = envelope.get(key)
        if isinstance(inner, dict):
            normalized = dict(inner)  # the real parameter schema (shallow copy)
            name = envelope.get("name")
            if name is not None:
                normalized["name"] = name
            description = envelope.get("description")
            if isinstance(description, str):
                normalized["description"] = description  # envelope's name/desc win
            return normalized
    return None


# ---------------------------------------------------------------------------
# Legacy `definitions` -> `$defs` normalization (§7.6)
# ---------------------------------------------------------------------------

_DEFINITIONS_REF_PREFIX = "#/definitions/"
_DEFS_REF_PREFIX = "#/$defs/"

_DEFINITIONS_CONFLICT_REASON = (
    "definitions and $defs both present on the same schema object (§7.6)"
)


def _normalize_legacy_definitions(obj: Any) -> tuple[Any, bool]:
    """Rename `definitions` to `$defs` and rewrite `#/definitions/` $ref pointers.

    Returns ``(normalized, conflict)``. When any dict carries both ``definitions``
    and ``$defs``, ``conflict`` is True and the returned tree is a deep copy of
    the input (unchanged) so the tool can fall back without data loss.
    """
    if isinstance(obj, dict):
        if "definitions" in obj and "$defs" in obj:
            return copy.deepcopy(obj), True
        conflict = False
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "definitions":
                norm_value, child_conflict = _normalize_legacy_definitions(value)
                out["$defs"] = norm_value
                conflict = conflict or child_conflict
            elif key == "$ref" and isinstance(value, str) and value.startswith(_DEFINITIONS_REF_PREFIX):
                out["$ref"] = _DEFS_REF_PREFIX + value[len(_DEFINITIONS_REF_PREFIX):]
            else:
                norm_value, child_conflict = _normalize_legacy_definitions(value)
                out[key] = norm_value
                conflict = conflict or child_conflict
        return out, conflict
    if isinstance(obj, list):
        items: list[Any] = []
        conflict = False
        for item in obj:
            norm_item, child_conflict = _normalize_legacy_definitions(item)
            items.append(norm_item)
            conflict = conflict or child_conflict
        return items, conflict
    return obj, False


def _nullable_to_union(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite one schema object's OpenAPI ``nullable: true`` to a null branch (§7.6)."""
    type_val = schema.get("type")
    meaningful = _meaningful_keys(schema)

    if meaningful == {"type"}:
        if isinstance(type_val, str):
            if type_val == "null":
                return schema
            out = dict(schema)
            out["type"] = [type_val, "null"]
            return out
        if isinstance(type_val, list):
            if "null" in type_val:
                return schema
            out = dict(schema)
            out["type"] = [*type_val, "null"]
            return out

    if meaningful == {"$ref"}:
        return {"anyOf": [{"$ref": schema["$ref"]}, {"type": "null"}]}

    return {"anyOf": [schema, {"type": "null"}]}


def _normalize_nullable(obj: Any) -> Any:
    """Recursively rewrite OpenAPI ``nullable: true`` into a JSON Schema null branch."""
    if isinstance(obj, dict):
        nullable = obj.get("nullable")
        children: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "nullable":
                continue
            children[key] = _normalize_nullable(value)
        if nullable is False:
            return children
        if nullable is True:
            return _nullable_to_union(children)
        return children
    if isinstance(obj, list):
        return [_normalize_nullable(item) for item in obj]
    return obj


def _prepare_tool_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    """§7.6 pre-pass on one tool-shaped dict: unwrap, definitions, then nullable."""
    unwrapped = _unwrap_provider_envelope(schema)
    if unwrapped is not None:
        schema = unwrapped
    normalized, conflict = _normalize_legacy_definitions(schema)
    if conflict:
        return copy.deepcopy(schema), _DEFINITIONS_CONFLICT_REASON
    return _normalize_nullable(normalized), None


# ---------------------------------------------------------------------------
# Shaped object position detection (to_cats serialization constraint)
# ---------------------------------------------------------------------------

def _has_shaped_object_in_non_encodable_position(
    type_node: Node,
    is_field_direct_type: bool = False,
    is_array_element_of_field: bool = False,
) -> bool:
    """Check if type_node contains a shaped Object in a non-encodable position.

    A shaped Object (Object with ≥1 field) is encodable only at:
      1. Direct field type: `field object { nested block }`
      2. Direct array element at field level: `field array<object> { nested block }`

    Any other position (Union branch with object, array<array<object>>, etc.)
    loses the object's fields on serialization — this is a behavior change, so
    those tools must fall back to raw JSON Schema (§7.5).

    Bare objects (no fields, validates only {}) are fine anywhere.
    """
    if isinstance(type_node, Object):
        if type_node.fields:  # shaped object
            if not (is_field_direct_type or is_array_element_of_field):
                # Shaped object in non-encodable position -> should fall back
                return True
            # This object is OK; check its nested fields for deeper non-encodable shapes
            for field in type_node.fields:
                if _has_shaped_object_in_non_encodable_position(
                    field.type, is_field_direct_type=True, is_array_element_of_field=False
                ):
                    return True
        return False

    if isinstance(type_node, Array):
        if type_node.element is None:
            return False
        # Recurse into element; it's encodable only if the array itself is at field level
        return _has_shaped_object_in_non_encodable_position(
            type_node.element,
            is_field_direct_type=False,
            is_array_element_of_field=is_field_direct_type,
        )

    if isinstance(type_node, Union):
        # Check all branches; none can have shaped objects (Union can't have nested blocks)
        for branch in type_node.branches:
            if _has_shaped_object_in_non_encodable_position(
                branch, is_field_direct_type=False, is_array_element_of_field=False
            ):
                return True
        return False

    # Other types (String, Integer, Reference, etc.)
    return False


# ---------------------------------------------------------------------------
# Schema input loading (JSON text vs Python dict)
# ---------------------------------------------------------------------------

# JSON Schema keys whose values are booleans (not the type word "null" in a
# `type` array, not enum members, etc.).
_BOOLEAN_SCHEMA_KEYS = frozenset({
    "additionalProperties",
    "uniqueItems",
    "readOnly",
    "writeOnly",
    "deprecated",
    "nullable",  # OpenAPI 3.0; rewritten to a null branch in _normalize_nullable (§7.6)
    "unique",    # rare alias; harmless if seen
})

# Keys whose value may be JSON null as a real default/const (not the type name).
_NULL_VALUE_KEYS = frozenset({"const", "default"})

# Lists where string members are schema vocabulary, never coerced to bool/null.
_LITERAL_STRING_LIST_KEYS = frozenset({"type", "enum", "required"})


def _normalize_json_literals(obj: Any, parent_key: Optional[str] = None) -> Any:
    """Recursively coerce hand-typed JSON spellings on known boolean/null slots.

    After `json.loads`, booleans are already `True`/`False`. When a schema is a
    Python dict built by hand, `"false"` on `additionalProperties` becomes
    `False`. The type word `"null"` in `{"type": ["string", "null"]}` is left
    as a string — coercing it to Python `None` would break nullable unions.

    Does NOT fix bare `false` in Python source (a syntax error); use `load_schema`
    with JSON text for that.
    """
    if isinstance(obj, dict):
        return {
            key: _normalize_json_literals(value, key) for key, value in obj.items()
        }
    if isinstance(obj, list):
        if parent_key in _LITERAL_STRING_LIST_KEYS:
            return list(obj)
        return [_normalize_json_literals(item, parent_key) for item in obj]
    if isinstance(obj, str) and parent_key is not None:
        lower = obj.casefold()
        if parent_key in _BOOLEAN_SCHEMA_KEYS:
            if lower == "true":
                return True
            if lower == "false":
                return False
        if parent_key in _NULL_VALUE_KEYS and lower == "null":
            return None
    return obj


def load_schema(schema: str | dict[str, Any] | list[Any]) -> Any:
    """Accept schema input as JSON text or as an already-parsed dict/list.

    JSON text may use lowercase `true`, `false`, and `null` (RFC 8259). Python
    dict literals must use `True`, `False`, and `None` instead — or pass a JSON
    string here and avoid that mismatch entirely.
    """
    if isinstance(schema, str):
        parsed = json.loads(schema)
    else:
        parsed = schema
    return _normalize_json_literals(parsed)


# ---------------------------------------------------------------------------
# Opt-in input normalizations (§7.7)
# ---------------------------------------------------------------------------

def _is_object_typed_schema(schema: dict[str, Any]) -> bool:
    declared = schema.get("type")
    if declared == "object":
        return True
    if isinstance(declared, list) and "object" in declared:
        return True
    return False


def _visit_schema_subtrees(schema: Any, path: str, visit: Any) -> None:
    """Walk every subschema dict in a tool envelope or bare schema tree.

    Must enumerate the same subschema positions ``from_json`` treats as live
    (parameters, properties, items, $defs, anyOf branches, etc.). Any new schema
    shape the encoder recurses into must be mirrored here so §7.7 normalizations
    reach it.
    """
    if isinstance(schema, list):
        for index, item in enumerate(schema):
            _visit_schema_subtrees(item, f"{path}/{index}", visit)
        return
    if not isinstance(schema, dict):
        return

    visit(schema, path)

    for key, value in schema.items():
        if key in ("parameters", "input_schema") and isinstance(value, dict):
            _visit_schema_subtrees(value, f"{path}/{key}", visit)
        elif key == "function" and isinstance(value, dict):
            _visit_schema_subtrees(value, f"{path}/function", visit)
        elif key in ("$defs", "definitions") and isinstance(value, dict):
            for name, sub in value.items():
                _visit_schema_subtrees(sub, f"{path}/{key}/{name}", visit)
        elif key == "properties" and isinstance(value, dict):
            for name, sub in value.items():
                _visit_schema_subtrees(sub, f"{path}/properties/{name}", visit)
        elif key == "patternProperties" and isinstance(value, dict):
            for name, sub in value.items():
                _visit_schema_subtrees(
                    sub, f"{path}/patternProperties/{name}", visit
                )
        elif key in ("items", "not", "if", "then", "else") and isinstance(value, dict):
            _visit_schema_subtrees(value, f"{path}/{key}", visit)
        elif key == "additionalProperties" and isinstance(value, dict):
            _visit_schema_subtrees(value, f"{path}/additionalProperties", visit)
        elif key in ("anyOf", "oneOf", "allOf", "prefixItems") and isinstance(
            value, list
        ):
            for index, sub in enumerate(value):
                if isinstance(sub, dict):
                    _visit_schema_subtrees(sub, f"{path}/{key}/{index}", visit)
        elif key == "items" and isinstance(value, list):
            for index, sub in enumerate(value):
                if isinstance(sub, dict):
                    _visit_schema_subtrees(sub, f"{path}/items/{index}", visit)


def _apply_map_python_types(
    schema: Any, report: PythonTypeRenameReport
) -> None:
    """Rename the four Python ``type`` aliases in place (§7.7.2)."""

    def visit(node: dict[str, Any], path: str) -> None:
        if "type" not in node:
            return
        declared = node["type"]
        if not isinstance(declared, str):
            return
        location = path or "/"
        if declared == "any":
            del node["type"]
            report.any_count += 1
            report.touched_locations.append(location)
            return
        if declared not in _PYTHON_TYPE_ALIASES:
            return
        node["type"] = _PYTHON_TYPE_ALIASES[declared]
        if declared == "float":
            report.float_count += 1
        elif declared == "dict":
            report.dict_count += 1
        else:
            report.tuple_count += 1
        report.touched_locations.append(location)

    _visit_schema_subtrees(schema, "", visit)


def _apply_assume_closed(
    schema: Any, records: list[AssumedClosedRecord]
) -> None:
    """Treat omitted ``additionalProperties`` on object schemas as false (§7.7.1)."""

    def visit(node: dict[str, Any], path: str) -> None:
        if not _is_object_typed_schema(node):
            return
        if "additionalProperties" in node:
            return
        node["additionalProperties"] = False
        records.append(AssumedClosedRecord(location=path or "/"))

    _visit_schema_subtrees(schema, "", visit)


def _apply_input_normalizations(
    schema: Any,
    *,
    assume_closed: bool,
    map_python_types: bool,
) -> tuple[Any, list[AssumedClosedRecord], PythonTypeRenameReport]:
    """Apply §7.7 options in spec order: ``map_python_types`` then ``assume_closed``."""
    if not assume_closed and not map_python_types:
        return schema, [], PythonTypeRenameReport()

    normalized = copy.deepcopy(schema)
    rename_report = PythonTypeRenameReport()
    assumed_closed_records: list[AssumedClosedRecord] = []

    if map_python_types:
        _apply_map_python_types(normalized, rename_report)
    if assume_closed:
        _apply_assume_closed(normalized, assumed_closed_records)

    return normalized, assumed_closed_records, rename_report


def normalize_map_python_types(schema: Any) -> tuple[Any, PythonTypeRenameReport]:
    """Apply §7.7.2 only; return a deep copy and per-alias rename stats.

    Used by the Part 1 eval to validate/dedupe on the same normalized bytes the
    converter will read when ``map_python_types=True``.
    """
    normalized, _, report = _apply_input_normalizations(
        schema,
        assume_closed=False,
        map_python_types=True,
    )
    return normalized, report


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def from_json_with_report(
    schema: str | dict[str, Any] | list[Any],
    *,
    assume_closed: bool = False,
    map_python_types: bool = False,
) -> ConversionResult:
    """Read a JSON Schema input into a CATS AST, recording every tool fallback.

    `schema` may be a JSON string (with `true`/`false`/`null`) or a parsed dict or
    list. Dispatches on the envelope shape (§7.2.1): a list -> Document; a dict
    with a top-level "name" -> a single tool; any other dict/bool -> a bare type
    node. Out-of-scope constructs fall back per tool to a RawSchema.
    """
    schema = load_schema(schema)
    schema, assumed_closed, python_type_renames = _apply_input_normalizations(
        schema,
        assume_closed=assume_closed,
        map_python_types=map_python_types,
    )
    fallbacks: list[FallbackRecord] = []

    if isinstance(schema, list):
        ast: Node = _read_document(schema, fallbacks)
    elif isinstance(schema, dict):
        prepared, conflict_reason = _prepare_tool_schema(schema)
        if conflict_reason is not None:
            fallbacks.append(FallbackRecord(prepared.get("name"), conflict_reason))
            ast = RawSchema(schema=prepared)
        elif "name" in prepared:
            ast = _read_single_tool(prepared, fallbacks)
        else:
            try:
                ast = _read_type(prepared)
            except _Unencodable as exc:
                fallbacks.append(FallbackRecord(None, exc.reason))
                ast = RawSchema(schema=copy.deepcopy(prepared))
    else:
        try:
            ast = _read_type(schema)
        except _Unencodable as exc:
            fallbacks.append(FallbackRecord(None, exc.reason))
            ast = RawSchema(schema=copy.deepcopy(schema))
    return ConversionResult(
        ast=ast,
        fallbacks=fallbacks,
        assumed_closed=assumed_closed,
        python_type_renames=python_type_renames,
    )


def from_json(
    schema: str | dict[str, Any] | list[Any],
    *,
    assume_closed: bool = False,
    map_python_types: bool = False,
) -> Node:
    """Read a JSON Schema (draft 2020-12) input into a CATS AST node.

    The AST-only entry point (the round-trip oracle calls this). Use
    `from_json_with_report` when the per-tool fallback list/count is needed.
    """
    return from_json_with_report(
        schema,
        assume_closed=assume_closed,
        map_python_types=map_python_types,
    ).ast
