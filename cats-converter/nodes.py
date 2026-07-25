"""
nodes.py — Abstract Syntax Tree node definitions for the CATS converter.

This module defines the in-memory tree that both conversion directions share:

    JSON Schema text  --(from_json)-->  AST  --(to_json)-->  JSON Schema text
    CATS text         --(lexer+parser)->  AST  --(to_cats)-->  CATS text

The AST is the single contract every other module signs. The lexer/parser and
the JSON-Schema reader BUILD these nodes; the serializers READ them. Nothing
else in the project should invent its own representation of a schema.

DESIGN DECISIONS baked into this file (see conversation/spec for rationale):

  1. Annotations (spec section 6) are modeled as NAMED FIELDS on each type node,
     NOT as a generic annotation list. Each type carries ONLY the annotations
     that section 6.5 permits for it. This makes illegal states (e.g. a numeric
     bound on a string) impossible to represent rather than merely catchable.

  2. `integer` and `number` are SEPARATE node classes. Spec section 2.5 fixes the
     integer/number distinction at the type slot, so the tree carries it rather
     than re-deriving it later.

  3. A single shared empty base class `Node` lets other modules type-hint
     "some AST node" without caring which one.

  4. Defaults distinguish "no default present" from "default is literally null".
     `None` is a legal default value (null), so absence is marked by the
     `NO_DEFAULT` sentinel instead of `None`.

  5. `RawSchema` is the fallback escape hatch (spec section 7.5). Any construct
     with no CATS form (allOf, not, discriminated unions, etc., per section 8)
     routes into a RawSchema node that carries the original JSON verbatim.

Each class notes the spec section(s) it derives from.

This module is intentionally pure data: no parsing, no serialization, no
validation logic lives here. Those belong in parser.py, to_json.py /
from_json.py, and validate.py respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Sentinel for "no default value present"
# ---------------------------------------------------------------------------
# A field's default (spec 4.5) may legitimately be the JSON value null, which we
# represent in Python as None. We therefore CANNOT use None to mean "this field
# has no default". This unique sentinel object fills that role: `default is
# NO_DEFAULT` means absent, anything else (including None) means a real default.

class _NoDefault:
    """Unique marker meaning 'no default was specified'. Do not instantiate
    elsewhere; use the singleton NO_DEFAULT below."""
    _instance: Optional["_NoDefault"] = None

    def __new__(cls) -> "_NoDefault":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NO_DEFAULT"

    def __bool__(self) -> bool:
        return False


NO_DEFAULT = _NoDefault()

# A default value, once present, is an opaque JSON value: a scalar (str / int /
# float / bool / None) or a parsed JSON object/array (dict / list), per spec 4.5.
# We do not model its internal structure — JSON's own syntax governs it.
DefaultValue = Any


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """Empty base class for every AST node.

    Carries no data. Exists so other modules can write `x: Node` to mean
    'any node in the tree' and so isinstance checks have a common root.
    """
    pass


# A type expression slot (spec 5.1) is always filled by one of the type nodes
# defined below. This alias documents intent wherever a "type node" is expected.
TypeNode = Node


# ===========================================================================
# GROUP 1 — STRUCTURAL NODES  (the skeleton: spec sections 3 and 4)
# ===========================================================================

@dataclass
class Field(Node):
    """One field line (spec 4.1).

    The workhorse node. Represents a single parameter/property: its name, the
    type it accepts, whether it is required, an optional default, and an
    optional description.

    NOTE: annotations do NOT live here. They attach to the type expression at
    the single-type level (spec 4.4), so they live on `type` (the type node),
    not on Field.

    A Field MUST NOT carry both `required=True` and a real default
    (spec 4.1 / 4.5). That exclusivity is a PROSE rule enforced in validate.py,
    not prevented by this structure — both are representable here on purpose, so
    the validator can produce a clear error rather than the parser silently
    refusing to build the tree.
    """
    name: str
    type: TypeNode
    required: bool = False
    default: DefaultValue = NO_DEFAULT
    description: Optional[str] = None


@dataclass
class ToolBlock(Node):
    """One complete tool definition (spec 3.3).

    Header (name + optional description) plus an indented body of field lines
    describing the input-parameter schema. There is no separate name:/
    description:/input_schema: wrapper — those are folded into this node's shape.
    """
    name: str
    fields: list[Field] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class Definition(Node):
    """One reusable definition inside the $defs block (spec 3.2).

    Structurally identical to ToolBlock (the spec says the two productions are
    the same), but kept as a distinct class so code can tell a $defs entry from
    a tool by type rather than by position. A definition's name is restricted to
    a bare identifier (spec 3.2) — no hyphens, no quoting — which validate.py
    can check.
    """
    name: str
    fields: list[Field] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class Document(Node):
    """The root node — one per CATS document (spec 3.1).

    An optional $defs block followed by one or more tool blocks. `defs` is
    `None` when no `$defs` block appears in the source; `[]` when a `$defs`
    header was written but contains zero definitions; otherwise a non-empty list
    of Definition nodes. A document MUST contain at least one tool block
    (spec 3.1); that minimum is a prose rule for validate.py, not enforced here.

    A tool entry is normally a `ToolBlock`. The reverse reader (from_json.py)
    may also place a `RawSchema` here: a tool containing a construct CATS cannot
    encode falls back as a WHOLE TOOL (spec 7.5), carried verbatim. Fallback is
    all-or-nothing per tool, so the entry is the whole RawSchema, never inline
    raw JSON inside an otherwise-CATS tool block.
    """
    tools: list[ToolBlock | RawSchema] = field(default_factory=list)
    defs: Optional[list[Definition]] = None


# ===========================================================================
# GROUP 2 — TYPE NODES  (the type vocabulary: spec sections 5 and 6)
#
# The eight closed type words (spec 2.4 / 5.1) plus the composition forms.
# Each primitive/composite carries ONLY the annotations section 6.5 permits
# for it, as named optional fields (design decision 1).
# ===========================================================================

@dataclass
class String(Node):
    """The `string` type (spec 5.2).

    Permitted annotations (spec 6.1, 6.3): :format, :length[min,max],
    :regex[...], :encoding[...], :media[...]. Length bounds are always
    inclusive (spec 6.3), so — unlike numeric bounds — no exclusivity flags are
    needed. Either length bound may be absent (None).
    """
    format: Optional[str] = None                  # :format (spec 6.1)
    min_length: Optional[int] = None              # :length lower (spec 6.3)
    max_length: Optional[int] = None              # :length upper (spec 6.3)
    pattern: Optional[str] = None                 # :regex value  (spec 6.3)
    encoding: Optional[str] = None                # :encoding     (spec 6.3)
    media: Optional[str] = None                   # :media        (spec 6.3)


@dataclass
class Integer(Node):
    """The `integer` type (spec 5.2).

    Permitted annotations (spec 6.2): :format, numeric bounds, and %multipleOf.
    Bounds are a lower/upper pair, each optionally exclusive. The four
    bracket/paren pairings of spec 6.2 are captured by the two exclusivity
    booleans. An absent bound is None.
    """
    format: Optional[str] = None                  # :format (spec 6.1)
    minimum: Optional[float] = None               # lower bound value (spec 6.2)
    maximum: Optional[float] = None               # upper bound value (spec 6.2)
    exclusive_min: bool = False                    # True if '(' i.e. exclusiveMinimum
    exclusive_max: bool = False                    # True if ')' i.e. exclusiveMaximum
    multiple_of: Optional[float] = None           # %divisor (spec 6.2)


@dataclass
class Number(Node):
    """The `number` type (spec 5.2).

    Same annotation set as Integer (spec 6.2); kept as a separate class because
    spec 2.5 fixes the integer/number distinction at the type slot. The values
    here are arbitrary numerics, hence float typing.
    """
    format: Optional[str] = None                  # :format (spec 6.1)
    minimum: Optional[float] = None               # lower bound value (spec 6.2)
    maximum: Optional[float] = None               # upper bound value (spec 6.2)
    exclusive_min: bool = False
    exclusive_max: bool = False
    multiple_of: Optional[float] = None           # %divisor (spec 6.2)


@dataclass
class Boolean(Node):
    """The `boolean` type (spec 5.2).

    Section 6.5 permits ONLY :format on a boolean. No other annotation field
    exists here on purpose — making, e.g., a numeric bound on a boolean
    impossible to represent.
    """
    format: Optional[str] = None                  # :format (spec 6.1)


@dataclass
class Null(Node):
    """The `null` type word (spec 5.2).

    Denotes the JSON null value, mainly as a branch of nullable unions
    (string|null, spec 5.5). Distinct from the null *literal* of spec 2.5, which
    is a value, not a type. Only :format is permitted (spec 6.5).
    """
    format: Optional[str] = None                  # :format (spec 6.1)


@dataclass
class AnyType(Node):
    """The `any` type word (spec 5.1 / 5.2).

    Accepts any JSON value; the canonical encoding of the empty schema {} and of
    description-only schemas (spec 5.2). Named `AnyType` (not `Any`) to avoid shadowing typing.Any. Carries no constraints by definition;
    :format is technically allowed on any base type (spec 6.5) but is unusual
    here, so it is included for completeness.
    """
    format: Optional[str] = None                  # :format (spec 6.1), rarely used


@dataclass
class Array(Node):
    """The `array` type (spec 5.3).

    `element` is the element type expression (another type node), or None for a
    bare untyped `array` (spec 5.3 — JSON Schema {"type": "array"} with no
    items). The recursion through `element` is what lets array<array<integer>>
    nest to any depth.

    Permitted annotations (spec 6.4): element-count bounds (min_items/max_items,
    always inclusive) and :unique. :format may also attach to any base type
    (spec 6.1, 6.5). Length, regex, encoding, media, and numeric bounds apply to
    the element type, not the array container.
    """
    element: Optional[TypeNode] = None            # element type, or None if bare (spec 5.3)
    format: Optional[str] = None                  # :format (spec 6.1)
    min_items: Optional[int] = None               # bounds lower = minItems (spec 6.4)
    max_items: Optional[int] = None               # bounds upper = maxItems (spec 6.4)
    unique: bool = False                           # :unique => uniqueItems (spec 6.4)


@dataclass
class Object(Node):
    """The `object` type (spec 5.4).

    Holds the nested block of field lines describing its properties (spec 4.7).
    All CATS objects are IMPLICITLY CLOSED (spec 5.4) — there is deliberately no
    field modeling additionalProperties, because closure is built into what an
    Object means.     A bare Object with empty `fields` validates only {} (spec 4.7).
    :format may attach to any base type (spec 6.1, 6.5).
    """
    fields: list[Field] = field(default_factory=list)
    format: Optional[str] = None                  # :format (spec 6.1)


@dataclass
class Reference(Node):
    """A $defs reference, written $Name in the type slot (spec 5.7).

    `name` is the bare identifier (no leading $ — the $ is the syntactic marker,
    not part of the name, spec 5.7). References compose anywhere a single type
    expression is valid (inside arrays, as union branches, etc.).
    :format may attach to any base type (spec 6.1, 6.5).
    """
    name: str                                      # the identifier after $ (spec 5.7)
    format: Optional[str] = None                  # :format (spec 6.1)


# --- Composition forms (layered on top of the base types) ------------------

@dataclass
class Union(Node):
    """A type union: pipe-separated type expressions (spec 5.5).

    Encodes JSON Schema anyOf (and oneOf, lossily — spec 5.5 / 7.4). Each branch
    is a full type node carrying its OWN annotation chain (spec 5.5), so
    annotations live on the branch nodes here, not on the Union.

    Homogeneity (all branches types, never mixed with literals) is a prose rule
    of spec 5.6 for validate.py.
    """
    branches: list[TypeNode] = field(default_factory=list)


@dataclass
class Enum(Node):
    """A multi-value enum union: pipe-separated value literals (spec 5.6).

    Distinct from Union: a Union is over TYPES, an Enum is over VALUES.
    `base_type` records the inferred primitive ("string" | "integer" |
    "number"), because the reverse direction emits an explicit `type` alongside
    `enum` (spec 7.2). `values` holds the literal members in declaration order.

    A two-member true/false set is encoded as Boolean, not Enum (spec 5.6) —
    another prose rule for validate.py.
    """
    values: list[Any] = field(default_factory=list)   # literal members (spec 5.6)
    base_type: Optional[str] = None                    # inferred: "string"|"integer"|"number" (spec 7.2)


@dataclass
class Const(Node):
    """A single-value enum (spec 5.6), emitted as JSON Schema `const` (spec 7.2).

    Kept separate from Enum so the mechanical reverse direction (spec 7.2) stays
    a clean one-to-one map: Const -> "const", multi-value Enum -> "enum". `value`
    is the lone literal (the quoted single value form, e.g. mode "automatic").
    """
    value: Any                                     # the single constant value (spec 5.6)


# ===========================================================================
# GROUP 3 — FALLBACK ESCAPE HATCH  (spec section 7.5)
# ===========================================================================

@dataclass
class RawSchema(Node):
    """Opaque raw JSON Schema carried verbatim (spec 7.5).

    The fallback target for every construct with no CATS form (allOf, not,
    contains, prefixItems, conditional keywords, discriminated unions, etc. —
    the full list in spec 8). Holds the original JSON subtree unchanged so it
    round-trips losslessly.

    `schema` is the parsed JSON (a dict, typically). In CATS text (§7.5) a
    fallen-back tool is written as that verbatim JSON object at column 0 in
    the document's tool sequence; the parser recognizes `{` at tool position as
    raw JSON rather than a CATS tool block header.
    """
    schema: Any                                    # verbatim JSON Schema subtree (spec 7.5)