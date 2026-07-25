"""
to_cats.py — the CATS AST -> CATS text serializer (the reverse of the parser).

    text --[lexer]--> tokens --[parser]--> AST --[validate]--> AST --[to_cats]--> text

Input  : a (validated) `Document` AST from nodes.py — or, for convenience, any
         single node (a ToolBlock, Definition, type node, or RawSchema).
Output : CATS text that the lexer + parser re-parse to a semantically identical
         AST. This module's correctness contract is exactly that round trip:
         parse_text(to_cats(ast)) reproduces `ast` (§3-§6, Appendix A).

The emission mirrors parser.py production-for-production. Two choices worth
calling out, both made to keep the round trip EXACT rather than merely spec-
pretty:

  - NUMERIC OPEN-END BRACKETS are always inclusive per §6.2 (`[1,]`, `[,10]`).
    Exclusivity flags apply only when that endpoint's bound value is present.
  - CLOSED-END exclusivity still follows the stored `exclusive_min` /
    `exclusive_max` booleans (`[1,3)` etc.).
  - QUOTING is applied only where a bare token would re-lex wrong (a value that
    collides with a type word / `true|false|null` / a number, or a name/value
    outside the bare `name` grammar of §2.3). Hyphenated names stay bare, since
    the `name` grammar admits the hyphen.

RAWSCHEMA / §7.5. A fallen-back tool is emitted as its verbatim JSON Schema
object at column 0 in the document's tool sequence (§7.5). A `{` at tool
position is not valid CATS tool-block syntax, so the parser distinguishes raw
JSON tools from CATS tool blocks unambiguously.
"""

from __future__ import annotations

import json
import re
from typing import Optional

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

INDENT_UNIT = "  "  # two spaces per nesting level (§2.2)

# The eight closed type words (§2.4): a bare value spelling one of these would
# re-lex as a TYPE, so such a value must be quoted (§2.6 trigger 2).
_TYPE_WORDS = frozenset(
    {"string", "integer", "number", "boolean", "array", "object", "null", "any"}
)

# JSON literals that, written bare, re-lex as a boolean/null VALUE rather than a
# string (§2.6 trigger 1).
_LITERAL_WORDS = frozenset({"true", "false", "null"})

# A clean bare `name` (§2.3): identifier start, then letters/digits/_/hyphen.
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")


# ---------------------------------------------------------------------------
# Scalar / token formatting
# ---------------------------------------------------------------------------

def _quote(text: str) -> str:
    """Wrap `text` in double quotes, escaping `\\` and `"` per §2.5."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _num(value: object) -> str:
    """A NUMBER lexeme that re-parses to the same Python value (§2.5).

    An int prints without a dot (re-parses to int); a float prints via `repr`,
    which always carries a `.` or exponent so the parser classifies it as a
    float — keeping integer/number value identity across the round trip.
    """
    if isinstance(value, bool):  # defensive: a bound should never be a bool
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return repr(value)


def _value_needs_quoting(text: str) -> bool:
    """True when a string VALUE must be quoted to re-lex as that string (§2.6)."""
    return (
        text in _TYPE_WORDS
        or text in _LITERAL_WORDS
        or _NAME_RE.match(text) is None
    )


def _name_needs_quoting(text: str) -> bool:
    """True when a NAME must be quoted: a reserved type word (§2.4) or outside
    the bare `name` grammar of §2.3 (leading digit/hyphen, dot, slash, …)."""
    return text in _TYPE_WORDS or _NAME_RE.match(text) is None


def _name(text: str) -> str:
    return _quote(text) if _name_needs_quoting(text) else text


def _enum_member(value: object) -> str:
    """One enum member literal (§5.6): booleans/null/numbers bare, strings quoted
    only when bare would re-lex wrong."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return _num(value)
    if isinstance(value, str):
        return _quote(value) if _value_needs_quoting(value) else value
    return _quote(json.dumps(value))  # non-scalar enum member: not real CATS


def _const_literal(value: object) -> str:
    """A single-value enum / `const` (§5.6). Per the task, a string const is
    ALWAYS quoted (the canonical `mode "automatic"` form). `null` has no value
    spelling distinct from the type word, so a null const emits `null` — which
    re-parses to the Null type; behavior is identical (both accept only null)
    though the node type differs (flagged)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return _num(value)
    if isinstance(value, str):
        return _quote(value)
    return _quote(json.dumps(value))


def _default_literal(value: object) -> str:
    """A default value glued after `=` (§4.5). Strings are always quoted; objects
    and arrays are compact JSON (no internal whitespace, so it stays one token)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return _num(value)
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return json.dumps(value)


def _description_suffix(description: Optional[str]) -> str:
    """The ` # ...` tail for a header or field line (§4.6).

    A description containing `#` is wrapped in quotes (§2.6 trigger 3); the
    parser strips the outer pair WITHOUT decoding escapes, so we add no escapes.
    """
    if description is None:
        return ""
    text = _quote(description) if "#" in description else description
    return " # " + text


# ---------------------------------------------------------------------------
# Annotation chains (§6) — emitted in the canonical §6.5 order
# ---------------------------------------------------------------------------

def _format_prefix(fmt: Optional[str]) -> str:
    return f":{fmt}" if fmt is not None else ""


def _string_annotations(node: String) -> str:
    """:format :length :regex :encoding :media — canonical order (§6.3/§6.5)."""
    chain = _format_prefix(node.format)
    if node.min_length is not None or node.max_length is not None:
        lo = str(node.min_length) if node.min_length is not None else ""
        hi = str(node.max_length) if node.max_length is not None else ""
        chain += f":length[{lo},{hi}]"
    if node.pattern is not None:
        chain += f":regex[{_quote(node.pattern)}]"
    if node.encoding is not None:
        chain += f":encoding[{node.encoding}]"
    if node.media is not None:
        chain += f":media[{_quote(node.media)}]"
    return chain


def _numeric_annotations(node: Integer | Number) -> str:
    """:format, numeric bounds, then %divisor — canonical order (§6.2/§6.5).

    Open endpoints always use inclusive brackets (§6.2); exclusivity flags apply
    only when the bound value on that side is present.
    """
    chain = _format_prefix(node.format)
    if node.minimum is not None or node.maximum is not None:
        if node.minimum is None:
            lo_char = "["
        else:
            lo_char = "(" if node.exclusive_min else "["
        if node.maximum is None:
            hi_char = "]"
        else:
            hi_char = ")" if node.exclusive_max else "]"
        lo = _num(node.minimum) if node.minimum is not None else ""
        hi = _num(node.maximum) if node.maximum is not None else ""
        chain += f"{lo_char}{lo},{hi}{hi_char}"
    if node.multiple_of is not None:
        chain += f"%{_num(node.multiple_of)}"
    return chain


def _array_annotations(node: Array) -> str:
    """:format, element-count bounds, then :unique — canonical order (§6.4/§6.5).
    Array bounds are always inclusive (§6.4), so the brackets are always `[`/`]`.
    """
    chain = _format_prefix(node.format)
    if node.min_items is not None or node.max_items is not None:
        lo = str(node.min_items) if node.min_items is not None else ""
        hi = str(node.max_items) if node.max_items is not None else ""
        chain += f"[{lo},{hi}]"
    if node.unique:
        chain += ":unique"
    return chain


# ---------------------------------------------------------------------------
# Type expressions (§5) — the inline, single-line form (no nested block)
# ---------------------------------------------------------------------------

def _type_expression(node: Node) -> str:
    """Serialize a type node to its inline type expression (§5.1).

    An `object` (and `array<object>`) emits just the bare type word here; its
    field lines live in the §4.7 nested block, which the field-level emitter
    writes after the line — see `_emit_field`.
    """
    if isinstance(node, String):
        return "string" + _string_annotations(node)
    if isinstance(node, Integer):
        return "integer" + _numeric_annotations(node)
    if isinstance(node, Number):
        return "number" + _numeric_annotations(node)
    if isinstance(node, Boolean):
        return "boolean" + _format_prefix(node.format)
    if isinstance(node, Null):
        return "null" + _format_prefix(node.format)
    if isinstance(node, AnyType):
        return "any" + _format_prefix(node.format)
    if isinstance(node, Array):
        text = "array"
        if node.element is not None:
            text += f"<{_type_expression(node.element)}>"
        return text + _array_annotations(node)
    if isinstance(node, Object):
        return "object" + _format_prefix(node.format)
    if isinstance(node, Reference):
        return f"${node.name}" + _format_prefix(node.format)
    if isinstance(node, Union):
        return "|".join(_type_expression(branch) for branch in node.branches)
    if isinstance(node, Enum):
        return "|".join(_enum_member(v) for v in node.values)
    if isinstance(node, Const):
        return _const_literal(node.value)
    raise TypeError(
        f"cannot serialize node of type {type(node).__name__} as a type expression"
    )


def _nested_fields(type_node: Node) -> Optional[list[Field]]:
    """The §4.7 nested-block fields a field of this type carries, or None.

    Only an `object` with fields, or an `array<object>` whose element object has
    fields, opens a block — the two shapes the parser's `_attach_nested_block`
    accepts. An empty-bodied object emits no block (§4.7).
    """
    if isinstance(type_node, Object) and type_node.fields:
        return type_node.fields
    if (
        isinstance(type_node, Array)
        and isinstance(type_node.element, Object)
        and type_node.element.fields
    ):
        return type_node.element.fields
    return None


# ---------------------------------------------------------------------------
# Field lines and blocks (§3, §4)
# ---------------------------------------------------------------------------

def _emit_field(field: Field, level: int) -> list[str]:
    """One field line (§4.1) at `level`, plus its nested block if any (§4.7)."""
    indent = INDENT_UNIT * level
    head = _name(field.name) + ("*" if field.required else "")
    line = f"{indent}{head} {_type_expression(field.type)}"
    if field.default is not NO_DEFAULT:
        line += f" ={_default_literal(field.default)}"
    line += _description_suffix(field.description)

    lines = [line]
    nested = _nested_fields(field.type)
    if nested is not None:
        for child in nested:
            lines.extend(_emit_field(child, level + 1))
    return lines


def _emit_block(
    header_name: str, description: Optional[str], fields: list[Field], level: int
) -> list[str]:
    """A header line (tool / definition) plus its indented field body."""
    indent = INDENT_UNIT * level
    lines = [f"{indent}{_name(header_name)}{_description_suffix(description)}"]
    for field in fields:
        lines.extend(_emit_field(field, level + 1))
    return lines


def _emit_raw(raw: RawSchema, level: int) -> list[str]:
    """A whole-tool RawSchema fallback (§7.5): verbatim JSON at column 0."""
    if level != 0:
        # Fallback is whole-tool only; nested raw JSON inside a CATS block is N/A.
        indent = INDENT_UNIT * level
        payload = json.dumps(raw.schema, separators=(",", ":"))
        return [f"{indent}{payload}"]
    payload = json.dumps(raw.schema, separators=(",", ":"))
    return [payload]


def _emit_defs_block(defs: list[Definition]) -> list[str]:
    """The `$defs` block (§3.2): the `$defs` header, then each definition."""
    lines = ["$defs"]
    for definition in defs:
        lines.extend(
            _emit_block(definition.name, definition.description, definition.fields, level=1)
        )
    return lines


def _emit_document(doc: Document) -> str:
    """A whole document (§3.1): optional `$defs` block, then tool blocks,
    each block separated by one blank line."""
    blocks: list[str] = []
    if doc.defs is not None:
        blocks.append("\n".join(_emit_defs_block(doc.defs)))
    for tool in doc.tools:
        if isinstance(tool, RawSchema):
            blocks.append("\n".join(_emit_raw(tool, level=0)))
        else:
            blocks.append(
                "\n".join(_emit_block(tool.name, tool.description, tool.fields, level=0))
            )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def to_cats(node: Node) -> str:
    """Serialize a CATS AST node to CATS text.

    A `Document` becomes a full document; a `ToolBlock`/`Definition` becomes its
    single block; a bare type node becomes its inline type expression (handy for
    tests); a `RawSchema` becomes verbatim JSON (§7.5). The result carries no
    trailing newline.
    """
    if isinstance(node, Document):
        return _emit_document(node)
    if isinstance(node, ToolBlock):
        return "\n".join(_emit_block(node.name, node.description, node.fields, level=0))
    if isinstance(node, Definition):
        return "\n".join(_emit_block(node.name, node.description, node.fields, level=0))
    if isinstance(node, RawSchema):
        return "\n".join(_emit_raw(node, level=0))
    return _type_expression(node)
