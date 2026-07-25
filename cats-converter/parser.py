"""
parser.py — the CATS parser (a flat token list -> an AST).

Second stage of the pipeline:

    text --[lexer]--> tokens --[parser]--> AST --[validate]--> AST --[serializer]--> text

Input  : the list[Token] produced by lexer.tokenize().
Output : a `Document` node (nodes.py) — the shared AST both directions hinge on.

THE CORE RULE: parse PERMISSIVELY, do not validate (Appendix A; project design).
The CATS grammar is deliberately permissive — many real rules live in PROSE for
validate.py to enforce later, NOT here. This parser therefore BUILDS a tree for
input that is grammatically derivable but semantically illegal, and leaves the
judging to validate.py with the whole tree in hand. Concretely, the parser does
NOT reject (it builds anyway):

  - a field with both `*` and a default  (§4.1 forbids — Field carries both)
  - a mixed type/value union  (§5.6 forbids — a Union with a Const branch)
  - annotations out of canonical order  (§6.5 — order is simply not retained)

It DOES raise `ParseError` for structurally un-parseable input AND for
annotations that have no legal home on their base type (§6.5) — e.g. a numeric
bound on a `string`, `:length` on an `integer`, `:unique` on a `boolean`. Those
are treated as structural errors (the annotation has nowhere to attach), with a
message naming the annotation, the type, and §6.5. `:format` attaches to ANY base
type (§6.1) and must land on every one.

KEY SEAM — annotations map onto NAMED FIELDS (project decision 1):

    :format                -> .format            (any base type)
    numeric bounds [a,b)   -> .minimum/.maximum + .exclusive_min/.exclusive_max (integer/number)
    %divisor               -> .multiple_of       (integer/number)
    :length[a,b]           -> .min_length/.max_length (string)
    :regex["..."]          -> .pattern           (string)
    :encoding[v]           -> .encoding          (string)
    :media["..."]          -> .media             (string)
    array bounds [a,b]     -> .min_items/.max_items (array)
    :unique                -> .unique            (array)

This module is pure structure. It performs exactly one piece of interpretation
the lexer deferred: classifying each bare IDENT by POSITION (§2.4/§2.5) — a type
word in the type slot becomes its type node, `null` there is the Null *type*
(whereas `=null` is the null *value*), and any other bare word in value position
is a string value. Everything else is shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from lexer import Token, TokenType
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

# The eight closed type words (§2.4). A bare IDENT equal to one of these, in the
# type slot, is a TYPE; any other bare IDENT there is a value literal (§5.6).
_TYPE_WORDS = frozenset(
    {"string", "integer", "number", "boolean", "array", "object", "null", "any"}
)

# Named string/array annotations whose keyword is followed by a bracketed arg
# (`:length[..]`, `:regex[..]`, `:encoding[..]`, `:media[..]`). Distinguished
# from a format value of the same spelling by the presence of the `[` (§6.1/§6.3).
_BRACKET_ANNOTATIONS = frozenset({"length", "regex", "encoding", "media"})


class ParseError(Exception):
    """A structural syntax error — the token stream does not form a tree.

    Carries a 1-based location (taken from the offending token) so callers can
    point at it. NOT used for semantic/prose violations (those are built and
    handed to validate.py); only for input that cannot be assembled at all.
    """

    def __init__(self, message: str, line: int, col: int) -> None:
        self.message = message
        self.line = line
        self.col = col
        super().__init__(f"{message} (line {line}, column {col})")


def _parse_number(lexeme: str) -> int | float:
    """Turn a NUMBER lexeme (§2.5) into a Python int or float.

    A fractional or exponent part means a float; otherwise an int. The lexer has
    already validated the shape, so this never sees garbage.
    """
    if any(c in lexeme for c in ".eE"):
        return float(lexeme)
    return int(lexeme)


def _strip_description_quotes(raw: str) -> str:
    """Apply the §4.6 description rule to the lexer's RAW description text.

    The lexer hands descriptions over verbatim, INCLUDING surrounding quotes
    when the text was quoted to hide a literal `#` (§2.6 trigger 3) — see the
    lexer test `test_quoted_description_stays_raw`. Stripping those quotes is the
    parser's job. We strip a single outer pair if present and DO NOT
    escape-decode the inside: §4.6 prose is uninterpreted, so a literal
    backslash in description text stays a literal backslash (unlike a STRING).
    """
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    return raw


@dataclass
class _Annotations:
    """Mutable scratch for one parsed annotation chain (§6).

    A neutral, type-agnostic collection of whatever annotation pieces appeared,
    in whatever order. `_Parser._apply_annotations` writes these onto the named
    fields of the specific base-type node and raises ParseError if §6.5 forbids
    the pairing. Location fields point at the token where each piece started.
    """

    format: Optional[str] = None
    format_loc: Optional[Token] = None
    # (lower, upper, exclusive_lower, exclusive_upper); None lower/upper = open.
    bounds: Optional[tuple[Optional[float], Optional[float], bool, bool]] = None
    bounds_loc: Optional[Token] = None
    mult: Optional[float] = None
    mult_loc: Optional[Token] = None
    length: Optional[tuple[Optional[int], Optional[int]]] = None
    length_loc: Optional[Token] = None
    regex: Optional[str] = None
    regex_loc: Optional[Token] = None
    encoding: Optional[str] = None
    encoding_loc: Optional[Token] = None
    media: Optional[str] = None
    media_loc: Optional[Token] = None
    unique: bool = False
    unique_loc: Optional[Token] = None

    def has_any(self) -> bool:
        return any(
            (
                self.format is not None,
                self.bounds is not None,
                self.mult is not None,
                self.length is not None,
                self.regex is not None,
                self.encoding is not None,
                self.media is not None,
                self.unique,
            )
        )


@dataclass
class _Branch:
    """One element of a (possibly single-element) pipe list (§5.1, §5.5, §5.6).

    A branch is either a TYPE (a built type node) or a VALUE literal (a parsed
    Python value plus its inferred base type, used to label an Enum). Keeping
    the two distinguishable lets `_finish_type_expression` decide between a
    type-union, an enum-union, a single type, and a single-value enum (§5.6).
    """

    kind: str                       # "type" | "value"
    node: object = None             # the type node when kind == "type"
    value: object = None            # the Python value when kind == "value"
    vbt: Optional[str] = None       # inferred value base type: string/integer/number/boolean


def _infer_enum_base(vbts: list[Optional[str]]) -> Optional[str]:
    """Infer an Enum's `base_type` (§7.2) from its members' value base types.

    All-string -> "string", all-boolean -> "boolean", all-integer -> "integer",
    integers mixed with floats -> "number". A genuinely mixed set (e.g. a string
    next to a number — a §5.6 homogeneity violation we still build) yields None;
    validate.py is responsible for flagging that, not us.
    """
    distinct = set(vbts)
    if distinct == {"string"}:
        return "string"
    if distinct == {"boolean"}:
        return "boolean"
    if distinct <= {"integer"}:
        return "integer"
    if distinct <= {"integer", "number"}:
        return "number"
    return None


class _Parser:
    """A single-pass recursive-descent parser over a fixed token list.

    Holds the token list and a read cursor. Every `parse_*`/`_parse_*` method
    consumes the tokens for its production and returns the built node, advancing
    the cursor. The grammar followed is Appendix A (authoritative); body
    fragments in §3-§6 are illustrative.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # --- cursor helpers ----------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        """The token `offset` ahead, clamped to the trailing EOF token."""
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[idx]

    def _advance(self) -> Token:
        """Return the current token and step past it."""
        tok = self._peek()
        if self._pos < len(self._tokens):
            self._pos += 1
        return tok

    def _expect(self, type_: TokenType, what: str) -> Token:
        """Consume a token of `type_` or raise a structural ParseError."""
        tok = self._peek()
        if tok.type is not type_:
            raise ParseError(
                f"expected {what}, got {tok.type.name}"
                + (f" {tok.value!r}" if tok.value else ""),
                tok.line,
                tok.col,
            )
        return self._advance()

    # --- document level (§3.1) --------------------------------------------

    def parse_document(self) -> Document:
        """document ::= defs-block? tool-block+   (§3.1, A.1)

        Permissive on the count: a document with zero tool blocks (only `$defs`,
        or empty) still builds; the §3.1 "at least one tool" rule is validate's.
        """
        defs: Optional[list[Definition]] = None
        # A `$defs` block is the only construct that starts with `$` at column 0
        # (tool names never begin with `$`); the lexer emits it as DOLLAR + IDENT.
        if (
            self._peek().type is TokenType.DOLLAR
            and self._peek(1).type is TokenType.IDENT
            and self._peek(1).value == "defs"
        ):
            defs = self._parse_defs_block()  # [] if header present but empty

        tools: list[ToolBlock] = []
        while self._peek().type is not TokenType.EOF:
            tools.append(self._parse_tool_block())

        return Document(tools=tools, defs=defs)

    def _parse_defs_block(self) -> list[Definition]:
        """defs-block ::= "$defs" NEWLINE INDENT definition+ DEDENT

        A `#` description on the `$defs` header itself is ill-formed (§3.2).
        An empty `$defs` (no INDENT) has
        no valid grammar form (§3.2); we build it as zero definitions and leave
        the complaint to validate.py rather than rejecting here.
        """
        self._advance()  # DOLLAR
        self._advance()  # IDENT "defs"
        if self._peek().type is TokenType.HASH:
            raise ParseError(
                "the $defs header takes no description (§3.2)",
                self._peek().line,
                self._peek().col,
            )
        self._expect(TokenType.NEWLINE, "newline after '$defs'")

        definitions: list[Definition] = []
        if self._peek().type is TokenType.INDENT:
            self._advance()
            while self._peek().type not in (TokenType.DEDENT, TokenType.EOF):
                definitions.append(self._parse_definition())
            if self._peek().type is TokenType.DEDENT:
                self._advance()
        return definitions

    def _parse_definition(self) -> Definition:
        """definition ::= identifier description? NEWLINE INDENT field-line+ DEDENT

        The grammar restricts a definition name to a bare identifier; we accept
        an IDENT (which also covers a hyphenated name the lexer can't tell apart)
        or a quoted STRING, and leave the §3.2 identifier-only check to validate.
        """
        name_tok = self._peek()
        if name_tok.type not in (TokenType.IDENT, TokenType.STRING):
            raise ParseError(
                "expected a definition name", name_tok.line, name_tok.col
            )
        self._advance()

        description = None
        if self._peek().type is TokenType.HASH:
            description = self._parse_description()
        self._expect(TokenType.NEWLINE, "newline after definition header")

        fields = self._parse_body()
        return Definition(name=name_tok.value, fields=fields, description=description)

    def _parse_tool_block(self) -> ToolBlock:
        """tool-block ::= (name | string-literal) description? NEWLINE INDENT field-line+ DEDENT"""
        name_tok = self._peek()
        if name_tok.type not in (TokenType.IDENT, TokenType.STRING):
            raise ParseError(
                "expected a tool name at the start of a tool block",
                name_tok.line,
                name_tok.col,
            )
        self._advance()

        description = None
        if self._peek().type is TokenType.HASH:
            description = self._parse_description()
        self._expect(TokenType.NEWLINE, "newline after tool header")

        fields = self._parse_body()
        return ToolBlock(name=name_tok.value, fields=fields, description=description)

    def _parse_body(self) -> list[Field]:
        """The INDENT field-line+ DEDENT body shared by tools and definitions.

        Permissive on the body being empty: a header with no indented block
        yields an empty field list rather than an error, so a parameterless tool
        is representable. The grammar's `field-line+` minimum is validate's call.
        """
        fields: list[Field] = []
        if self._peek().type is TokenType.INDENT:
            self._advance()
            while self._peek().type not in (TokenType.DEDENT, TokenType.EOF):
                fields.append(self._parse_field_line())
            if self._peek().type is TokenType.DEDENT:
                self._advance()
        return fields

    # --- field lines (§4) --------------------------------------------------

    def _parse_field_line(self) -> Field:
        """field-line ::= field-name required-marker? SP type-expression
                          default? description? NEWLINE nested-block?   (§4.1, A.2)

        The field name and a type expression are mandatory; everything from the
        default onward is optional. The single separating space is gone (the
        lexer consumed it), so the type expression follows the name/`*` directly.
        """
        # field-name ::= name | string-literal
        name_tok = self._peek()
        if name_tok.type not in (TokenType.IDENT, TokenType.STRING):
            raise ParseError("expected a field name", name_tok.line, name_tok.col)
        self._advance()
        name = name_tok.value

        # required-marker ::= "*"
        required = False
        if self._peek().type is TokenType.STAR:
            self._advance()
            required = True

        # type-expression (mandatory). Catch "field has no type" explicitly for a
        # clearer message than the generic branch error would give.
        if self._peek().type not in (
            TokenType.IDENT,
            TokenType.STRING,
            TokenType.NUMBER,
            TokenType.DOLLAR,
        ):
            bad = self._peek()
            raise ParseError(
                f"field {name!r} has no type expression", bad.line, bad.col
            )
        type_node = self._parse_type_expression()

        # default ::= "=" (value-literal | json-literal)
        default = NO_DEFAULT
        if self._peek().type is TokenType.EQUALS:
            self._advance()
            default = self._parse_default_value()

        # description ::= "#" description-text
        description = None
        if self._peek().type is TokenType.HASH:
            description = self._parse_description()

        self._expect(TokenType.NEWLINE, "end of field line")

        # nested-block ::= INDENT field-line+ DEDENT  (§4.7)
        if self._peek().type is TokenType.INDENT:
            indent_tok = self._peek()
            nested = self._parse_body()  # reuse INDENT..DEDENT consumer
            self._attach_nested_block(type_node, nested, indent_tok)

        return Field(
            name=name,
            type=type_node,
            required=required,
            default=default,
            description=description,
        )

    def _attach_nested_block(
        self, type_node: object, nested: list[Field], indent_tok: Token
    ) -> None:
        """Attach a §4.7 nested block to the object it describes.

        §4.7 permits a nested block only on a field typed `object` or
        `array<object>`. Those are the only nodes with a `fields` container, so a
        block under anything else is an INDENT where no block can open — a
        structural error (not a semantic one), since the AST has nowhere to put
        the fields.
        """
        if isinstance(type_node, Object):
            type_node.fields = nested
        elif isinstance(type_node, Array) and isinstance(type_node.element, Object):
            type_node.element.fields = nested
        else:
            raise ParseError(
                "nested block under a field whose type cannot contain fields "
                "(only 'object' or 'array<object>' may, per §4.7)",
                indent_tok.line,
                indent_tok.col,
            )

    def _parse_description(self) -> str:
        """description ::= "#" description-text  (§4.6)

        The lexer emits HASH then one DESCRIPTION token (the raw rest of line,
        quotes included). We strip an outer quote pair here (§4.6) without
        decoding escapes.
        """
        self._expect(TokenType.HASH, "'#'")
        if self._peek().type is TokenType.DESCRIPTION:
            raw = self._advance().value
            return _strip_description_quotes(raw)
        return ""

    def _parse_default_value(self) -> object:
        """default value ::= value-literal | json-literal  (§4.5, A.2/A.5)

        Called with the cursor just past `=`. A JSON_LITERAL (`={}`, `=[1,2]`) is
        parsed as opaque JSON. A bare IDENT in this position is a VALUE, so
        `null` here is the null value (None) — unlike `null` in the type slot,
        which is the Null type.
        """
        tok = self._peek()
        if tok.type is TokenType.JSON_LITERAL:
            self._advance()
            try:
                return json.loads(tok.value)
            except json.JSONDecodeError as exc:
                raise ParseError(
                    f"invalid JSON default {tok.value!r}: {exc.msg}",
                    tok.line,
                    tok.col,
                ) from exc
        if tok.type is TokenType.STRING:
            self._advance()
            return tok.value
        if tok.type is TokenType.NUMBER:
            self._advance()
            return _parse_number(tok.value)
        if tok.type is TokenType.IDENT:
            self._advance()
            if tok.value == "true":
                return True
            if tok.value == "false":
                return False
            if tok.value == "null":
                return None
            return tok.value  # bare enum-member default (§4.5)
        raise ParseError("expected a default value after '='", tok.line, tok.col)

    # --- type expressions (§5) --------------------------------------------

    def _parse_type_expression(self) -> object:
        """type-expression ::= type-union | enum-union | single-value-enum | single-type

        Parsed uniformly as one-or-more pipe-separated branches, then resolved by
        §5.6's rule (branches all type words => union of types; all value
        literals => enum/const; a mix is the illegal case we still build as a
        Union carrying Const branches).
        """
        branches = [self._parse_branch()]
        while self._peek().type is TokenType.PIPE:
            self._advance()
            branches.append(self._parse_branch())
        return self._finish_type_expression(branches)

    def _finish_type_expression(self, branches: list[_Branch]) -> object:
        """Collapse parsed branches into the right node per §5.6 / §7.2."""
        if len(branches) == 1:
            only = branches[0]
            if only.kind == "type":
                return only.node
            # single-value-enum -> Const (§5.6, emitted as JSON Schema const §7.2)
            return Const(value=only.value)

        if all(b.kind == "value" for b in branches):
            # enum-union: pipe-separated value literals (§5.6)
            return Enum(
                values=[b.value for b in branches],
                base_type=_infer_enum_base([b.vbt for b in branches]),
            )

        # type-union (§5.5), or the illegal mixed union (§5.6) — built either way.
        # Value branches in a mix are wrapped as Const so the Union stays a list
        # of type nodes and validate.py can spot the Const-in-Union violation.
        return Union(
            branches=[
                b.node if b.kind == "type" else Const(value=b.value)
                for b in branches
            ]
        )

    def _parse_branch(self) -> _Branch:
        """One branch of a type expression: a single-type OR a value literal.

        Classification by position (§2.4/§2.5): a bare IDENT that is one of the
        eight type words is a TYPE; `true`/`false` are boolean VALUES; any other
        bare word is a string VALUE; a quoted STRING or a NUMBER is a VALUE; `$`
        opens a reference. (`null` is a type word, so it is handled as the Null
        type, never as a value here.)
        """
        tok = self._peek()

        if tok.type is TokenType.DOLLAR:
            return _Branch(kind="type", node=self._parse_single_type())

        if tok.type is TokenType.IDENT:
            if tok.value in _TYPE_WORDS:
                return _Branch(kind="type", node=self._parse_single_type())
            self._advance()
            if tok.value == "true":
                return _Branch(kind="value", value=True, vbt="boolean")
            if tok.value == "false":
                return _Branch(kind="value", value=False, vbt="boolean")
            return _Branch(kind="value", value=tok.value, vbt="string")

        if tok.type is TokenType.STRING:
            self._advance()
            return _Branch(kind="value", value=tok.value, vbt="string")

        if tok.type is TokenType.NUMBER:
            self._advance()
            number = _parse_number(tok.value)
            return _Branch(
                kind="value",
                value=number,
                vbt="integer" if isinstance(number, int) else "number",
            )

        raise ParseError(
            f"expected a type or value, got {tok.type.name}", tok.line, tok.col
        )

    def _parse_single_type(self) -> object:
        """single-type ::= base-type annotation-chain?   (§5.1)

        base-type is a primitive word, an `array`(<...>), `object`, or a `$`
        reference. The annotation chain (if any) is parsed and applied onto the
        node's named fields.
        """
        tok = self._peek()

        # reference ::= "$" identifier  (§5.7) — recombine DOLLAR + IDENT.
        if tok.type is TokenType.DOLLAR:
            self._advance()
            name_tok = self._expect(TokenType.IDENT, "a name after '$'")
            node: object = Reference(name=name_tok.value)
            self._apply_annotations(node, self._parse_annotation_chain())
            return node

        word = tok.value
        self._advance()
        if word == "array":
            array = Array()
            # array-type ::= "array" ("<" type-expression ">")?  (§5.3)
            if self._peek().type is TokenType.LANGLE:
                self._advance()
                array.element = self._parse_type_expression()
                self._expect(TokenType.RANGLE, "'>' to close the array element type")
            node = array
        elif word == "object":
            node = Object()
        elif word == "string":
            node = String()
        elif word == "integer":
            node = Integer()
        elif word == "number":
            node = Number()
        elif word == "boolean":
            node = Boolean()
        elif word == "null":
            node = Null()
        elif word == "any":
            node = AnyType()
        else:
            # Unreachable: _parse_branch only routes type words / `$` here.
            raise ParseError(f"unknown type word {word!r}", tok.line, tok.col)

        self._apply_annotations(node, self._parse_annotation_chain())
        return node

    # --- annotation chain (§6) --------------------------------------------

    def _parse_annotation_chain(self) -> _Annotations:
        """annotation-chain ::= format? bounds? mult? length? regex? encoding?
                                media? unique?   (§6.5, A.4)

        Parsed ORDER-INDEPENDENTLY and repeat-tolerantly: we loop over whatever
        annotation-starting tokens appear (`:`, `[`/`(`, `%`) and record each.
        Out-of-order chains (§6.5) are accepted here and left for validate — in
        practice the order is simply not retained once it lands in named fields.
        A later repeat of the same slot overwrites the earlier value.

        Legality of each annotation for the target base type is checked in
        `_apply_annotations`, which has the node in hand and raises ParseError
        at the stored token location if §6.5 forbids the pairing.
        """
        ann = _Annotations()
        while True:
            tok = self._peek()

            if tok.type is TokenType.COLON:
                colon = self._advance()
                self._parse_colon_annotation(ann, colon)
            elif tok.type in (TokenType.LBRACKET, TokenType.LPAREN):
                ann.bounds_loc = tok
                ann.bounds = self._parse_bounds()
            elif tok.type is TokenType.PERCENT:
                ann.mult_loc = self._advance()
                num = self._peek()
                if num.type is not TokenType.NUMBER:
                    raise ParseError(
                        "expected a number after '%'", num.line, num.col
                    )
                self._advance()
                ann.mult = _parse_number(num.value)
            else:
                break
        return ann

    def _parse_colon_annotation(self, ann: _Annotations, colon: Token) -> None:
        """A single `:`-introduced annotation (cursor is just past the `:`).

        Disambiguates a named bracket annotation (`:length[..]`, `:regex[..]`,
        `:encoding[..]`, `:media[..]`) from a plain `:format` value of the same
        spelling by whether a `[` follows the keyword (§6.1/§6.3). `:unique`
        (§6.4) takes no argument.
        """
        kw_tok = self._peek()
        if kw_tok.type is not TokenType.IDENT:
            raise ParseError(
                "expected an annotation name after ':'", kw_tok.line, kw_tok.col
            )
        keyword = kw_tok.value

        if (
            keyword in _BRACKET_ANNOTATIONS
            and self._peek(1).type is TokenType.LBRACKET
        ):
            self._advance()  # keyword
            if keyword == "length":
                ann.length_loc = kw_tok
                self._expect(TokenType.LBRACKET, f"'[' after ':{keyword}'")
                lower = self._optional_number()
                self._expect(TokenType.COMMA, "',' in ':length'")
                upper = self._optional_number()
                self._expect(TokenType.RBRACKET, "']' to close ':length'")
                ann.length = (lower, upper)
            elif keyword == "regex":
                ann.regex_loc = kw_tok
                self._expect(TokenType.LBRACKET, f"'[' after ':{keyword}'")
                value = self._expect(TokenType.STRING, "a quoted regex pattern")
                self._expect(TokenType.RBRACKET, "']' to close ':regex'")
                ann.regex = value.value
            elif keyword == "encoding":
                ann.encoding_loc = kw_tok
                self._expect(TokenType.LBRACKET, f"'[' after ':{keyword}'")
                value = self._expect(TokenType.IDENT, "an encoding identifier")
                self._expect(TokenType.RBRACKET, "']' to close ':encoding'")
                ann.encoding = value.value
            else:  # media
                ann.media_loc = kw_tok
                self._expect(TokenType.LBRACKET, f"'[' after ':{keyword}'")
                value = self._expect(TokenType.STRING, "a quoted media type")
                self._expect(TokenType.RBRACKET, "']' to close ':media'")
                ann.media = value.value
        elif keyword == "unique":
            self._advance()
            ann.unique = True
            ann.unique_loc = kw_tok
        else:
            # format ::= ":" format-value  (§6.1) — verbatim passthrough.
            self._advance()
            ann.format = keyword
            ann.format_loc = colon

    def _parse_bounds(
        self,
    ) -> tuple[Optional[float], Optional[float], bool, bool]:
        """bounds ::= bracket-open number? "," number? bracket-close  (§6.2/§6.4)

        `[`/`]` are inclusive, `(`/`)` exclusive; either endpoint may be omitted
        (open) per §6.2. Returns (lower, upper, exclusive_lower, exclusive_upper)
        with the bracket interpretation left to `_apply_annotations` by type.
        """
        open_tok = self._advance()  # LBRACKET or LPAREN
        exclusive_lower = open_tok.type is TokenType.LPAREN
        lower = self._optional_number()
        self._expect(TokenType.COMMA, "',' separating the two bounds")
        upper = self._optional_number()

        close_tok = self._peek()
        if close_tok.type is TokenType.RBRACKET:
            exclusive_upper = False
        elif close_tok.type is TokenType.RPAREN:
            exclusive_upper = True
        else:
            raise ParseError(
                "expected ']' or ')' to close the bounds",
                close_tok.line,
                close_tok.col,
            )
        self._advance()
        return (lower, upper, exclusive_lower, exclusive_upper)

    def _optional_number(self) -> Optional[float]:
        """A NUMBER if one is next (an endpoint value), else None (open side)."""
        if self._peek().type is TokenType.NUMBER:
            return _parse_number(self._advance().value)
        return None

    @staticmethod
    def _type_label(node: object) -> str:
        """Human-readable base-type name for §6.5 error messages."""
        if isinstance(node, String):
            return "string"
        if isinstance(node, Integer):
            return "integer"
        if isinstance(node, Number):
            return "number"
        if isinstance(node, Boolean):
            return "boolean"
        if isinstance(node, Null):
            return "null"
        if isinstance(node, AnyType):
            return "any"
        if isinstance(node, Array):
            return "array"
        if isinstance(node, Object):
            return "object"
        if isinstance(node, Reference):
            return "reference"
        return "type"

    def _reject_annotation(self, node: object, loc: Token, annotation: str) -> None:
        """Raise when an annotation has no legal home on this base type (§6.5)."""
        raise ParseError(
            f"{annotation} cannot attach to a {self._type_label(node)} type (§6.5)",
            loc.line,
            loc.col,
        )

    def _apply_annotations(self, node: object, ann: _Annotations) -> None:
        """Write a parsed annotation chain onto a node's NAMED fields (decision 1).

        Raises ParseError if any present annotation is forbidden for this base
        type under §6.5 (structural: nowhere to attach). :format is allowed on
        every base type except `null` and `any` (§5.2 / §6.5).
        """
        if isinstance(node, (Null, AnyType)) and ann.has_any():
            loc = (
                ann.format_loc
                or ann.bounds_loc
                or ann.mult_loc
                or ann.length_loc
                or ann.regex_loc
                or ann.encoding_loc
                or ann.media_loc
                or ann.unique_loc
                or self._peek()
            )
            self._reject_annotation(node, loc, "annotation")

        numeric_ok = isinstance(node, (Integer, Number))
        string_ok = isinstance(node, String)
        array_ok = isinstance(node, Array)
        if ann.bounds is not None:
            loc = ann.bounds_loc or self._peek()
            lower, upper, excl_lo, excl_hi = ann.bounds
            if array_ok:
                assert isinstance(node, Array)
                node.min_items = lower
                node.max_items = upper
            elif numeric_ok:
                assert isinstance(node, (Integer, Number))
                node.minimum = lower
                node.maximum = upper
                node.exclusive_min = excl_lo
                node.exclusive_max = excl_hi
            else:
                self._reject_annotation(node, loc, "numeric bound")

        if ann.mult is not None:
            loc = ann.mult_loc or self._peek()
            if numeric_ok:
                assert isinstance(node, (Integer, Number))
                node.multiple_of = ann.mult
            else:
                self._reject_annotation(node, loc, "%divisor")

        if ann.length is not None:
            loc = ann.length_loc or self._peek()
            if string_ok:
                assert isinstance(node, String)
                node.min_length, node.max_length = ann.length
            else:
                self._reject_annotation(node, loc, ":length")

        if ann.regex is not None:
            loc = ann.regex_loc or self._peek()
            if string_ok:
                assert isinstance(node, String)
                node.pattern = ann.regex
            else:
                self._reject_annotation(node, loc, ":regex")

        if ann.encoding is not None:
            loc = ann.encoding_loc or self._peek()
            if string_ok:
                assert isinstance(node, String)
                node.encoding = ann.encoding
            else:
                self._reject_annotation(node, loc, ":encoding")

        if ann.media is not None:
            loc = ann.media_loc or self._peek()
            if string_ok:
                assert isinstance(node, String)
                node.media = ann.media
            else:
                self._reject_annotation(node, loc, ":media")

        if ann.unique:
            loc = ann.unique_loc or self._peek()
            if array_ok:
                assert isinstance(node, Array)
                node.unique = True
            else:
                self._reject_annotation(node, loc, ":unique")

        if ann.format is not None:
            # :format attaches to any base type (§6.1); every base node has .format.
            node.format = ann.format  # type: ignore[attr-defined]


def parse(tokens: list[Token]) -> Document:
    """Parse a lexer token list into a `Document` AST (the module entry point).

    Pure structure: builds permissively and raises `ParseError` only on
    structurally un-parseable input. Semantic legality is validate.py's job.
    """
    return _Parser(tokens).parse_document()


def _split_document_blocks(text: str) -> list[str]:
    """Split a document into top-level blocks separated by blank lines (§3.1)."""
    stripped = text.strip()
    if not stripped:
        return []
    blocks: list[str] = []
    current: list[str] = []
    for line in stripped.split("\n"):
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _parse_raw_json_tool_block(block: str) -> dict:
    """Parse one tool-position raw JSON Schema object (§7.5).

    Distinction rule: a top-level block whose first non-space character is `{`
    is a raw fallback tool on a single physical line at column 0 — the scanner
    does not descend into the JSON object.
    """
    lines = block.split("\n")
    if len(lines) > 1 and any(line.strip() for line in lines[1:]):
        raise ParseError(
            "raw JSON Schema tool must be a single physical line (§7.5)",
            2,
            1,
        )
    stripped = lines[0].strip()
    if not stripped.startswith("{"):
        raise ParseError(
            "internal error: raw JSON tool block must start with '{'",
            1,
            1,
        )
    try:
        obj, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f"invalid raw JSON Schema tool object: {exc.msg}",
            1,
            (exc.pos or 0) + 1,
        ) from exc
    if not isinstance(obj, dict):
        raise ParseError(
            "raw JSON Schema tool must be a JSON object",
            1,
            1,
        )
    if stripped[end:].strip():
        raise ParseError(
            "trailing content after raw JSON Schema tool object",
            1,
            end + 1,
        )
    return obj


def parse_text(text: str) -> Document:
    """Parse a CATS document: CATS tool blocks and/or raw JSON Schema tools (§7.5).

    A document is a sequence of top-level blocks separated by blank lines: an
    optional `$defs` block, then one or more tools. Each tool is either a CATS
    tool block (name at column 0) or a raw JSON Schema object (``{`` at column 0).
    """
    from lexer import tokenize

    blocks = _split_document_blocks(text)
    defs: Optional[list[Definition]] = None
    tools: list[ToolBlock | RawSchema] = []

    for block in blocks:
        first_line = block.lstrip().split("\n", 1)[0].lstrip()
        if first_line.startswith("{"):
            tools.append(RawSchema(schema=_parse_raw_json_tool_block(block)))
            continue
        segment = parse(tokenize(block))
        if segment.defs is not None:
            if defs is not None:
                first = block.split("\n", 1)[0]
                raise ParseError(
                    "multiple '$defs' blocks in one document",
                    1,
                    1,
                )
            defs = segment.defs
        tools.extend(segment.tools)

    return Document(tools=tools, defs=defs)
