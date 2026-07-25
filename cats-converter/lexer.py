"""
lexer.py — the CATS tokenizer (raw text -> a flat list of tokens).

This is the FIRST stage of the strict pipeline:

    text --[lexer]--> tokens --[parser]--> AST --[validate]--> AST --[serializer]--> text

Scope (deliberately narrow): this module turns raw CATS text into a flat list of
`Token`s and nothing more. It does NOT parse, build an AST, or judge any semantic
legality. In particular, it never decides whether a bare word is a *type word*, a
*field name*, or a *literal value* — that classification is position-dependent
(the spec itself says so for `null`, §2.5) and therefore belongs to the parser.
The lexer's contract is: characters in, tokens out, errors only for things that
are broken at the *lexical* level (bad indentation, stray characters, an
unterminated string).

Spec basis:
  - §2   lexical structure (character set, whitespace/indentation, identifiers,
         reserved words, literals, quoting),
  - §3-6 the structural delimiters that appear in document/field/type/annotation
         syntax,
  - Appendix A.5 the authoritative list of lexical terminals.

The one genuinely hard part is INDENTATION (§2.2, §3.4): two spaces per level,
tabs forbidden, anything not a multiple of two is ill-formed. We handle it the
way Python's own tokenizer does — an indentation stack that emits synthetic
INDENT / DEDENT tokens when the level rises or returns to an enclosing level.

DESIGN NOTE — keyword recognition is left to the parser (surface this if it
matters): the eight type words (§2.4), `$defs`, and the bare literals
`true` / `false` / `null` (§2.5) all match the identifier grammar, so the lexer
emits a single `IDENT` token for every bare word and lets the parser interpret
it by position against the closed reserved-word set. This keeps the lexer
context-free and is the only safe choice for `null`, which §2.5 explicitly says
is the literal or the type word *depending on context*. Quoting already gives the
lexer the one distinction it CAN see: a quoted value is a STRING, a bare word is
an IDENT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """The lexical categories the lexer can emit.

    Grouped by role. Section references point at where each element is defined
    in spec.md / Appendix A.
    """

    # --- Layout / structural virtual tokens (Appendix A.5) -----------------
    INDENT = auto()        # indentation rose one (or more) levels (§2.2, §3.4)
    DEDENT = auto()        # indentation returned to an enclosing level (§3.4)
    NEWLINE = auto()       # end of a non-blank logical line (§2.1)
    EOF = auto()           # end of input (convenience marker for the parser)

    # --- Words and literals (§2.3, §2.4, §2.5) -----------------------------
    IDENT = auto()         # any bare word: name / identifier / type word /
                           #   true / false / null. The parser classifies it.
    STRING = auto()        # quoted string literal, escapes decoded (§2.5)
    NUMBER = auto()        # JSON number lexeme, kept as raw text (§2.5)
    JSON_LITERAL = auto()  # inline JSON object/array default, opaque (§4.5, A.5)
    DESCRIPTION = auto()   # free text from after '#' to end of line (§4.6)

    # --- Delimiters used across §3-§6 --------------------------------------
    DOLLAR = auto()        # '$'  reference / $defs marker (§2.3, §5.7)
    STAR = auto()          # '*'  required marker (§4.2)
    EQUALS = auto()        # '='  default introducer (§4.5)
    HASH = auto()          # '#'  description introducer (§4.6)
    PIPE = auto()          # '|'  union separator (§5.5, §5.6)
    COLON = auto()         # ':'  format and named annotations (§6.1, §6.3, §6.4)
    PERCENT = auto()       # '%'  multipleOf divisor (§6.2)
    COMMA = auto()         # ','  bound separator (§6.2-§6.4)
    LANGLE = auto()        # '<'  array element open (§5.3)
    RANGLE = auto()        # '>'  array element close (§5.3)
    LBRACKET = auto()      # '['  inclusive bound / annotation arg (§6.2-§6.4)
    RBRACKET = auto()      # ']'  inclusive bound / annotation arg (§6.2-§6.4)
    LPAREN = auto()        # '('  exclusive bound open (§6.2)
    RPAREN = auto()        # ')'  exclusive bound close (§6.2)


@dataclass
class Token:
    """One lexical token.

    `value` carries the token's textual content where that is meaningful:
    the decoded text for STRING, the raw lexeme for NUMBER / IDENT /
    JSON_LITERAL, the literal delimiter character for the punctuation tokens,
    and the raw text for DESCRIPTION. Layout tokens (INDENT/DEDENT/NEWLINE/EOF)
    carry an empty string.

    `line` and `col` are 1-based and mark where the token STARTS, so later
    stages can point error messages at a precise location (§task requirement).
    """

    type: TokenType
    value: str
    line: int
    col: int


class LexError(Exception):
    """A purely lexical error: bad indentation, a stray character, or an
    unterminated string. Carries a location so callers can report it.

    Semantic problems (illegal annotation/type pairings, required+default,
    union homogeneity, etc.) are NOT lexer errors — they are caught later in
    validate.py per the project's permissive-parser design.
    """

    def __init__(self, message: str, line: int, col: int) -> None:
        self.message = message
        self.line = line
        self.col = col
        super().__init__(f"{message} (line {line}, column {col})")


# Single-character delimiters that need no special look-around. Note that '#'
# and '=' are handled separately in the scan loop because each switches the
# lexer into a small line-tail mode (description text / glued default value).
_SINGLE_CHAR: dict[str, TokenType] = {
    "$": TokenType.DOLLAR,
    "*": TokenType.STAR,
    "|": TokenType.PIPE,
    ":": TokenType.COLON,
    "%": TokenType.PERCENT,
    ",": TokenType.COMMA,
    "<": TokenType.LANGLE,
    ">": TokenType.RANGLE,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
}

# A bare word: identifier start then identifier-continue OR hyphen (the `name`
# grammar of §2.3, the most permissive bare-word form). The parser narrows this
# to identifier-only where the grammar demands it (definition names, §3.2).
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")

# A number lexeme. Intentionally a touch more permissive than the strict JSON
# grammar of §2.5 (it tolerates leading zeros): the lexer grabs the maximal
# numeric run as one NUMBER token and leaves strict-form checking to later
# stages, consistent with the permissive-parser design.
_NUMBER_RE = re.compile(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def _read_string(line: str, start: int, line_no: int) -> tuple[str, int]:
    """Read a double-quoted string literal beginning at `line[start] == '"'`.

    Recognizes only the escapes `\\"` and `\\\\` (§2.5); a backslash that does
    not form one of those represents a literal backslash, since §2.5 says all
    other characters represent themselves. Returns the DECODED value and the
    index just past the closing quote. Raises LexError if the quote is never
    closed before end of line (strings do not span lines).
    """
    i = start + 1
    n = len(line)
    out: list[str] = []
    while i < n:
        c = line[i]
        if c == "\\":
            nxt = line[i + 1] if i + 1 < n else ""
            if nxt in ('"', "\\"):
                out.append(nxt)
                i += 2
                continue
            # Not a recognized escape: the backslash stands for itself (§2.5).
            out.append("\\")
            i += 1
            continue
        if c == '"':
            return "".join(out), i + 1
        out.append(c)
        i += 1
    raise LexError("unterminated string literal", line_no, start + 1)


def _read_json_literal(line: str, start: int, line_no: int) -> tuple[str, int]:
    """Read an opaque JSON object/array default glued after '=' (§4.5, A.5).

    The literal is one contiguous token; internal ``"`` strings (including spaces
    and ``/``) must not terminate the scan early. Nesting of ``{}`` / ``[]`` is
    tracked only outside JSON strings.
    """
    if start >= len(line) or line[start] not in "{[":
        raise LexError("json-literal must begin with '{' or '['", line_no, start + 1)

    depth = 0
    i = start
    n = len(line)
    in_string = False

    while i < n:
        c = line[i]
        if in_string:
            if c == "\\":
                if i + 1 >= n:
                    raise LexError("unterminated string in json-literal", line_no, i + 1)
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue

        if c == '"':
            in_string = True
            i += 1
            continue
        if c in "{[":
            depth += 1
            i += 1
            continue
        if c in "}]":
            depth -= 1
            i += 1
            if depth == 0:
                return line[start:i], i
            if depth < 0:
                raise LexError("unexpected closing bracket in json-literal", line_no, i)
            continue
        i += 1

    raise LexError("unterminated json-literal", line_no, start + 1)


def tokenize(text: str) -> list[Token]:
    """Convert raw CATS text into a flat list of tokens.

    Pure function: the same input always yields the same token list, and it has
    no side effects. Raises LexError on a lexical-level problem (bad
    indentation, a tab where a space is required, a stray character, an
    unterminated string).

    Indentation handling (§2.2, §3.4): leading spaces are measured per line;
    tabs in indentation are rejected; a count that is not a multiple of two is
    rejected. An indentation *stack* of levels drives synthetic INDENT/DEDENT
    emission — INDENT when the level rises, one DEDENT per level unwound when it
    falls, and a LexError if a dedent lands between two enclosing levels.
    """
    # §2.1: a BOM must not start the document.
    if text.startswith("\ufeff"):
        raise LexError(
            "byte-order mark (U+FEFF) must not appear at the start of a document",
            1,
            1,
        )

    tokens: list[Token] = []
    # The stack holds nesting LEVELS (0 = column 0). It starts with the base
    # level so a return to column 0 is just a dedent down to it.
    indent_stack: list[int] = [0]

    # Split on LF; a trailing CR on each piece handles CRLF, treated as
    # equivalent to LF (§2.1). A lone trailing "" (file ended with a newline)
    # is a blank line and is skipped below.
    lines = text.split("\n")

    for idx, raw_line in enumerate(lines):
        line_no = idx + 1
        line = raw_line[:-1] if raw_line.endswith("\r") else raw_line

        # §2.2: blank lines (empty or whitespace-only) have no effect on scope.
        if line.strip() == "":
            continue

        # --- Measure indentation -------------------------------------------
        n_spaces = 0
        while n_spaces < len(line) and line[n_spaces] == " ":
            n_spaces += 1
        # §2.2: indentation is spaces only; a tab in the indent region is fatal.
        if n_spaces < len(line) and line[n_spaces] == "\t":
            raise LexError(
                "tab characters must not be used for indentation", line_no, n_spaces + 1
            )
        # §2.2: indentation must be a whole number of two-space levels.
        if n_spaces % 2 != 0:
            raise LexError(
                "indentation must be a multiple of two spaces", line_no, 1
            )
        level = n_spaces // 2

        # --- Emit INDENT / DEDENT against the stack (§3.4) -----------------
        top = indent_stack[-1]
        if level > top:
            # A rise of more than one level is permitted here (one INDENT is
            # emitted, Python-style); whether a multi-level jump is structurally
            # legal is left to the parser.
            indent_stack.append(level)
            tokens.append(Token(TokenType.INDENT, "", line_no, n_spaces + 1))
        elif level < top:
            while indent_stack[-1] > level:
                indent_stack.pop()
                tokens.append(Token(TokenType.DEDENT, "", line_no, n_spaces + 1))
            if indent_stack[-1] != level:
                raise LexError(
                    "dedent does not match any enclosing indentation level",
                    line_no,
                    n_spaces + 1,
                )

        # --- Scan the rest of the line into content tokens -----------------
        i = n_spaces
        n = len(line)
        while i < n:
            c = line[i]
            col = i + 1

            # Inter-token separator: §2.2 admits only U+0020 between tokens.
            if c == " ":
                i += 1
                continue
            if c == "\t":
                raise LexError(
                    "tab character is not a valid token separator", line_no, col
                )

            # '#' opens a description that runs to end of line (§4.6). Whatever
            # follows is free prose and must NOT be tokenized further, so we
            # grab it whole. We drop a single separating space (the SP of
            # `SP "#" SP description-text`, A.2) but keep the raw text otherwise;
            # the parser handles the §2.6 quoting of descriptions.
            if c == "#":
                tokens.append(Token(TokenType.HASH, "#", line_no, col))
                j = i + 1
                if j < n and line[j] == " ":
                    j += 1
                tokens.append(Token(TokenType.DESCRIPTION, line[j:], line_no, j + 1))
                break

            # String literal (§2.5): may contain spaces and structural
            # characters, so it is read as one token up to its closing quote.
            if c == '"':
                value, end = _read_string(line, i, line_no)
                tokens.append(Token(TokenType.STRING, value, line_no, col))
                i = end
                continue

            # '=' introduces a default (§4.5). The value is glued to '='. Only
            # the object/array form needs special care: a json-literal (`{...}`
            # / `[...]`) is an opaque token (A.5) whose braces are not CATS
            # delimiters. Scalar defaults (string/number/bool/null) need no
            # special case — the normal scan handles them next.
            if c == "=":
                tokens.append(Token(TokenType.EQUALS, "=", line_no, col))
                i += 1
                if i < n and line[i] in "{[":
                    start = i
                    literal, i = _read_json_literal(line, start, line_no)
                    tokens.append(
                        Token(TokenType.JSON_LITERAL, literal, line_no, start + 1)
                    )
                continue

            # Number literal (§2.5). A '-' is only a number if a digit follows;
            # a lone '-' cannot begin any other token, so it is a lexical error.
            if c == "-" or c.isdigit():
                m = _NUMBER_RE.match(line, i)
                if m:
                    tokens.append(Token(TokenType.NUMBER, m.group(), line_no, col))
                    i = m.end()
                    continue
                raise LexError("'-' is not the start of a valid number", line_no, col)

            # Single-character delimiters.
            single = _SINGLE_CHAR.get(c)
            if single is not None:
                tokens.append(Token(single, c, line_no, col))
                i += 1
                continue

            # Bare word: name / identifier / type word / true / false / null.
            m = _IDENT_RE.match(line, i)
            if m:
                tokens.append(Token(TokenType.IDENT, m.group(), line_no, col))
                i = m.end()
                continue

            raise LexError(f"unexpected character {c!r}", line_no, col)

        # Every non-blank logical line ends with a NEWLINE token, even the last
        # line of a file that has no trailing newline — this gives the parser a
        # uniform line terminator.
        tokens.append(Token(TokenType.NEWLINE, "", line_no, n + 1))

    # Close any still-open blocks, then mark end of input.
    closing_line = len(lines) + 1
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token(TokenType.DEDENT, "", closing_line, 1))
    tokens.append(Token(TokenType.EOF, "", closing_line, 1))

    return tokens
