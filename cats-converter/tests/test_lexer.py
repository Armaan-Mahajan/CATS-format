"""Comprehensive unit tests for the CATS lexer (cats-converter/lexer.py).

Coverage map (lexical only — no parsing semantics):
  - Every TokenType and delimiter used in spec sections 2–6
  - Names, identifiers, type words, and literal forms (§2.3–§2.5)
  - Full annotation-chain punctuation (§6)
  - Type expressions: arrays, unions, references (§5)
  - Field-line pieces: required marker, defaults, descriptions (§4)
  - Document-shaped indentation: $defs, tools, deep nesting (§3)
  - Line endings, EOF dedents, locations
  - Lexical errors: BOM, tabs, odd indent, bad dedent, bad strings/chars
"""

from __future__ import annotations

import pytest

from lexer import LexError, Token, TokenType, tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def kinds(tokens: list[Token]) -> list[TokenType]:
    return [t.type for t in tokens]


def values(tokens: list[Token]) -> list[str]:
    return [t.value for t in tokens]


def content(tokens: list[Token]) -> list[Token]:
    """Drop layout tokens (INDENT/DEDENT/NEWLINE/EOF) for content-only checks."""
    layout = {
        TokenType.INDENT,
        TokenType.DEDENT,
        TokenType.NEWLINE,
        TokenType.EOF,
    }
    return [t for t in tokens if t.type not in layout]


def assert_sequence(tokens: list[Token], expected: list[tuple[TokenType, str]]) -> None:
    """Assert exact (type, value) pairs for the full token stream."""
    assert len(tokens) == len(expected), (
        f"length {len(tokens)} != {len(expected)}\n"
        f"got:      {[(t.type.name, t.value) for t in tokens]}\n"
        f"expected: {[(t.name, v) for t, v in expected]}"
    )
    for tok, (typ, val) in zip(tokens, expected, strict=True):
        assert tok.type is typ, f"expected {typ.name}, got {tok.type.name} ({tok.value!r})"
        assert tok.value == val, f"expected value {val!r}, got {tok.value!r}"


def assert_lex_error(text: str, *, line: int | None = None, substring: str | None = None) -> None:
    with pytest.raises(LexError) as exc_info:
        tokenize(text)
    err = exc_info.value
    if line is not None:
        assert err.line == line, f"expected line {line}, got {err.line}: {err}"
    if substring is not None:
        assert substring in err.message, f"{substring!r} not in {err.message!r}"


# ---------------------------------------------------------------------------
# §2.3 — names and identifiers (all emitted as IDENT unless quoted)
# ---------------------------------------------------------------------------


class TestIdentifiersAndNames:
    @pytest.mark.parametrize(
        "word",
        [
            "string",
            "integer",
            "number",
            "boolean",
            "array",
            "object",
            "null",
            "any",
            "true",
            "false",
            "defs",
        ],
    )
    def test_type_words_and_reserved_bare_words_are_ident(self, word: str) -> None:
        """Lexer does not classify keywords; parser assigns meaning (§2.4, §2.5)."""
        tokens = tokenize(f"x {word}\n")
        assert values(tokens)[:2] == ["x", word]
        assert tokens[1].type is TokenType.IDENT

    @pytest.mark.parametrize(
        "name",
        [
            "get-weather",
            "model-name",
            "user_id",
            "_private",
            "Address",
            "a",
            "CamelCase",
        ],
    )
    def test_name_shapes(self, name: str) -> None:
        tokens = tokenize(f"{name} string\n")
        assert tokens[0].value == name
        assert tokens[0].type is TokenType.IDENT

    def test_quoted_field_name_with_required_marker(self) -> None:
        tokens = tokenize('"user.id"* string\n')
        assert_sequence(
            tokens,
            [
                (TokenType.STRING, "user.id"),
                (TokenType.STAR, "*"),
                (TokenType.IDENT, "string"),
                (TokenType.NEWLINE, ""),
                (TokenType.EOF, ""),
            ],
        )

    def test_quoted_tool_header_name(self) -> None:
        tokens = tokenize('"my.tool" # contract-bound name\n  x string\n')
        assert tokens[0].type is TokenType.STRING
        assert tokens[0].value == "my.tool"
        assert tokens[1].type is TokenType.HASH


# ---------------------------------------------------------------------------
# §2.5 — string and number literals
# ---------------------------------------------------------------------------


class TestStringLiterals:
    def test_empty_string(self) -> None:
        tokens = tokenize('mode "automatic"\n')
        assert tokens[1].type is TokenType.STRING
        assert tokens[1].value == "automatic"

    def test_escape_quote_and_backslash(self) -> None:
        # Use a normal string so the input ends with a real newline, not literal \n.
        tokens = tokenize('msg string "say \\"hi\\" and \\\\ok"\n')
        s = next(t for t in tokens if t.type is TokenType.STRING)
        assert s.value == 'say "hi" and \\ok'

    def test_unrecognized_escape_keeps_backslash(self) -> None:
        tokens = tokenize('p string "a\\bc"\n')
        s = next(t for t in tokens if t.type is TokenType.STRING)
        assert s.value == "a\\bc"

    def test_string_with_spaces_and_structural_chars(self) -> None:
        tokens = tokenize('x string "[0,100] | a#b"\n')
        s = next(t for t in tokens if t.type is TokenType.STRING)
        assert s.value == "[0,100] | a#b"

    def test_unterminated_string_at_eol(self) -> None:
        assert_lex_error('x string "open\n', line=1, substring="unterminated")

    def test_string_default_after_equals(self) -> None:
        tokens = tokenize('label string ="quoted default"\n')
        eq_idx = kinds(tokens).index(TokenType.EQUALS)
        assert tokens[eq_idx + 1].type is TokenType.STRING
        assert tokens[eq_idx + 1].value == "quoted default"


class TestNumberLiterals:
    @pytest.mark.parametrize(
        "lexeme",
        ["0", "42", "-17", "3.14", "1e10", "1E10", "-2.5E-3"],
    )
    def test_number_forms(self, lexeme: str) -> None:
        tokens = tokenize(f"priority {lexeme}|other\n")
        # In enum position the number is still a NUMBER token.
        num_tok = next(t for t in tokens if t.type is TokenType.NUMBER)
        assert num_tok.value == lexeme

    def test_json_plus_prefix_not_lexed_as_number(self) -> None:
        """Lexer number grammar has no leading '+'; JSON §2.5 uses optional minus only."""
        assert_lex_error("priority +3|other\n", substring="unexpected")

    def test_numbers_in_numeric_bounds(self) -> None:
        tokens = tokenize("limit integer[1,100)\n")
        nums = [t for t in tokens if t.type is TokenType.NUMBER]
        assert [t.value for t in nums] == ["1", "100"]

    def test_lone_minus_is_lexical_error(self) -> None:
        assert_lex_error("x string |-\n", substring="not the start of a valid number")


# ---------------------------------------------------------------------------
# §4 — field lines: markers, defaults, descriptions
# ---------------------------------------------------------------------------


class TestFieldLinePieces:
    def test_required_marker_glued_to_name(self) -> None:
        tokens = tokenize("latitude* number\n")
        assert_sequence(
            tokens,
            [
                (TokenType.IDENT, "latitude"),
                (TokenType.STAR, "*"),
                (TokenType.IDENT, "number"),
                (TokenType.NEWLINE, ""),
                (TokenType.EOF, ""),
            ],
        )
        assert tokens[1].col == 9

    @pytest.mark.parametrize(
        "source,default_kind,default_value",
        [
            ("count integer =5", TokenType.NUMBER, "5"),
            ("flag boolean =true", TokenType.IDENT, "true"),
            ("opt string =null", TokenType.IDENT, "null"),
            ('name string ="x"', TokenType.STRING, "x"),
            ("ratio number =3.14", TokenType.NUMBER, "3.14"),
        ],
    )
    def test_scalar_defaults(
        self, source: str, default_kind: TokenType, default_value: str
    ) -> None:
        tokens = tokenize(source + "\n")
        eq = kinds(tokens).index(TokenType.EQUALS)
        assert tokens[eq + 1].type is default_kind
        assert tokens[eq + 1].value == default_value

    @pytest.mark.parametrize(
        "literal",
        ["={}", "=[]", '={"k":1}', "=[1,2,3]"],
    )
    def test_json_literal_defaults_are_opaque(self, literal: str) -> None:
        tokens = tokenize(f"cfg object {literal}\n")
        json_tok = next(t for t in tokens if t.type is TokenType.JSON_LITERAL)
        assert json_tok.value == literal[1:]  # strip leading '='

    def test_array_type_with_json_default_and_leading_space_before_equals(self) -> None:
        """§4.5: space before '=' prevents misreading '>='."""
        tokens = tokenize("items array<string> =[]\n")
        assert kinds(tokens)[:-2] == [
            TokenType.IDENT,
            TokenType.IDENT,
            TokenType.LANGLE,
            TokenType.IDENT,
            TokenType.RANGLE,
            TokenType.EQUALS,
            TokenType.JSON_LITERAL,
        ]
        assert tokens[6].value == "[]"

    def test_description_with_canonical_hash_space(self) -> None:
        tokens = tokenize("limit integer # between 1 and 100\n")
        assert tokens[2].type is TokenType.HASH
        assert tokens[3].type is TokenType.DESCRIPTION
        assert tokens[3].value == "between 1 and 100"

    def test_description_without_space_after_hash(self) -> None:
        tokens = tokenize("limit integer #tight\n")
        assert tokens[3].value == "tight"

    def test_description_empty_after_hash(self) -> None:
        tokens = tokenize("x string #\n")
        assert tokens[3].type is TokenType.DESCRIPTION
        assert tokens[3].value == ""

    def test_description_swallows_delimiters(self) -> None:
        tokens = tokenize("x integer # uses | pipes [and] (parens) :colons\n")
        assert "|" not in [t.value for t in content(tokens)]
        assert tokens[3].value == "uses | pipes [and] (parens) :colons"

    def test_quoted_description_stays_raw(self) -> None:
        """§4.6 seam (DECISION): when a description must be quoted because it
        contains '#' (§2.6 trigger 3), the lexer emits the DESCRIPTION token
        with its surrounding quotes INCLUDED, as raw text. It does NOT decode it
        as a STRING.

        Rationale: a STRING is quoted under §2.6 for structural reasons (escapes
        decoded, treated as one value-unit); a description is quoted under §4.6
        purely to disambiguate a literal '#', and §4.6 treats description content
        as uninterpreted free-form prose. Applying STRING escape-decoding to a
        description would wrongly transform literal backslashes in prose. So
        quote-stripping for descriptions is the PARSER's job (a §4.6 rule),
        keeping the lexer honest to 'descriptions are raw, grabbed whole'.

        This test pins that contract: the parser depends on receiving the raw,
        quote-included text here, so a future refactor must not silently start
        decoding it.
        """
        tokens = tokenize('x string # "weight in kg (#2 priority)"\n')
        desc = next(t for t in tokens if t.type is TokenType.DESCRIPTION)
        # Quotes are PRESENT and the inner '#' is preserved verbatim.
        assert desc.value == '"weight in kg (#2 priority)"'
        # And it is a DESCRIPTION, not decoded into a STRING token.
        assert desc.type is TokenType.DESCRIPTION
        # A literal backslash inside a quoted description is left untouched
        # (NOT escape-decoded the way a STRING token would be).
        tokens2 = tokenize('p string # "path C:\\temp #1"\n')
        desc2 = next(t for t in tokens2 if t.type is TokenType.DESCRIPTION)
        assert desc2.value == '"path C:\\temp #1"'

    def test_hash_in_type_expression_before_description(self) -> None:
        """Only the field-line '# description' is swallowed; not earlier tokens."""
        tokens = tokenize("x string:email # contact\n")
        assert TokenType.COLON in kinds(tokens)
        hash_tok = next(t for t in tokens if t.type is TokenType.HASH)
        desc_tok = next(t for t in tokens if t.type is TokenType.DESCRIPTION)
        assert desc_tok.value == "contact"
        assert tokens.index(hash_tok) < tokens.index(desc_tok)


# ---------------------------------------------------------------------------
# §5 — type expressions (token sequences only)
# ---------------------------------------------------------------------------


class TestTypeExpressionTokens:
    def test_parameterized_array(self) -> None:
        tokens = tokenize("tags array<string>\n")
        assert kinds(tokens)[:-2] == [
            TokenType.IDENT,
            TokenType.IDENT,
            TokenType.LANGLE,
            TokenType.IDENT,
            TokenType.RANGLE,
        ]

    def test_nested_array_parameterization(self) -> None:
        tokens = tokenize("matrix array<array<integer>>\n")
        assert values(tokens)[:2] == ["matrix", "array"]
        assert kinds(tokens).count(TokenType.LANGLE) == 2
        assert kinds(tokens).count(TokenType.RANGLE) == 2

    def test_type_union_pipes(self) -> None:
        tokens = tokenize("query string|array<string>|null\n")
        assert kinds(tokens).count(TokenType.PIPE) == 2

    def test_enum_union_values(self) -> None:
        tokens = tokenize("visibility public|private|default\n")
        body = content(tokens)
        assert [t.value for t in body] == [
            "visibility",
            "public",
            "|",
            "private",
            "|",
            "default",
        ]

    def test_single_value_enum_is_quoted_string(self) -> None:
        tokens = tokenize('mode "automatic"\n')
        assert tokens[1].type is TokenType.STRING

    def test_nullable_union_branch(self) -> None:
        tokens = tokenize("name string|null\n")
        body = content(tokens)
        assert [t.value for t in body] == ["name", "string", "|", "null"]

    def test_reference_sigil(self) -> None:
        tokens = tokenize("addr $Address\n")
        assert_sequence(
            tokens[:-2],
            [
                (TokenType.IDENT, "addr"),
                (TokenType.DOLLAR, "$"),
                (TokenType.IDENT, "Address"),
            ],
        )

    def test_reference_inside_array(self) -> None:
        tokens = tokenize("guests array<$Person>\n")
        dollar = next(i for i, t in enumerate(tokens) if t.type is TokenType.DOLLAR)
        assert tokens[dollar + 1].value == "Person"

    def test_defs_block_header(self) -> None:
        tokens = tokenize("$defs\n  Person\n    id integer\n")
        assert tokens[0].type is TokenType.DOLLAR
        assert tokens[1].value == "defs"


# ---------------------------------------------------------------------------
# §6 — annotation chain delimiters
# ---------------------------------------------------------------------------


class TestAnnotationChainTokens:
    def test_format_colon(self) -> None:
        tokens = tokenize("t string:date-time\n")
        body = content(tokens)
        assert [t.type for t in body] == [
            TokenType.IDENT,
            TokenType.IDENT,
            TokenType.COLON,
            TokenType.IDENT,
        ]
        assert body[-1].value == "date-time"

    def test_numeric_bounds_all_bracket_styles(self) -> None:
        tokens = tokenize("a integer[1,100]\n")
        assert (TokenType.LBRACKET, "[") in [(t.type, t.value) for t in tokens]
        assert (TokenType.RBRACKET, "]") in [(t.type, t.value) for t in tokens]

        tokens = tokenize("b number(0,1)\n")
        assert (TokenType.LPAREN, "(") in [(t.type, t.value) for t in tokens]
        assert (TokenType.RPAREN, ")") in [(t.type, t.value) for t in tokens]

        # Mixed: inclusive lower '[', exclusive upper ')' — no ']' token (§6.2)
        tokens = tokenize("c integer[0,100)\n")
        assert kinds(tokens).count(TokenType.LBRACKET) == 1
        assert kinds(tokens).count(TokenType.RPAREN) == 1
        assert kinds(tokens).count(TokenType.RBRACKET) == 0

        tokens = tokenize("d integer[1,)\n")
        assert kinds(tokens).count(TokenType.COMMA) >= 1

    def test_multiple_of_percent(self) -> None:
        """Open upper bound uses ')' before '%' glues to the closing bracket (§6.2)."""
        tokens = tokenize("qty integer[1,)%5\n")
        assert kinds(tokens)[:-2] == [
            TokenType.IDENT,
            TokenType.IDENT,
            TokenType.LBRACKET,
            TokenType.NUMBER,
            TokenType.COMMA,
            TokenType.RPAREN,
            TokenType.PERCENT,
            TokenType.NUMBER,
        ]

    def test_string_length_regex_encoding_media(self) -> None:
        line = (
            'u string:length[1,20]:regex["^[a-z]+$"]:encoding[base64]:media["application/pdf"]\n'
        )
        tokens = tokenize(line)
        assert TokenType.COLON in kinds(tokens)
        assert any(t.value == "length" for t in tokens if t.type is TokenType.IDENT)
        assert any(t.type is TokenType.STRING and "^" in t.value for t in tokens)
        assert "base64" in values(tokens)

    def test_array_bounds_and_unique(self) -> None:
        tokens = tokenize("emails array<string>[1,]:unique\n")
        assert kinds(tokens)[:-2] == [
            TokenType.IDENT,
            TokenType.IDENT,
            TokenType.LANGLE,
            TokenType.IDENT,
            TokenType.RANGLE,
            TokenType.LBRACKET,
            TokenType.NUMBER,
            TokenType.COMMA,
            TokenType.RBRACKET,
            TokenType.COLON,
            TokenType.IDENT,
        ]
        assert tokens[-3].value == "unique"

    def test_annotation_chain_on_union_branch(self) -> None:
        tokens = tokenize("count integer[0,100]|null\n")
        assert kinds(tokens).index(TokenType.PIPE) > kinds(tokens).index(TokenType.RBRACKET)


# ---------------------------------------------------------------------------
# §3 — document structure and indentation
# ---------------------------------------------------------------------------


class TestIndentation:
    def test_nested_tool_and_object_blocks(self) -> None:
        text = (
            "get-weather\n"
            "  location object\n"
            "    city string\n"
            "  units string\n"
        )
        seq = kinds(tokenize(text))
        assert seq.count(TokenType.INDENT) == 2
        assert seq.count(TokenType.DEDENT) == 2
        assert seq[-1] is TokenType.EOF

    def test_defs_block_three_column_levels(self) -> None:
        """$defs at 0, definition header at 2, body at 4 (§3.4)."""
        text = (
            "$defs\n"
            "  Address\n"
            "    street string\n"
            "    city string\n"
            "get-tool\n"
            "  home $Address\n"
        )
        tokens = tokenize(text)
        assert tokens[0].type is TokenType.DOLLAR
        # Three opens: defs body, definition body, tool body — then matching closes.
        assert kinds(tokens).count(TokenType.INDENT) == 3
        assert kinds(tokens).count(TokenType.DEDENT) == 3

    def test_deep_four_level_nesting(self) -> None:
        text = "t\n  a object\n    b object\n      c object\n        d string\n"
        tokens = tokenize(text)
        # Four nested bodies (cols 2/4/6/8) => four INDENTs, four DEDENTs at EOF.
        assert kinds(tokens).count(TokenType.INDENT) == 4
        assert kinds(tokens).count(TokenType.DEDENT) == 4

    def test_eof_closes_all_open_blocks(self) -> None:
        text = "tool\n  x string\n    y string\n"
        tokens = tokenize(text)
        # Two INDENTs while reading; EOF emits two DEDENTs before EOF token.
        tail = kinds(tokens)[-5:]
        assert tail.count(TokenType.DEDENT) == 2
        assert tail[-1] is TokenType.EOF

    def test_blank_and_whitespace_only_lines_ignored(self) -> None:
        text = "a string\n\n   \n  \nb string\n"
        assert kinds(tokenize(text)).count(TokenType.INDENT) == 0

    def test_no_trailing_newline_still_emits_newline_and_eof(self) -> None:
        tokens = tokenize("only string")
        assert kinds(tokens)[-2:] == [TokenType.NEWLINE, TokenType.EOF]

    def test_crlf_line_endings(self) -> None:
        tokens = tokenize("a string\r\nb string\r\n")
        assert values(tokens)[0] == "a"
        assert kinds(tokens).count(TokenType.NEWLINE) == 2

    def test_trailing_spaces_on_line_are_part_of_content_scan(self) -> None:
        """§2.2: trailing whitespace permitted; still tokenized if non-space chars."""
        tokens = tokenize("name string  \n")
        assert tokens[0].value == "name"
        assert kinds(tokens)[-2:] == [TokenType.NEWLINE, TokenType.EOF]

    @pytest.mark.parametrize("spaces", [1, 3, 5, 7])
    def test_odd_indentation_rejected(self, spaces: int) -> None:
        pad = " " * spaces
        assert_lex_error(f"tool\n{pad}field string\n", line=2, substring="multiple of two")

    def test_tab_in_indent_region(self) -> None:
        assert_lex_error("tool\n\tfield string\n", substring="tab")

    def test_tab_as_separator_in_content(self) -> None:
        assert_lex_error("tool\n  a\tstring\n", line=2, substring="separator")

    def test_unmatched_dedent_level(self) -> None:
        assert_lex_error("a\n    b string\n  c string\n", substring="dedent")

    def test_single_step_indent_columns_tracked(self) -> None:
        tokens = tokenize("t\n  f string\n")
        indent = next(t for t in tokens if t.type is TokenType.INDENT)
        assert indent.line == 2
        assert indent.col == 3  # first content char after two spaces


# ---------------------------------------------------------------------------
# Realistic multi-construct lines
# ---------------------------------------------------------------------------


class TestRealisticFragments:
    def test_full_field_line_from_spec_examples(self) -> None:
        line = 'username* string:length[1,20]:regex["^[a-z0-9_]+$"] ="" # handle\n'
        tokens = tokenize(line)
        assert tokens[0].value == "username"
        assert TokenType.STAR in kinds(tokens)
        assert any(t.type is TokenType.STRING and "^" in t.value for t in tokens)
        assert TokenType.HASH in kinds(tokens)
        assert tokens[-3].type is TokenType.DESCRIPTION

    def test_tool_block_with_description_and_nested_fields(self) -> None:
        text = (
            "search # full-text search\n"
            "  query* string\n"
            "  limit integer =10\n"
        )
        tokens = tokenize(text)
        assert tokens[0].value == "search"
        assert tokens[1].type is TokenType.HASH
        assert tokens[2].value == "full-text search"
        assert kinds(tokens).count(TokenType.INDENT) == 1
        assert any(t.value == "limit" and t.type is TokenType.IDENT for t in tokens)

    def test_minimal_document_shape(self) -> None:
        text = (
            "$defs\n"
            "  Foo\n"
            "    x integer\n"
            "my-tool\n"
            "  y string\n"
        )
        tokens = tokenize(text)
        assert tokens[-1].type is TokenType.EOF
        assert kinds(tokens).count(TokenType.NEWLINE) == 5


# ---------------------------------------------------------------------------
# §2.1 — encoding and lexical errors
# ---------------------------------------------------------------------------


class TestLexicalErrors:
    def test_bom_rejected(self) -> None:
        assert_lex_error("\ufefftool\n", line=1, substring="byte-order mark")

    def test_unexpected_dot_in_bare_word(self) -> None:
        assert_lex_error("user.name string\n", substring="unexpected")

    def test_unexpected_at_sign(self) -> None:
        assert_lex_error("x @y\n", substring="unexpected")

    def test_unexpected_unicode_outside_quotes(self) -> None:
        assert_lex_error("名前 string\n", substring="unexpected")

    def test_empty_input_yields_only_eof(self) -> None:
        tokens = tokenize("")
        # No lines to scan; EOF is emitted on the synthetic closing line (line 2).
        assert tokens == [Token(TokenType.EOF, "", 2, 1)]

    def test_only_blank_lines_yield_only_eof(self) -> None:
        tokens = tokenize("\n\n  \n")
        assert kinds(tokens) == [TokenType.EOF]


# ---------------------------------------------------------------------------
# Token locations
# ---------------------------------------------------------------------------


class TestLocations:
    def test_first_token_at_line_one_column_one(self) -> None:
        tokens = tokenize("alpha string\n")
        assert (tokens[0].line, tokens[0].col) == (1, 1)

    def test_indented_field_starts_after_spaces(self) -> None:
        tokens = tokenize("tool\n  field string\n")
        field = next(t for t in tokens if t.value == "field")
        assert field.line == 2
        assert field.col == 3

    def test_string_token_column_at_opening_quote(self) -> None:
        tokens = tokenize('x "y"\n')
        s = next(t for t in tokens if t.type is TokenType.STRING)
        assert s.col == 3

    def test_eof_dedent_uses_synthetic_closing_line(self) -> None:
        """DEDENTs emitted at EOF use line len(lines)+1, not the last content line."""
        tokens = tokenize("t\n  a string\n")
        dedent = next(t for t in tokens if t.type is TokenType.DEDENT)
        # Trailing newline adds an extra split segment; EOF/dedent use len(lines)+1.
        assert dedent.line == 4
        assert dedent.col == 1


# ---------------------------------------------------------------------------
# Delimiter inventory — every punctuation token appears
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional corner cases and lexer-policy boundaries."""

    def test_single_indent_emitted_for_multi_level_jump(self) -> None:
        """Lexer emits one INDENT per rise (Python-style); legality is for parser."""
        text = "tool\n    deep string\n"
        tokens = tokenize(text)
        assert kinds(tokens).count(TokenType.INDENT) == 1

    def test_leading_zero_number_lexed_permissively(self) -> None:
        tokens = tokenize("x 007|other\n")
        assert next(t for t in tokens if t.type is TokenType.NUMBER).value == "007"

    def test_quoted_enum_member_colliding_with_type_word(self) -> None:
        tokens = tokenize('kind "string"\n')
        assert tokens[1].type is TokenType.STRING
        assert tokens[1].value == "string"

    def test_equals_with_no_value_at_end_of_line(self) -> None:
        tokens = tokenize("x string =\n")
        assert kinds(tokens)[:-2] == [TokenType.IDENT, TokenType.IDENT, TokenType.EQUALS]

    def test_json_literal_with_nested_braces(self) -> None:
        tokens = tokenize('cfg object ={"a":{"b":1}}\n')
        lit = next(t for t in tokens if t.type is TokenType.JSON_LITERAL)
        assert lit.value == '{"a":{"b":1}}'

    def test_json_literal_with_spaces_inside_string_values(self) -> None:
        tokens = tokenize(
            'headers object ={"Authorization":"Bearer your_api_key_here","Content-Type":"application/json"}\n'
        )
        lit = next(t for t in tokens if t.type is TokenType.JSON_LITERAL)
        assert lit.value == (
            '{"Authorization":"Bearer your_api_key_here","Content-Type":"application/json"}'
        )

    def test_json_literal_array_default_with_space_in_string(self) -> None:
        tokens = tokenize('data_field array<string> =["Personal Info"]\n')
        lit = next(t for t in tokens if t.type is TokenType.JSON_LITERAL)
        assert lit.value == '["Personal Info"]'

    def test_description_on_tool_header(self) -> None:
        tokens = tokenize("my-tool # does things\n  x string\n")
        assert tokens[0].value == "my-tool"
        assert tokens[1].type is TokenType.HASH
        assert tokens[2].value == "does things"

    def test_pipe_inside_quoted_string_default_not_a_union(self) -> None:
        tokens = tokenize('mode string ="a|b"\n')
        assert kinds(tokens).count(TokenType.PIPE) == 0

    def test_percent_only_without_bounds(self) -> None:
        tokens = tokenize("amount number%0.01\n")
        assert kinds(tokens)[:-2] == [
            TokenType.IDENT,
            TokenType.IDENT,
            TokenType.PERCENT,
            TokenType.NUMBER,
        ]

    def test_open_lower_bound_only(self) -> None:
        tokens = tokenize("count integer[,100]\n")
        nums = [t for t in tokens if t.type is TokenType.NUMBER]
        assert [t.value for t in nums] == ["100"]

    def test_open_upper_bound_only(self) -> None:
        tokens = tokenize("count integer[1,]\n")
        nums = [t for t in tokens if t.type is TokenType.NUMBER]
        assert [t.value for t in nums] == ["1"]


class TestDelimiterInventory:
    def test_all_single_char_delimiters_in_one_line(self) -> None:
        """Smoke line that forces every punctuation token type to appear."""
        # Not valid CATS semantically; valid lexically.
        line = "f $x*a|b:c%d,e<f>g[(h]i)j # tail\n"
        tokens = tokenize(line)
        expected_types = {
            TokenType.DOLLAR,
            TokenType.STAR,
            TokenType.PIPE,
            TokenType.COLON,
            TokenType.PERCENT,
            TokenType.COMMA,
            TokenType.LANGLE,
            TokenType.RANGLE,
            TokenType.LBRACKET,
            TokenType.RPAREN,
            TokenType.LPAREN,
            TokenType.RBRACKET,
            TokenType.HASH,
        }
        found = {t.type for t in tokens}
        assert expected_types <= found