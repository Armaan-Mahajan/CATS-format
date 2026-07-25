# 1. Introduction

## 1.1 Purpose and scope

CATS *(Compact Agent Tool Schema)* is a notation for the tool-definition payload that host applications send to large language models in tool-calling APIs. It is a **more content-dense** alternative to JSON Schema in this specific position of the tool-calling request, with the purpose of reducing the number of tokens used to communicate LLM tool definitions.

CATS targets one stage of the tool-calling lifecycle: the outbound request that informs the model which tools are available and how their inputs are shaped. The arguments a model emits when invoking a tool, and the results a host returns from executing a tool, remain JSON. Host implementations that parse model-emitted tool calls or inject tool results into subsequent requests are unaffected by CATS.

A CATS document encodes a complete tool definition — the tool's name, its natural-language description, and its input-parameter schema — as a single block. CATS is not a partial encoding that wraps a JSON-Schema-encoded parameter schema inside outer notation; the entire tool definition is expressed in CATS.

CATS is **not** a general replacement for JSON Schema outside the tool-calling context, a wire format for tool-call arguments or results, or a notation that requires model fine-tuning to be read correctly — it can simply be included in the system prompt with only short context being provided to the model.

## 1.2 Design goals

CATS is designed against three goals: **compression**, **fidelity**, and **model-readability**. The three are partially in tension, and the notation's design trades one against another where they conflict. Compression represents the primary benefit of CATS.

The compression goal is to *eliminate structural redundancy*. JSON (and JSON Schema) are designed for computer interpretation, but are verbose in nature — language models do not need such high levels of verbosity to behave correctly. Furthermore, by dropping redundant syntax from JSON, input token costs may be significantly decreased. Thus, compression also represents a larger economic benefit for individuals using LLMs. However, compression in CATS is not equivalent to arbitrary shortening; the target is information the model does not use, not condensation of information the model does use.

The fidelity goal is *behavior-preserving* rather than *keyword-preserving*. A schema converted from JSON Schema to CATS and back produces a JSON Schema that validates the same set of values as the original, though the surface form of the recovered schema may differ. This behavior-preservation is unconditional: every construct CATS encodes round-trips with its accepted value set intact. Constructs whose only faithful encoding would change the accepted value set are not encoded at all — they are out of scope and take the raw JSON Schema fallback path (§7.5), where they round-trip verbatim. The remaining round-trip losses are purely keyword-level — the surface keyword form is not recovered, but validation behavior is identical — and are documented in §7.4. Two explicitly opt-in converter options (§7.7), both disabled by default, re-interpret specific input conventions before conversion; the guarantee then holds with respect to the normalized input, and the default conversion is unaffected.

The model-readability goal is for frontier large language models to read CATS correctly with no fine-tuning and no extended priming — only short, in-prompt context. This constrains every notational choice: characters, delimiters, and structural conventions are drawn from patterns the model has already seen in its training data, and novel patterns are avoided where an existing convention conveys the same information.

Where these goals conflict, CATS treats model-readability as an upper bound on how aggressively compression can be pursued, and concedes only keyword-level fidelity, in the cases documented in §7.4. Behavioral fidelity is never conceded: a construct that cannot be encoded without changing its accepted value set is placed out of scope (§7.5) rather than converted lossily.

## 1.3 Relationship to JSON Schema

CATS is defined relative to **JSON Schema draft 2020-12**. Every well-formed CATS document corresponds to a JSON Schema document under this draft that validates the same set of values, modulo the keyword-level differences documented in §7.4, and every CATS-encodable JSON Schema document has a CATS form.

CATS is a re-encoding of JSON Schema, not a replacement. It is neither a strict subset nor a strict superset: CATS encodes the subset of JSON Schema constructs that real-world tool definitions actually use, and defers the remainder to a raw JSON Schema fallback path specified in §7.5. Constructs that fall outside the encoded subset are not expressible in CATS notation; a tool containing one is carried whole as raw JSON Schema rather than encoded.

CATS also accepts input from similar notations — earlier JSON Schema drafts, OpenAPI 3.0, and OpenAPI 3.1 — through converter-side transformations rather than separate notation. These transformations normalize input variants into canonical CATS without introducing additional surface forms; the full handling is specified in §7.6.

## 1.4 Document conventions

Technical terms specific to CATS — for example, *tool block*, *field line*, *type slot*, and *annotation chain* — are defined at first use in the section that introduces the corresponding construct.

The grammar fragments in this specification, and the complete grammar in Appendix A, are written in the W3C EBNF dialect used by W3C specifications such as XML and HTML. The dialect uses `::=` for production rules, `|` for alternation, parentheses for grouping, `?` `*` `+` for optional and repeated terms, and quoted strings for terminal symbols.

Conversions between JSON Schema and CATS are shown using paired code blocks, labeled `json` for JSON Schema input or output and `cats` for CATS notation.

The key words "**MUST**", "**MUST NOT**", "**SHOULD**", "**SHOULD NOT**", and "**MAY**" in this document are to be interpreted as described in BCP 14 [RFC2119] [RFC8174] when, and only when, they appear in all capitals, as shown here. The full BCP 14 vocabulary includes additional synonyms (e.g., *REQUIRED*, *SHALL*, *RECOMMENDED*, *OPTIONAL*); these are not used here — each remaining keyword serves as the canonical form for its semantic level.

## 1.5 Deployment (non-normative)

This specification defines the CATS notation and its conversion semantics. It does not specify how a CATS document is delivered to a model at request time. Because CATS rides in the system prompt as plain text rather than on a provider's native `tools` channel (§1.1), a deploying application must additionally give the model brief in-prompt context for reading CATS and a convention for reporting a tool call. How to do this — a short *primer* placed ahead of the tool definitions, the placement of the tool blocks themselves, and the model's output contract — is described in the companion *CATS Usage Protocol*. That document is **non-normative**: it records what works in practice, its recommendations are not conformance requirements, and an application MAY substitute its own framing. Nothing in this specification depends on it. This subsection exists only to point a reader of the spec toward the deployment guidance the "short, in-prompt context" of §1.1 refers to.

# 2. Lexical structure

## 2.1 Character set and encoding

A CATS document is a sequence of Unicode characters encoded in **UTF-8** (RFC 3629). Other encodings are not supported and MUST NOT be used.

Line endings MAY be either LF (`U+000A`) or CRLF (`U+000D U+000A`); a CATS parser treats the two as equivalent. A byte-order mark (BOM, `U+FEFF`) MUST NOT appear at the start of a document.

## 2.2 Whitespace and indentation

Whitespace in CATS serves two distinct structural roles: as an **inter-token separator** within a field line, and as **indentation** at the start of a line establishing nesting depth.

Within a field line, exactly one space character (`U+0020`) separates the field name (with its required marker, if present) from the type slot — the separator falls after the `*` when a `*` is present, as specified in §4.3. Additional structural whitespace (outside of quoted values) inside a field line is not part of the grammar. Whitespace inside a quoted string or within free-text content forms part of that content.

Indentation is composed exclusively of space characters; tab characters MUST NOT be used for indentation. A CATS document uses **two spaces per nesting level** as the indentation unit, and any indentation that is not a multiple of two spaces makes the document ill-formed.

Block boundaries are determined by indentation level alone. A line indented more deeply than its predecessor opens a child block; a line indented less deeply closes the inner block and returns to the enclosing block. No explicit block terminator is used.

Blank lines MAY appear between top-level constructs and between field lines; they have no effect on block scope. Trailing whitespace at the end of a line is permitted but SHOULD be avoided.

## 2.3 Identifiers

An **identifier** is the lexical category used for type words, enum values, and the names of reusable definitions in the `$defs` block. An identifier consists of an initial character drawn from the ASCII letters or underscore (`A`–`Z`, `a`–`z`, `_`), followed by zero or more ASCII letters, digits, or underscores (`A`–`Z`, `a`–`z`, `0`–`9`, `_`). Non-ASCII characters MUST NOT appear in identifiers; tool authors needing non-ASCII labels express them in description text instead.

```ebnf
identifier  ::= id-start id-continue*
id-start    ::= [A-Za-z_]
id-continue ::= [A-Za-z0-9_]
```

Identifiers are case-sensitive: `userName`, `username`, and `USERNAME` are three distinct identifiers. No length limit is imposed by this specification.

A **name** is the lexical category used for tool names and field names. It extends the identifier grammar with the ASCII hyphen (`-`): a name begins with an identifier start character and continues with any mix of ASCII letters, digits, underscores, and hyphens. The hyphen is admitted bare because hyphenated names — `get-weather`, `model-name` — are common in real tool definitions, the hyphen collides with no structural delimiter in name position, and quoting every such name would add noise to the most frequent case.

```ebnf
name        ::= id-start (id-continue | "-")*
```

A name may not begin with a hyphen or a digit; `-weather` and `2fa` are not bare names and, where such a label is unavoidable, must be quoted (§2.6) in the positions that permit quoting. Every identifier is a name, but not every name is an identifier: a name containing a hyphen is not a valid identifier, which is why definition names — referenced through the identifier-only `reference` production (§5.7) — remain restricted to identifiers (§3.2).

Reference names in the `$defs` block are identifiers as defined above; when referenced from a field's type slot, the identifier is preceded by a literal `$` character (e.g., `$Address`). The `$` is not part of the identifier itself but a syntactic marker introduced in §5.7.

## 2.4 Reserved words

CATS defines two categories of reserved words. Both categories occupy positions in the grammar where their interpretation is fixed and cannot be redefined by tool authors.

The first category is the **type vocabulary** — the eight type words `string`, `integer`, `number`, `boolean`, `array`, `object`, `null`, and `any`. This vocabulary is closed: no additional type words MAY be introduced by extensions or SDK generators.

The second category contains the single keyword `$defs`, which introduces the document-scope block of reusable definitions specified in §3.2.

A field name that is a reserved word, falls outside the name grammar of §2.3 (for instance by beginning with a digit or hyphen, or containing a character other than a letter, digit, underscore, or hyphen), or is otherwise not a valid bare name MUST be written as a quoted string in the field-name position — for example, `"user.id"`, `"object"`, or `"名前"`. A hyphenated name such as `model-name` is a valid bare name and is not quoted. The quotes are not part of the key and are stripped during conversion. Reserved words MAY appear as enum values or string literals when quoted in accordance with §2.6.

Although CATS grammar permits reserved words as quoted field names, tool authors SHOULD NOT use reserved words as field names even when quoting makes them syntactically valid. A field named `"string"` or `"object"` is legal but confusing to both human readers and models; rename the field in the source schema where possible.

## 2.5 Literals

CATS defines four literal forms for primitive values.

**String literals** are sequences of Unicode characters enclosed in double quotes. The escape sequences `\"` (literal double quote) and `\\` (literal backslash) are recognized within a string literal; all other characters represent themselves. Strings appear in the grammar wherever an unambiguous textual value is required — see §2.6 for the rule governing when quoting is required.

```ebnf
string-literal ::= '"' (string-char | string-escape)* '"'
string-char    ::= any Unicode code point except U+0022 ("), U+005C (\), U+000A (LF), and U+000D (CR)
string-escape  ::= '\"' | '\\'
```

A string literal is confined to a single line: line terminators (LF, CR) are excluded from `string-char`, so a quoted value cannot span lines. This keeps the line-oriented document model (§2.2, §3.4) intact — no quoted string can hide a line break that the indentation scanner would otherwise have to reason about.

**Number literals** follow the same grammar as JSON numbers (RFC 8259, §6). A number consists of an optional minus sign, an integer part, an optional fractional part, and an optional exponent introduced by `e` or `E` and an optional sign. The integer part is either a single `0` or a non-zero digit followed by zero or more digits. Examples of well-formed numbers include `0`, `42`, `-17`, `3.14`, `1e10`, and `-2.5E-3`. CATS does not distinguish between integer and number literals at the lexical level; the distinction is established by the type slot of the enclosing field.

**Boolean literals** are the two tokens `true` and `false`, written in lowercase.

The **null literal** is the token `null`, written in lowercase. The null literal is distinct from the `null` type word of §2.4: the literal appears as a value, the type word appears in the type slot. Context determines the interpretation.

## 2.6 Quoting rules

CATS follows a single quoting principle: **free-text values appear unquoted by default**, and quotes are introduced only when the bare form would be syntactically ambiguous. The principle has three triggers, all of which produce the same result — wrap the entire value in double quotes.

A value MUST be quoted when:

1. **JSON literal collision** — the bare token would parse as a number, `true`, `false`, or `null` but is intended as a string. Example: a string-valued enum member written `"true"` rather than `true`.
2. **Type vocabulary collision** — the bare token would parse as one of the eight type words of §2.4 but is intended as a value. Example: a single-value enum whose value happens to be `string`, written `"string"`.
3. **Delimiter collision** — the value contains a character that would otherwise be interpreted as a structural delimiter in its enclosing position. The set of relevant delimiters depends on context and is specified at the point each delimiter is introduced.

When any of these triggers fires, the entire value is quoted as a single unit; partial quoting of only the colliding portion is not permitted. Inside a quoted value, the escape sequences `\"` and `\\` produce a literal double quote and a literal backslash respectively.

The quote-when-ambiguous principle governs string values, enum members, and description text. Two annotation values — regular-expression patterns and MIME types — are quoted **unconditionally** rather than under this rule; the unconditional cases are defined in §6.3 where those annotations are introduced.

# 3. Document structure

## 3.1 Top-level form

A CATS document consists of an optional `$defs` block followed by one or more tool entries. The `$defs` block, when present, MUST appear before any tool entries. Both top-level constructs begin at column 0.

A document MUST contain at least one tool entry. A document consisting of only a `$defs` block is ill-formed.

The top-level grammar is:

```ebnf
document    ::= defs-block? tool-entry+
tool-entry  ::= tool-block | raw-tool
defs-block  ::= "$defs" NEWLINE INDENT definition+ DEDENT
tool-block  ::= (name | string-literal) description? NEWLINE (INDENT field-line+ DEDENT)?
definition  ::= identifier description? NEWLINE INDENT field-line+ DEDENT
```

The grammar uses `INDENT` and `DEDENT` as virtual terminals produced when indentation increases or returns to a previous level; the scoping rule is specified in §3.4. The body of a `tool-block` is optional: a header-only tool with no indented block is well-formed and encodes a tool that takes no parameters (§3.3). A `definition` body is not optional — a definition with no fields encodes nothing reusable. The `raw-tool` alternative carries a fallen-back tool as a verbatim JSON Schema object at document scope; its form is specified in §7.5 and its production in Appendix A. The `tool-block` and `definition` productions are otherwise structurally identical, differing in position (definitions appear inside the `$defs` block, tool blocks at document scope) and in their header name.

## 3.2 The `$defs` block

The `$defs` block declares reusable schema shapes that may be referenced from any tool block in the same document. The block consists of the reserved word `$defs` at column 0, with one or more named definitions on the lines below.

Each definition has the same structural form as a tool block: an identifier as the header, an optional inline `#` description, and an indented body containing field lines. The body uses the same grammar that applies inside any other block in CATS — required markers, type slots, annotation chains, defaults, and descriptions all apply identically.

Definitions declared in `$defs` are visible to every tool block in the same document. There is no tool-scoped variant; the ordering between the `$defs` block and the tools that reference its definitions does not affect resolution. References to definitions use a `$`-prefixed identifier in the type slot, the full grammar for which is specified in §5.7.

The `$defs` block is optional and SHOULD be omitted when no shared definitions exist; an empty `$defs` block has no valid grammar form.

The `$defs` header line carries no description. JSON Schema attaches no documentation to the `$defs` keyword itself, so there is nothing for a header description to round-trip to; `$defs # shared shapes` is therefore ill-formed. (Individual definitions inside the block carry descriptions normally, on their own header lines, per §4.6.)

Definition names SHOULD be written in PascalCase by convention, matching the naming style used by JSON Schema's `definitions` and modern generic-programming languages. The convention is a recommendation rather than a grammar requirement; identifiers in any case permitted by §2.3 are accepted.

Unlike tool names (§3.3) and field names (§4.2) — both of which are names that may be hyphenated and, beyond that, quoted when they contain other special characters — a definition name is restricted to a bare identifier (§2.3): no hyphens, no quoted escape hatch. This restriction is deliberate. A definition is reached through a reference of the form `$Name` (§5.7), and admitting non-identifier definition names would force a quoted reference syntax such as `$"some-name"` — a surface form frontier models have rarely seen and read less reliably, which directly undercuts the model-readability goal (§1.2). The restriction costs little in practice: a definition name is author-chosen, never contract-bound the way a tool name is, so an author can always pick an identifier-legal name, and the PascalCase convention above already steers names well clear of the characters that would need an escape hatch.

## 3.3 Tool blocks

A tool block encodes one complete tool definition as a single block. The three components of a tool definition — the tool's name, its natural-language description, and its input-parameter schema — are folded into the block's shape rather than expressed as separate fields.

The header line consists of the tool name at column 0, followed optionally by an inline `#` description. The tool name is a **name** (§2.3): a bare identifier extended to allow hyphens, so `get-weather` needs no quoting. A tool name that falls outside the name grammar — one containing a dot, slash, or other special character, or colliding with a reserved word — is written as a quoted string instead, following the same quote-when-ambiguous rule as field names (§2.4, §2.6). The escape hatch matters for tool names specifically because the name is contract-bound: it is the identifier the model emits when calling the tool, so it cannot be silently renamed to fit the identifier grammar the way an internal label could. The header carries no trailing punctuation, and there is no separate `name:` or `description:` field. The body of the block is the indented set of field lines representing the input-parameter schema; there is no nested `input_schema:` wrapper.

The body uses the same field-line grammar that applies anywhere else in the document. Each line describes one parameter of the tool — its required marker, type slot, annotations, default, and description follow the rules of §4. The body MAY contain nested object blocks for compound parameters.

A tool MAY have no parameters at all. A header-only tool block — a header line with no indented body, such as `ping # Check connectivity` — is well-formed and encodes a tool whose input schema is the empty closed object (`{"type": "object", "properties": {}, "additionalProperties": false}`), i.e. a tool called with no arguments. Parameterless tools are common in real tool definitions (connectivity checks, current-time lookups, session resets), so the notation admits them directly rather than requiring a placeholder field. The companion *CATS Usage Protocol* additionally recommends that, when any header-only tool is present, the model be told such tools are called with an empty arguments object, since the empty body leaves no visible cue on its own.

Tool-level fields beyond the name, description, and parameter schema — including JSON Schema's tool-level `title`, OpenAPI's `summary` and `operationId`, and provider-specific runtime flags such as OpenAI's `strict` or Anthropic's `cache_control` — are not encoded in CATS. The handling of these constructs is specified in §8.3.

## 3.4 Block boundaries and indentation scoping

This section specifies the indentation depths at which the document constructs of §3.1–§3.3 appear and the rule by which their bodies are delimited.

Top-level constructs — the `$defs` block and tool blocks — appear at column 0. The body of a tool block appears at column 2. Within a `$defs` block, each definition header appears at column 2, and that definition's body appears at column 4. Nested object blocks inside any body appear one indentation level deeper than their parent field line, recursively.

A block is the maximal run of lines whose indentation is strictly greater than the block's header. The block ends at the first subsequent line whose indentation returns to the header's level or shallower. No explicit terminator marks the end of a block; the indentation mechanics of §2.2 alone determine block membership.

## 3.5 Name uniqueness

Three name-uniqueness conditions hold over a CATS document. None is expressible in the context-free grammar — each is a property of the assembled document rather than of any single production — so all three are prose well-formedness rules (§A.6), enforced after parsing.

**Definition names MUST be unique.** Two definitions in the `$defs` block sharing a name are ill-formed: a `$Name` reference (§5.7) must resolve to exactly one definition, and duplicate declarations make resolution ambiguous.

**Field names MUST be unique within a block** (§4.2): no two field lines in the same block may share a name, since each becomes a key in the enclosing object's `properties`.

**Tool names SHOULD be unique within a document.** The output envelope (§7.2.1) serializes a multi-tool document to an array of tool schemas keyed by `name`, and tool-calling APIs dispatch on that name; two tools sharing a name leave the host unable to tell which the model means to call. Uniqueness is a SHOULD rather than a MUST because a document with a repeated tool name is still structurally parseable and serializable — the ambiguity is a downstream dispatch hazard, not a structural defect — but authors and converters SHOULD treat a collision as an error to be corrected at the source.

# 4. Field lines

## 4.1 Canonical field-line grammar

Field lines form the body of every block in a CATS document — tool blocks, `$defs` definitions, and nested object blocks all consist of field lines. All field lines follow the same grammar:

```ebnf
field-line       ::= field-name required-marker? SP type-expression default? description? NEWLINE nested-block?
field-name       ::= name | string-literal
required-marker  ::= "*"
default          ::= SP "=" (value-literal | json-literal)
description      ::= SP "#" SP description-text
nested-block     ::= INDENT field-line+ DEDENT
```

The components appear in canonical positional order — field name, required marker, type expression, default, description — and this is the only permitted order. Each component from the default onward is optional; the field name and type expression are mandatory on every field line. Annotations are not a separate field-line component: they attach inside the type expression at the `single-type` level (§5.1), so the annotation chain is governed by §5 and §6 rather than occupying a slot of its own here.

A field MUST NOT carry both a required marker and a default value. The combination is semantically contradictory — a required field has no value to fall back to — and is ill-formed regardless of whether the grammar would otherwise permit it.

The `type-expression` production is specified in §5; the `annotation-chain` production is specified in §6.5. The following subsections cover each field-line component in turn.

## 4.2 Field names and the required marker

A field name is a **name** (§2.3) — a bare identifier extended to allow hyphens — or a quoted string when the name falls outside the name grammar: when it begins with a digit or hyphen, contains a character other than a letter, digit, underscore, or hyphen (such as a dot or slash), collides with a reserved word, or requires non-ASCII characters — for example, `"2fa"`, `"user.id"`, `"object"`, or `"名前"`. A hyphenated name such as `model-name` needs no quoting. The full rule governing when quoting is required is specified in §2.4.

A required field is marked by appending `*` directly after the field name with no intervening whitespace — `latitude`* for a bare name, `model-name*` for a hyphenated name, `"user.id"*` for a quoted name. A field without a `*` marker is optional; optionality is the default state, inherited from JSON Schema's convention that a property is optional unless it appears in the enclosing schema's `required` array.

Field names MUST be unique within a single block. Two field lines with the same name in the same tool block, definition, or nested object block are ill-formed, since each maps to a key in the enclosing JSON object's `properties` and JSON objects cannot carry a duplicate key. This is a prose well-formedness rule (§A.6); the grammar admits repeated field lines, and the uniqueness condition is enforced over the assembled block.

## 4.3 The type slot

The type slot occupies the position immediately following the field name (and required marker, if present) and exactly one space. Its content is a **type expression** — a whitespace-free unit that specifies the kind of value the field accepts. The full grammar for type expressions, covering all forms from bare type words to parameterized arrays, pipe-unions, and `$`-prefixed references, is specified in §5. The type slot ends at the first unenclosed space, `=`, `#`, or end-of-line, where "unenclosed" means outside of angle brackets or quotation marks.

## 4.4 The annotation chain

Annotations are constraint markers that attach directly to a type expression with no intervening whitespace, forming a chain immediately to its right. The annotation chain appears after the type expression and before the default value. The full vocabulary and syntax of each annotation is specified in §6.

Because annotations attach at the `single-type` level (§5.1), there is no field-level annotation slot to fill — every annotation belongs to the type expression itself, whether the field carries a single type or a pipe-union. When the type expression is a pipe-union (§5.5), annotations bind to the branch they appear on rather than to the union as a whole, and each branch carries its own annotation chain independently within the type expression.

When multiple annotations attach to the same type expression or branch, they MUST appear in the canonical order specified in §6.5. A field line with annotations outside this order is ill-formed.

## 4.5 Defaults

A default value appears after the type expression and annotation chain. It is introduced by a single space followed by `=`, with the value glued directly to `=` and no following space — `count integer =5`, `config object ={}`. The leading space is always present, regardless of the preceding token. Its purpose is to prevent the sequence `>=` from arising when the type expression ends in `>` — which happens when the type is a parameterized array, as in `array<string> =[]` — where a directly glued `=` could be misread as a comparison operator (greater than or equal to). Making the space unconditional rather than conditional on a trailing `>` keeps the rule uniform and lets the `default` production stay context-free.

Primitive defaults follow the quote-when-ambiguous rule of §2.6: string defaults MUST be quoted; numeric, boolean, and null defaults are written unquoted. When the type expression is a pipe-union of enum values, the default is written as a bare literal matching one of the union members.

Array and object defaults are written as inline JSON literal syntax — for example, `={}` for an empty object or `=[]` for an empty array. The literal is treated as an opaque JSON value: JSON's own syntax governs its content, not CATS's quoting or identifier rules. Whitespace MUST NOT appear inside the literal; the entire default must form a single contiguous token.

A field MUST NOT carry both a required marker (§4.2) and a default value, as stated in §4.1.

## 4.6 Descriptions

A description appears at the end of a field line, after any default value, introduced by the three-character sequence `#` (space, hash, space). The description text is free-form prose extending to end-of-line; the CATS grammar does not interpret its content. On round-trip, description text is carried into JSON Schema's `description` keyword.

When the description text itself contains a `#` character, the entire description text MUST be wrapped in double quotes in accordance with §2.6's third quoting trigger. The enclosing quotation marks are stripped during output and do not appear in the resulting `description` value.

The `#` description syntax of this section applies uniformly across all block headers in CATS. Tool block headers, `$defs` definition headers, and nested object parent field lines all use the same delimiter and the same quoting rule introduced here; §3 introduced those headers positionally but deferred the description rule to this section.

Descriptions are optional on every field line and every block header.

## 4.7 Nested blocks

A field whose type expression is `object` or `array<object>` MAY carry a nested block — an indented set of field lines that describe the object's shape. The nested block begins on the line immediately following the parent field line, with each of its lines indented exactly one level (two spaces) deeper than the parent. The block ends at the first subsequent line whose indentation returns to the parent's level or shallower, following the scoping rule of §3.4.

The contents of a nested block are field lines following the full grammar of §4.1. All constructs — required markers, annotation chains, defaults, descriptions — apply to nested field lines without modification. Nesting is recursive: any field inside a nested block whose type is `object` or `array<object>` may itself carry a further-nested block, to any depth permitted by §2.2's indentation rules.

A field typed `object` or `array<object>` with no nested block is grammatically valid. All objects in CATS are implicitly closed (§5.4); a bare `object` field with no nested block therefore produces an object that validates only the empty value `{}`. The construct is permitted but unusual.

# 5. Types

## 5.1 The type vocabulary

The content of a type slot (§4.3) is a **type expression**: a whitespace-free unit that identifies the kind of value a field accepts. This section specifies the `type-expression` production deferred from §4.1. The grammar is:

```ebnf
type-expression   ::= type-union | enum-union | single-value-enum | single-type
single-type       ::= base-type annotation-chain?
base-type         ::= primitive-type | array-type | object-type | reference
primitive-type    ::= "string" | "integer" | "number" | "boolean" | "null" | "any"
array-type        ::= "array" ("<" type-expression ">")?
object-type       ::= "object"
reference         ::= "$" identifier
type-union        ::= single-type ("|" single-type)+
enum-union        ::= value-literal ("|" value-literal)+
single-value-enum ::= value-literal
```

A `single-type` MAY carry an `annotation-chain` — the constraint markers (`[0,100]`, `:date-time`, `:unique`, and the rest) attached directly to its right with no intervening whitespace. The chain attaches at the `single-type` level so that each branch of a type union carries its own chain independently (§4.4, §5.5); a bare-type field is the one-branch case of the same rule. An `enum-union` takes no annotation chain (§5.6). The `annotation-chain` production and the full vocabulary it admits are specified in §6.

Every type expression is built from the eight type words of §2.4 — `string`, `integer`, `number`, `boolean`, `array`, `object`, `null`, and `any`. This vocabulary is closed: no additional type words MAY be introduced. The eight words divide into two groups. The six **primitive types** (§5.2) occupy the type slot as bare words. The two **composite constructors** `array` and `object` (§5.3, §5.4) carry internal structure — an element type or a nested block. Layered on top of these are three composition forms: type unions (§5.5), enum unions (§5.6), and references to `$defs` (§5.7).

## 5.2 Primitive types

Six type words stand alone in the type slot without parameterization. `string`, `integer`, `number`, and `boolean` map directly to the JSON Schema types of the same names; `integer` and `number` preserve JSON Schema's distinction between integer-valued and arbitrary numeric fields.

`null` denotes the JSON null value. It is included in the vocabulary primarily to serve as a branch of nullable type unions (`string|null`, §5.5); a standalone `null`-typed field is valid but uncommon. The `null` type word is distinct from the `null` literal of §2.5 — the type word occupies the type slot, the literal appears as a value. A `null` type carries no annotation chain (§6): every annotation constrains the shape of a present value (a length, a range, a format), and `null` is the absence of a value, so a chain on it — `null:date-time`, `null[0,10]` — is contradictory and ill-formed.

`any` denotes a field that accepts any JSON value, encoding a source schema with no type declaration and no constraints — the empty schema `{}` or a description-only schema. A typeless schema that carries constraints but no explicit type (for example `{"minimum": 0}`) is **not** encoded as `any`; the converter does not infer a type from constraint vocabulary, and such schemas fall back to raw JSON Schema (§7.5).

In this sense `any` is a CATS-native canonicalization of the empty schema: the multiple JSON Schema spellings that accept any value — `{}` and description-only schemas — all converge on the single type word `any` on input, and `any` emits as `{}` (or a description-only schema) on output. Because `any` emits no `type` keyword, it carries no annotation chain (§6): a chain on `any` would emit a constraint with no type to anchor it — `any:date-time` would produce the constraint-only typeless schema `{"format": "date-time"}` that the forward direction explicitly refuses to encode (above). An annotation chain on `any` is therefore ill-formed; a source schema needing a constrained value declares its type.

## 5.3 Arrays

An array's element type is encoded as angle-bracket parameterization on the `array` type word. A typed array takes the form `array<element_type>` — for example, `attendees array<string>`. The element slot accepts any type expression, including a nested array (`matrix array<array<integer>>`) or a reference (`attendees array<$Person>`).

A bare `array` with no parameterization is valid and encodes the untyped-array case — JSON Schema's `{"type": "array"}` with no `items` schema. The parameterized form extends the type word rather than replacing it.

The form `array<object>` introduces an array of shaped objects; the indented block describing that object's fields is governed by §4.7 and is not part of the type expression itself. Array length bounds and the `:unique` annotation attach after the type expression and are specified in §6.4, not here.

## 5.4 Objects

The `object` type word denotes a JSON object. All objects in CATS are implicitly closed: the CATS-to-JSON-Schema converter emits `"additionalProperties": false` on every object. There is no syntax to express an open object or a typed-open variant; a source schema that relies on one is not silently closed but takes the fallback path (§7.4, §7.5).

A shaped object carries a nested block of field lines describing its properties, following the rules of §4.7. A bare `object` field with no nested block is valid but unusual — being implicitly closed, it validates only the empty value `{}` (§4.7).

An object whose keys are user-defined but whose values share a shape — the `Dict[str, X]` pattern, such as a headers or metadata map — has no direct CATS encoding. The author restructures such a field as an `array<object>` whose element has `key` and `value` sub-fields, rather than reaching for open-object syntax that CATS does not provide.

## 5.5 Type unions

A field that accepts a union of types is encoded as a pipe-separated list of type expressions in the type slot — for example, `query string|array<string>`. This form encodes JSON Schema's `anyOf`. The pipe character is bare at every nesting level: parentheses are never used to group union branches, including inside `array<...>` parameterization.

Each branch is a complete type expression and carries its own parameterization and annotation chain (§6) independently. Annotations therefore bind to the branch they appear on, not to the union as a whole, as established in §4.4 — `count integer[0,100]|null` bounds the integer branch alone.

A `null` branch (`string|null`) is the canonical encoding of a nullable type, covering JSON Schema's `{"type": ["X", "null"]}` and OpenAPI's `nullable: true` (§7.6).

A JSON Schema `type` array is not limited to the nullable case. A general multi-type array such as `{"type": ["string", "integer"]}` encodes as the pipe-union `string|integer` — each member becomes a branch — and the nullable form is just the special case whose last branch is `null`. This encoding is available only when the `type` array is the schema's sole constraining keyword. A `type` array that carries a *sibling constraint* — for example `{"type": ["string", "null"], "format": "date-time"}` — has no behavior-preserving CATS form, because CATS attaches an annotation to a single branch (§4.4), and a sibling sitting beside a `type` array does not say which branch it constrains: applying `format` to every branch (including `null`) or to none of them both change the accepted value set relative to the source's "applies wherever it is meaningful" semantics. The converter does not guess; a `type` array combined with any sibling constraint takes the fallback path (§7.5), where it round-trips verbatim.

JSON Schema's `oneOf` is **not** encoded as a CATS pipe-union. `oneOf` requires exactly one branch to match, whereas the pipe-union emits `anyOf`, which requires at least one; encoding `oneOf` as a pipe-union would widen the accepted value set, so `oneOf` takes the fallback path (§7.4, §7.5) and is preserved verbatim. Discriminated unions — multiple object shapes distinguished by a literal-typed discriminator field — depend on the same exclusivity and likewise fall back (§7.5).

## 5.6 Enum unions

A field constrained to a fixed set of values is encoded as a pipe-separated union of value literals in the type slot — for example, `visibility public|private|default`. No `enum` keyword appears and no separate type is declared; the underlying type is recoverable from the value forms. Non-string enums use the same syntax without modification: `priority 1|2|3|5` is an integer enum, and `offset -0.5|0|0.5` a number enum. A field whose values are exactly `true` and `false` is encoded as the `boolean` type, not as the enum `true|false`.

An enum union and a type union (§5.5) share the same pipe-separated surface syntax; the two are distinguished solely by whether the branches are type words or value literals, resolved through the quoting rule of §2.6. `string|integer` is a type union over two type words; `draft|published` is an enum over two string values; and a value that collides with a type word is quoted to mark it as a value — `"string"` is an enum member, `string` is a type. The same rule disambiguates a single-value enum from a bare typed field: `mode string` is a string-typed field, whereas `mode "automatic"` is a single-value enum, with the quotes restoring the cue that the token is a value. JSON Schema's `const` keyword is encoded as this quoted single-value form.

A single pipe-union MUST be wholly one kind or the other: its branches are either all type expressions or all value literals. A union mixing the two — `string|published`, intending "any string, or the specific value `published`" — is not valid CATS, because `string` already subsumes any literal string value and the mixed form carries no meaning the pure type union does not. A field needing a type alongside specific out-of-band values falls back to raw JSON Schema (§7.5).

The members of an enum union MUST further share a single JSON type. The reverse direction emits one inferred `type` keyword alongside `enum` (§7.2), so a mixed-type source enum — `enum: [1, "high"]` or `enum: ["a", null]`, where the members span more than one JSON type — has no single type to infer and no CATS encoding. Such an enum takes the fallback path (§7.5), where it round-trips verbatim, rather than being encoded under a type the members do not jointly share. (The two-member `true`/`false` case is not an exception: it is encoded as the `boolean` type, not as an enum, per the rule above.)

Annotations attach to individual union branches rather than to the union expression as a whole. How an enum paired with a sibling constraint converts depends on whether the converter can evaluate that constraint exactly against literal values, which partitions the constraints into two groups.

The first group is the **member-checkable** constraints — `minLength`, `maxLength`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, and `multipleOf` — each an exact comparison or arithmetic test on a literal value, with no dependence on an external matching dialect. For these, the converter applies a conditional drop. Because `enum` and a sibling constraint intersect — a value must satisfy both — the constraint is safe to discard only when every enum member already satisfies it, in which case the constraint rejects nothing and dropping it preserves the value set. The converter evaluates each member against the constraint: if all members pass, the constraint is dropped and the field encodes as a plain enum (behavior-preserving, documented as keyword-lossy in §7.4); if any member fails, dropping the constraint would widen the valid set, so the field instead falls back to raw JSON Schema (§7.5) rather than silently changing behavior.

The second group is every other constraint — any sibling constraint not in the member-checkable list above. None can be evaluated against an enum member with the exact, dialect-free test the conditional drop requires, so a field pairing an enum with any of them falls back to raw JSON Schema (§7.5). `format` and `pattern` are the cases that arise in practice. `format` is an opaque passthrough that the converter neither interprets nor validates (§6.1), so it has no way to test a member against it. `pattern` is ECMA-262 regular-expression syntax, and matching it correctly requires an ECMA-262-conformant engine whose semantics a converter cannot assume its host regex library provides; an incorrect match would drop a constraint that was not in fact redundant, reintroducing the silent value-set widening this rule exists to prevent. The remaining string constraints, `contentEncoding` and `contentMediaType` (§6.3), also fall here: they assert nothing the converter can check against a literal value, and pairing them with an enum of exact values is semantically incoherent in the first place. The rule is therefore closed — a member-checkable constraint is dropped when provably redundant and otherwise triggers fallback, and every other constraint triggers fallback unconditionally — so no enum-plus-constraint pairing is left without a defined disposition.

## 5.7 References to $defs

A field whose type is a reusable definition (§3.2) is encoded as the definition's name preceded by a literal `$` — for example, `home_address $Address`. The `$` sigil glues directly to the name with no intervening whitespace; it is a syntactic marker and not part of the identifier (§2.3).

References compose wherever a single type expression is valid: inside array parameterization (`attendees array<$Person>`), as a branch of a type union (`recipient $User|$Guest`), and as the type of a field that itself sits inside a nested block. A definition MAY reference another definition, and a definition MAY reference itself directly or through a cycle. Recursive and mutually recursive references are grammatically valid and round-trip without modification, but frontier models reason less reliably over recursive schemas regardless of encoding, so authors SHOULD prefer non-recursive structures where the use case permits.

Only references to named entries in the document's `$defs` block use this form. External `$ref` pointers to other documents, and internal pointers that target locations outside `$defs`, are not encoded in CATS notation; their handling is specified in §7.

Every `$Name` reference MUST resolve to a definition declared in the document's `$defs` block. A reference to a name that no definition declares — a *dangling reference* — is ill-formed. This is a prose well-formedness rule, not a grammatical one: the grammar admits `$AnyIdentifier` in a type slot, and whether the target exists is a whole-document property a context-free grammar cannot check (§A.6). Resolution is the reverse-side counterpart of the unused-definition condition noted in §3.2.

# 6. Annotations

## 6.1 Format annotations

An **annotation** is a constraint marker attached to the right of a `single-type` (§5.1) with no intervening whitespace. Annotations appear in a single canonical sequence (§6.5); different annotation families use different delimiter forms, but the whole sequence forms one whitespace-free unit glued to the base type.

A format annotation is encoded as a glued-colon postfix on the type — the base type, a colon, and the format value, as in `start_time string:date-time`. The base type stays visible and the format reads as a sub-annotation on it, preserving JSON Schema's two-layer `type`-plus-`format` model. The format value is passed through verbatim: spec formats (`date-time`, `email`, `uuid`), OpenAPI extensions (`int64`, `double`), and custom values (`phone`, `semver`) all encode identically, and the converter applies no lookup or policy.

A format annotation is not type-restricted: it attaches to any base type (`count integer:int64`, `price number:double`), not only strings. It MUST NOT be combined with an enum union; a field requiring both falls back to raw JSON Schema (§7.5).

Five values are reserved in format position: `length`, `regex`, `encoding`, `media`, and `unique` — the keywords that introduce the named annotations of §6.3–§6.4. Because a format value is otherwise a verbatim passthrough (above), a source `format` equal to one of these would produce a colon-prefixed token indistinguishable from the corresponding annotation: `string:length` could read as either the format `length` or the start of a `:length[…]` bound, and `string:length:length[1,20]` (format `length` plus an actual length bound) is unreadable. To keep the colon chain unambiguous, the annotation keyword always wins in this position, and a source schema whose `format` value collides with one of the five reserved keywords takes the fallback path (§7.5) for the containing tool rather than being encoded. The collision is rare — these five words are not registered JSON Schema formats — and fallback preserves the unusual schema verbatim, so no `format` value is lost.

## 6.2 Numeric constraints

Numeric bounds are encoded as glued mathematical interval notation directly after the type, with `[` and `]` marking inclusive endpoints (`minimum`, `maximum`) and `(` and `)` marking exclusive endpoints (`exclusiveMinimum`, `exclusiveMaximum`). The four bracket-parenthesis pairings cover all four inclusive/exclusive permutations:

```
limit integer[1,100] # 1 through 100, both ends inclusive
ratio number(0,1) # strictly between 0 and 1
score integer[0,100) # 0 inclusive, up to but not including 100
count integer[1,) # at least 1, no upper bound
```

When a bound is omitted on one side, the bracket on that side is open (`(` or `)`), following the convention that an unbounded endpoint is never included. An open bound may be written with either an inclusive or exclusive bracket on the absent side — `[1,]` and `[1,)` are equivalent, since a missing bound imposes no constraint; the converter emits the inclusive form. The interval glues to the type with no colon; unlike the named annotations of §6.3, bounds carry two value parameters and so use the bracket form directly.

The `multipleOf` constraint is encoded as a `%`-glued postfix following the bounds, with `%` glued to the closing bracket — or to the type itself when no bounds are present. The divisor MAY be any positive number, integer or fractional:

```
quantity integer[1,)%5 # minimum 1, in increments of 5
amount number%0.01 # to the nearest cent
```

The `%` follows the bounds because the range establishes the value space and divisibility filters within it.

## 6.3 String constraints

Four string-specific annotations occupy named positions in the colon chain. Each glues to the preceding token with no whitespace, and when several apply they appear in the canonical order of §6.5.

String **length** is encoded as `:length[lower,upper]`, with `minLength` mapping to the lower bound and `maxLength` to the upper. Both endpoints are always inclusive — JSON Schema defines no exclusive-length variant, so `(` and `)` never appear here — and either bound MAY be omitted using the open-bracket convention of §6.2. The `:length` keyword is required rather than bare brackets, because brackets on a `string` type carry ambiguous meaning (length, code-point range, substring indices) that the keyword resolves.

A **regex** pattern is encoded as `:regex["pattern"]`. The pattern is **always** quoted, unconditionally rather than under the quote-when-ambiguous rule of §2.6, because regex syntax overlaps with nearly every structural character in the notation (`#`, `|`, `*`, `(`, `)`, `[`, `]`), making case-by-case disambiguation unreliable. Backslashes are escaped as in any JSON string — a regex `\d+` is written `"\\d+"`.

**Content encoding** is encoded as `:encoding[value]` with the value unquoted, since all valid `contentEncoding` values are short lowercase identifiers (`base64`) with no structural characters. **Content media type** is encoded as `:media["value"]` with the value **always** quoted, for the same reason as regex: MIME types contain `/` and may contain `;`, `=`, and parameter strings. The two are independently optional; when both appear, `:encoding` precedes `:media`:

```
username* string:length[1,20]:regex["^[a-z0-9_]+$"] # lowercase handle
document string:encoding[base64]:media["application/pdf"] # encoded PDF
```

## 6.4 Array constraints

Array element-count bounds are encoded as glued interval notation directly after the parameterized array type — `array<element_type>[lower,upper]` — with `minItems` mapping to the lower bound and `maxItems` to the upper. Both endpoints are always inclusive, so `(` and `)` never appear in this context, and either bound MAY be omitted using the open-bracket convention of §6.2. Unlike string length (§6.3), array bounds need no keyword: brackets following `array<T>` read unambiguously as a count constraint on the container.

The `uniqueItems` constraint is encoded as a valueless `:unique` annotation in the colon chain; its presence signals `uniqueItems: true`, and its absence signals the JSON Schema default of `false`. When bounds are also present, they glue to the type first and `:unique` follows: `array<T>[bounds]:unique`.

```
tags array<string>[1,10] # 1 to 10 tags
emails array<string>[1,]:unique # at least one, all distinct
```

The split between bracket-glued bounds and the chain-positioned `:unique` is deliberate: bounds carry two parameters and take the parameter-bearing bracket form, whereas `:unique` is a parameterless flag and takes the named-annotation form alongside the string annotations of §6.3.

## 6.5 Canonical annotation order

When more than one annotation attaches to a single type, they MUST appear in one fixed order. An annotation chain outside this order is ill-formed. The full ordering is:

```
<type>[:format][bounds][%divisor][:length][:regex][:encoding][:media][:unique]
```

formalized as the `annotation-chain` production deferred from §5.1:

```ebnf
annotation-chain ::= format? bounds? mult? length? regex? encoding? media? unique?
```

The grammar is permissive; semantic validity is defined by the prose rules. No single field carries every slot, because the slots are gated by base type and several families are mutually exclusive. A **numeric** field (`integer`, `number`) may carry `:format`, numeric `bounds`, and `%divisor`; the `bounds` slot here means a value range. A **string** field may carry `:format`, `:length`, `:regex`, `:encoding`, and `:media`; it never carries numeric bounds or `%`. An **array** field may carry `bounds` (an element count, reusing the bracket slot) and `:unique`. The `bounds` slot is thus shared notation with type-dependent meaning — a value range on a number, an element count on an array — and the two readings never co-occur because a field has one base type. `:format` attaches to any *typed* base — string, numeric, array, object — but **not** to `null` or `any`: `null` is the absence of a value and `any` emits no `type` to anchor a constraint, so both reject every annotation, including `:format` (§5.2). An annotation chain on `null` or `any` is ill-formed.

# 7. Conversion semantics

This section specifies the behavior of the two converters: the JSON Schema → CATS direction (§7.1), which is behavior-preserving and makes documented keyword-level encoding choices, and the CATS → JSON Schema direction (§7.2), which is mechanical and emits a single canonical output form. §7.3 states the canonicalization guarantees the two directions jointly provide. §7.4 enumerates the round-trip cases where the recovered schema is not keyword-identical to the original. §7.5 specifies the raw JSON Schema fallback path that carries every construct CATS does not encode. §7.6 covers input drawn from OpenAPI rather than JSON Schema 2020-12. §7.7 specifies two opt-in input normalizations — both disabled by default — that re-interpret specific input conventions before conversion.

A note on terminology used throughout: a transformation is **automatic** when the converter performs it without author intervention, and an **author step** when the tool author must perform it manually before or during conversion. The two are distinguished explicitly because an implementer building the converter needs to know which transformations the tool owns and which it assumes a human has already applied.

## 7.1 JSON Schema → CATS

The forward direction sorts every JSON Schema construct into one of four treatments.

**Encoded directly.** Constructs with a CATS notation form are encoded per the rules of §4–§6: `type` and the type vocabulary, `properties`, the `required` array (each member becomes a `*` marker), `items`, `enum`, `const`, `format`, `default`, `description`, numeric constraints, string constraints, array constraints, `anyOf` (§5.5), and `$defs`/`$ref` (§5.7). `oneOf` is **not** in this set: its exclusive-match semantics cannot be encoded without widening the accepted value set, so it takes the fallback path (§7.4, §7.5). An `enum` carrying a member-checkable sibling constraint is encoded as a plain enum only when every member satisfies that constraint; an enum carrying a member-checkable constraint some member fails, or any non-member-checkable constraint, falls back to raw JSON Schema (§5.6, §7.4).

**Transformed automatically.** The converter rewrites these on the way in, each rewrite preserving the accepted value set: `nullable: true` becomes the pipe-union branch `null` (§7.6); and `definitions` is renamed to `$defs` with its `$ref` pointers rewritten (§7.6).

**Dropped automatically.** The converter discards these silently, as they carry no signal the model uses and removing them changes no validation behavior: `title`, `readOnly`, `writeOnly`, `$schema`, `$id`, `$comment`, and `$vocabulary`. `propertyNames` is **not** dropped — dropping it would remove key-name validation and widen the accepted value set — so it takes the fallback path (§7.5) instead.

**Author step.** Two constructs are designated *author steps*: the converter performs no automatic encoding of them, and a tool author who wants their content preserved must fold it into the field description as prose before conversion. An `examples` array is merged into the field description (for example, by appending "(e.g. …)"), and `deprecated: true` is surfaced in the description prose (for example, by appending "(deprecated)"). **If the author does not perform the step, the keyword is dropped — `examples` and `deprecated` are silently discarded during conversion** (they carry no validation semantics, so dropping them changes no behavior), and they appear in the dropped-keyword reference of §8.2 for that reason. The author step is the mechanism for *retaining* their content as prose, not a precondition for conversion: a schema carrying either keyword always converts, with or without the step. Separately, a typeless schema carrying constraints but no explicit `type` (such as `{"minimum": 0}`) is not encoded; the author corrects the source schema to declare its type, or the field falls back to raw JSON Schema (§7.5).

Constructs with no behavior-preserving CATS form and no author-side rewrite — `allOf`, `not`, `oneOf`, the `contains` family, `prefixItems`, conditional keywords, discriminated unions, open and typed-open objects, and the rest of §8 — take the fallback path of §7.5.

## 7.2 CATS → JSON Schema

The reverse direction is mechanical: every CATS construct maps to one canonical JSON Schema form, so a given CATS document always produces structurally identical JSON output. The canonical forms are:


| CATS construct                   | Emitted JSON Schema                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `*` required marker              | the field name is collected into the enclosing object's `"required"` array, in field-declaration order |
| `object` and its nested block    | `"properties": {...}` plus `"additionalProperties": false`                                             |
| `array<T>`                       | `"type": "array", "items": <T>`                                                                        |
| bare `array`                     | `"type": "array"` with no `items`                                                                      |
| type-union (`X | Y`)             | `{"anyOf": [<X>, <Y>]}`                                                                                |
| single-value enum / `const`      | `"const": <value>`                                                                                     |
| multi-value enum                 | `"type": <inferred>, "enum": [...]` — the inferred base type is always emitted alongside `enum`        |
| `$Name`                          | `"$ref": "#/$defs/Name"`                                                                               |
| `any`, no description            | `{}`                                                                                                   |
| `any`, with description          | `{"description": "..."}` — no `type` keyword                                                           |
| `:format`, constraints, defaults | the corresponding sibling keywords (§6)                                                                |


Three of these resolve choices that JSON Schema leaves open. Single-value enums emit `const` rather than a one-element enum, matching modern JSON Schema implementation practice. Multi-value enums always emit an explicit `type` alongside `enum`, because production APIs (notably OpenAI strict mode) require the declared type even though the values imply it. The `any` type word emits no `type` keyword in either form, mirroring the forward direction that produces `any` from `{}` and description-only schemas.

Within each emitted object, keywords appear in a fixed canonical order. At the **top level of a tool schema** (the envelope of §7.2.1) the order is: `name`, `description`, `$defs`, then the parameter-schema keywords. Within any schema object — the tool's parameter schema, a `$defs` entry, or a nested property — the keyword order is: `type`, `const`/`enum`, `format`, `default`, `properties`, `required`, `additionalProperties`, `items`, `anyOf`, `$ref`, the validation constraints, and finally `description`. Grouping the structural and validation keywords ahead of the prose `description` makes output deterministic and diff-stable. This canonical key order applies to **encoded tools only**; fallen-back tools are emitted byte-for-byte as they arrived and their top-level key order is not normalized (§7.5).

### 7.2.1 Output envelope

The atomic output unit is a single tool: one self-contained JSON Schema (draft 2020-12) object carrying its `name`, its `description` (when the tool block has one), its input-parameter schema (`type`, `properties`, `required`, `additionalProperties: false`), and a local `$defs` holding only the definitions that tool references — transitively, following each reference into the definitions it in turn reaches. The top-level keys appear in the fixed order `name`, `description`, `$defs`, then the parameter-schema keywords (§7.2): `name` first, `description` second when present, `$defs` next when the tool references any definition (omitted otherwise), and the schema body last. A reference still emits as `"$ref": "#/$defs/Name"` and now resolves against the tool's own embedded `$defs`.

A document containing multiple tool blocks serializes to a JSON array of these tool schemas, in document declaration order. Each tool schema must be independently valid because tool-calling APIs send tool schemas individually; a document-level shared `$defs` would dangle the moment a single tool is extracted, so for encoded tools the converter performs the transitive embedding described above and the definitions a tool needs travel inside it. A fallen-back tool (§7.5) is emitted verbatim and carries only whatever `$defs` its original input already contained, so its self-containedness is inherited from the source rather than constructed by the converter. This array-of-self-contained-tools form is the canonical envelope, which the reverse direction (§7.1) and the round-trip oracle mirror.

## 7.3 Canonicalization guarantees

CATS functions as a canonicalizing representation: it collapses inputs that validate the same set of values to a single form and emits a single deterministic form on output. The supporting properties are stated across §5, §6, and §7.2 individually; this subsection collects them, since the canonicalization they jointly provide is one of the format's most important properties beyond compression.

**Notation canonicality.** A given semantic content has exactly one well-formed CATS spelling. The type forms of §5 admit no alternate orderings or optional delimiters, and the annotation chain has a single canonical order (§6.5) — an out-of-order chain is ill-formed, not an accepted variant. The notation permits no alternate structural forms for the same construct.

**Inbound normalization (many-to-one).** JSON Schema inputs that differ only in keyword choice, while validating the same set of values, converge to one CATS form: `const` and a single-value `enum`, `definitions` and `$defs`, and OpenAPI's `nullable` and JSON Schema's `type: [X, null]` each map to a single construct. The distinctions are not preserved, but the accepted value set is (§7.4). `oneOf` does not normalize to `anyOf` — the two validate different value sets — so `oneOf` is out of scope and falls back (§7.4, §7.5) rather than converging.

**Outbound determinism (one-to-one).** A given CATS document produces exactly one JSON Schema document. Each construct has a single canonical emitted form, object keywords appear in the fixed order of §7.2, and `required` arrays follow field-declaration order. No construct carries an emit-time choice.

Together these yield **round-trip stability**: converting any JSON Schema to CATS and back reaches a fixed point that further round-trips leave unchanged. For the subset of JSON Schema that CATS encodes, the recovered schema validates the same set of values as the original; the first JSON Schema → CATS pass may be keyword-lossy (§7.4) but never behavior-changing. Constructs outside the encoded subset are carried verbatim through the fallback path (§7.5) and round-trip identically. CATS is therefore a stable canonical form for the subset of JSON Schema it encodes.

## 7.4 Lossy conversions

The JSON Schema → CATS conversion always preserves validation behavior: the recovered schema accepts and rejects the same JSON values as the original. What it does not always preserve is the original *keyword form*. This section documents every place that matters.

**Counts.** Across the whole conversion, the number of **behavior-changing** instances is **zero**: no conversion the converter performs alters the set of values a schema accepts. The number of **keyword-changing** instances — conversions whose output validates identically but does not recover the original keyword — is **five unconditional cases plus one conditional case**, enumerated below. (The five are counted by distinct *input* keyword form. Two of them, the OpenAPI `nullable` and the JSON Schema `type`-array spelling of nullability, converge on the same CATS output `X|null`; counted by distinct *output* form rather than input, the unconditional total is four. This document counts by input keyword, since that is what a tool author starts from.)

**The five unconditional keyword-changing conversions.** Each is performed automatically, has identical validation behavior to its source, and does not recover the original keyword on round-trip:

1. **Single-value `enum` → `const`.** A source `enum: [x]` with one member is encoded as the single-value form `mode "x"` (§5.6) and re-emitted as `"const": x` (§7.2), not as a one-member `enum`.
2. `**definitions` → `$defs`.** The pre-2020-12 `definitions` keyword and its `#/definitions/Name` pointers are renamed to `$defs` and `#/$defs/Name` (§7.6); the original keyword is not restored.
3. **OpenAPI `nullable: true` → `anyOf`.** `{"type": "X", "nullable": true}` becomes the union `X|null` (§5.5, §7.6), re-emitted as `{"anyOf": [{"type": "X"}, {"type": "null"}]}`.
4. `**type`-array nullability → `anyOf`.** The JSON Schema 2020-12 spelling `{"type": ["X", "null"]}` becomes the same `X|null` union and re-emits as the same `anyOf` form; the `type`-array keyword shape is not recovered.
5. **Two-member boolean `enum` → `boolean`.** A source `enum: [true, false]` is encoded as the `boolean` type (§5.6) and re-emitted as `"type": "boolean"`, not as the enum.

All five are behavior-preserving: in each case the recovered schema validates exactly the value set the source did.

**The one conditional keyword-changing conversion.** A multi-value `enum` paired with a member-checkable sibling constraint (such as `minLength`) drops that constraint **when, and only when, every enum member already satisfies it** — the constraint is then redundant, rejecting nothing, so the recovered schema is a plain enum and the original constraint keyword is gone. This is keyword-changing only in the case where the pairing occurs and the constraint is redundant; when the constraint is *not* redundant the field falls back instead (§5.6, §7.5), so a behavior-changing drop never occurs. It is listed separately from the five because it is contingent on the input, not performed on every occurrence of a keyword.

**Out-of-scope constructs fall back rather than convert lossily.** Some JSON Schema constructs cannot be encoded in CATS without altering which values the schema accepts or rejects. Rather than perform that conversion, CATS places these constructs out of scope: any tool containing one is carried verbatim as raw JSON Schema (§7.5), where it round-trips identically. The two most commonly encountered examples are listed here; the complete set is in §8.

- `**oneOf`.** `oneOf` requires exactly one branch to match; the CATS pipe-union emits as `anyOf`, which requires at least one. A value matching two branches passes `anyOf` but fails `oneOf`, so encoding `oneOf` as a pipe-union would widen the accepted set. `oneOf` therefore falls back. Discriminated unions depend on the same exclusivity and fall back for the same reason (§8).
- **Open objects.** A schema with `additionalProperties: true` (or omitted — the JSON Schema default) accepts values carrying extra properties beyond those declared. A CATS object is implicitly closed and rejects such values, so encoding an open object as CATS would narrow the accepted set. Open objects fall back rather than being silently closed; the opt-in `assume_closed` option (§7.7.1) lets a caller deliberately adopt the closed reading for the omitted case only. Typed-open objects (`additionalProperties: {schema}`) have no behavior-preserving CATS encoding; a tool author may restructure them as a key-value `array<object>` (§5.4), or they fall back (§7.5).

## 7.5 Raw JSON Schema fallback

A construct that CATS does not encode is carried through the fallback path: the affected tool is emitted as raw JSON Schema rather than as CATS notation. Fallback exists so that the converter never has to choose between two unacceptable outcomes — silently changing a tool's accepted value set, or refusing to convert a document at all. Every JSON Schema input therefore has a CATS output, even when part of that output is verbatim JSON Schema.

**Fallback is tool-level, not field-level.** When any field of a tool contains a construct with no behavior-preserving CATS encoding (the constructs enumerated in §7.4 and §8 — `allOf`, `not`, `oneOf`, conditional keywords, the `contains` family, `prefixItems`, open and typed-open objects, `propertyNames`, discriminated unions, and the rest), the **entire tool** falls back: its complete schema is carried as raw JSON Schema, unchanged. CATS does not interleave raw JSON Schema inside an otherwise-encoded CATS tool block. A partial encoding — compact CATS for the encodable fields with a raw-JSON fragment spliced in for one unencodable field — is deliberately excluded, because it would break the uniformity of the notation and force a model to context-switch between two syntaxes within a single tool definition. Fallback is therefore all-or-nothing per tool: a tool is either fully CATS or fully raw JSON Schema.

**Behavioral consequence.** A fallen-back tool is carried verbatim, so it round-trips identically: JSON Schema in, the same JSON Schema out. Fallback is thus the mechanism by which CATS preserves behavior for the constructs it cannot encode — they are neither converted lossily nor dropped, but passed through untouched. This is what makes the behavior-preservation guarantee of §1.2 unconditional across all input.

**Representation.** In the output envelope of §7.2.1, a fallen-back tool occupies the same position a CATS-encoded tool would — one entry in the array of tool schemas — and is carried verbatim exactly as it arrived, including any `$ref` pointers and any `$defs` block it already contained. The converter does not embed, resolve, rewrite, or reorder anything inside it (consistent with the byte-for-byte guarantee in §7.2 and the CATS-text single-line form below). Whether the tool is self-contained therefore depends on the original input already being self-contained; the converter does not make a non-self-contained fallback tool self-contained. A consumer of the output therefore treats every array entry uniformly as a JSON Schema; whether a given tool was encoded or fell back is not something the consumer must distinguish.

**CATS text representation.** In a CATS document, a fallen-back tool is emitted as its verbatim JSON Schema object in the document's tool sequence — occupying the same position a CATS tool block would, not interleaved inside one. A document is therefore a sequence of CATS tool blocks and raw JSON Schema tool objects, separated by the same blank-line conventions as ordinary tool blocks. **A raw tool object is emitted on a single line: the entire JSON Schema object is serialized with no internal line breaks, as one unbroken run of text at column 0.** This single-line form is what keeps the line-oriented indentation scanner (§2.2, §3.4) intact — the scanner never has to track brace nesting or distinguish structural braces from braces inside JSON strings, because the whole raw tool is one logical line that the scanner consumes without descending into it. The distinction is syntactic: a tool entry whose first non-space character is `{` is a raw JSON Schema tool; any other top-level tool entry is a CATS tool block (whose header is a name or quoted string, never `{`). The reader parses raw JSON tools into a `RawSchema` node carrying the JSON verbatim; the writer emits the same JSON unchanged on its single line. This text round-trip is identical to the JSON Schema envelope round-trip above.

**Reporting.** The converter records each fallback — which tool fell back and which construct triggered it — so that the proportion of a corpus requiring fallback can be measured. The fallback rate is a reported characteristic of a conversion, not a hidden one: it quantifies how much of real-world tool-definition usage lies within CATS's encoded subset, and is one of the format's primary empirical measures alongside token reduction.

**Author-facing strictness (non-normative).** The behavior above governs the JSON Schema → CATS converter, whose input is existing JSON Schema the author may not control. A separate author-facing tool that lints hand-written CATS may instead *reject* an un-encodable construct with a diagnostic, since its author can act on the feedback. That is a tooling choice outside the converter's normative behavior and does not change the conversion semantics specified here.

## 7.6 OpenAPI input handling

CATS accepts input from OpenAPI documents as well as JSON Schema 2020-12. Three OpenAPI-specific or legacy keywords are transformed automatically by the converter, since each carries content CATS already encodes by other means:

`**nullable`.** OpenAPI 3.0's `{"type": "X", "nullable": true}` becomes the pipe-union `X|null` (§5.5). OpenAPI 3.1 dropped the keyword in favor of JSON Schema's `type: ["X", "null"]`, which encodes to the same union.

**Singular `example`.** OpenAPI's singular `example` is the one-element analog of `examples`. Like `examples`, it carries no validation semantics; it is retained only if the author folds it into the description as prose (the author step of §7.1), and is otherwise dropped (§8.2). Normalizing the singular `example` into a one-element `examples` array is a planned convenience and is not yet performed by the converter — both forms are currently dropped unless the author surfaces them as prose.

`**definitions`.** The pre-2020-12 name for `$defs`, still emitted by older Pydantic v1 and OpenAPI 3.0 output, is normalized in a converter pre-pass that runs alongside provider-envelope unwrapping and before schema processing: wherever a schema object carries a `definitions` block (at tool scope or nested), that key is renamed to `$defs`, and every `$ref` string of the form `#/definitions/Name` is rewritten to `#/$defs/Name` throughout the schema tree. The rewrite is a pure keyword rename and pointer adjustment — the accepted value set is identical — after which the existing `$defs` encoding of §3.2 applies normally. If the same object carries both `definitions` and `$defs`, the converter does not merge or clobber either block; the containing tool takes the fallback path (§7.5) so no definition data is lost.

**Provider tool-definition envelopes.** Real tool definitions from LLM provider APIs wrap the parameter schema inside a provider-specific envelope; the converter unwraps three shapes to the inner schema before conversion. OpenAI Responses and Gemini `functionDeclarations` are flat — `{"name": …, "description": …, "parameters": {<schema>}}` — and the inner `parameters` object is the schema. OpenAI Chat Completions nests the same shape under a `function` key — `{"type": "function", "function": {"name": …, "parameters": {<schema>}}}` — unwrapped one level first. Anthropic uses `{"name": …, "description": …, "input_schema": {<schema>}}`, where `input_schema` holds the schema. In every case the tool's name and description are lifted from the envelope (the inner schema usually carries neither), and provider runtime flags that are not JSON Schema and not model-facing — `strict`, `cache_control`, and any `x-`* vendor extension — are dropped, as the provider-runtime flags of §8.3 that sit outside CATS's input entirely. After unwrapping, the inner schema converts normally: a wrapped tool whose inner schema is encodable encodes, and one whose inner schema is genuinely out of scope falls back on the inner schema's merits (§7.5), not on the wrapper. A dict carrying a top-level `properties` is the schema itself (or the CATS output envelope of §7.2.1), never an envelope, and is left untouched.

OpenAPI's `discriminator` keyword is a discriminated union and takes the fallback path (§7.5), per §8.

## 7.7 Opt-in input normalizations

The default JSON Schema → CATS conversion reads its input strictly as written: an omitted keyword carries its JSON Schema default meaning, and input that is not valid JSON Schema 2020-12 is never silently repaired. This section specifies two converter options that relax that strict reading in narrowly defined, widely encountered cases. Both are **disabled by default** and MUST be enabled explicitly by the caller.

The options do not weaken the fidelity guarantee of §1.2; they relocate it. With an option enabled, the converter first derives a *normalized input* by applying the option's re-interpretation, and conversion is then behavior-preserving with respect to that normalized input — the same unconditional guarantee, anchored one step later. What an option changes is which schema the converter reads, never how faithfully it converts what it read. To keep that change auditable, the converter's conversion report records every place an enabled option altered a disposition. Each option is a closed re-interpretation of one specific input convention; neither is a general repair mode, and no other deviation from the strict reading is performed under either.

When both options are enabled, `map_python_types` (§7.7.2) is applied before `assume_closed` (§7.7.1), since the closed reading applies to `object`-typed schemas and an input written in the Python-alias convention does not contain `object`-typed schemas until the aliases are mapped.

### 7.7.1 `assume_closed`

With `assume_closed` enabled, an object schema whose `additionalProperties` keyword is **omitted** is read as closed — as if it carried `"additionalProperties": false` — and encodes as an ordinary CATS object (§5.4) rather than taking the open-object fallback path (§7.4, §7.5).

The option's scope is exactly the omitted case. An explicit `"additionalProperties": true` is an author's stated intent to accept undeclared properties and is unaffected: the schema remains open and falls back as specified in §7.4. An explicit `"additionalProperties": false` is already closed and is likewise unaffected. A typed-open object (`additionalProperties: {schema}`) is unaffected and falls back per §5.4.

Under the strict default reading, an omitted `additionalProperties` means open — JSON Schema's default — so closing it narrows the accepted value set, which is why the option is off by default and the default converter falls back instead (§8.3). The option exists because the omitted-but-intended-closed pattern dominates real tool definitions: OpenAI's strict structured-output mode for function calling *requires* `"additionalProperties": false` on every object, so the closed reading is the contract the ecosystem's dominant validated tool-calling path already mandates, while authors writing outside strict mode pervasively omit the keyword without intending to accept undeclared arguments. `assume_closed` lets a deployer adopt that reading deliberately and visibly rather than by silent default.

Each object whose disposition the option changed — one that would have fallen back as open and instead encoded as closed — is recorded in the conversion report.

### 7.7.2 `map_python_types`

With `map_python_types` enabled, the converter applies a closed, four-entry rename of Python type names appearing as the value of a `type` keyword, before any other processing:


| Input `type` value | Normalized to                                                    |
| ------------------ | ---------------------------------------------------------------- |
| `"float"`          | `"number"`                                                       |
| `"dict"`           | `"object"`                                                       |
| `"tuple"`          | `"array"`                                                        |
| `"any"`            | the `type` keyword is removed; the remaining schema stands as-is |


These four names come from Python's type vocabulary and appear in tool definitions authored by Python-ecosystem tooling, which commonly emits them in place of the JSON Schema names. They are not valid JSON Schema 2020-12 — `type` admits only the seven defined names — so under the default strict reading, a tool containing one is invalid input and takes the fallback path (§7.5) verbatim, exactly as malformed as it arrived; the converter does not guess at repairs by default.

The first three renames (`float`→`number`, `dict`→`object`, `tuple`→`array`) are the conventional Python/Gorilla-to-OpenAPI equivalences. The treatment of `any` is deliberately *not* conventional: some ecosystem mapping tables narrow `any` to `string` for the convenience of an execution-time argument checker, but `string` accepts a strictly smaller value set than the source `any` intends. Removing the `type` keyword instead preserves the unconstrained value set — consistent with the fidelity principle (§1.2) that a normalization must never narrow what a schema accepts.

The mapping is a rename and nothing more. No other keyword, constraint, or structure is touched, and every downstream rule applies to the renamed schema exactly as it would to a natively written one: a renamed `"tuple"` whose `items` is a positional array still falls back via §8.1, a renamed `"dict"` with `additionalProperties` omitted is still open unless `assume_closed` is also enabled, and a removed `"any"` leaves either an empty or description-only schema (which encodes as the `any` type word, §5.2) or a constraint-only typeless schema (which falls back, §7.1). The four-entry list is closed: any other unknown `type` value remains invalid input and falls back.

Each schema the rename touched is recorded in the conversion report, with a count per alias.

# 8. Out of scope

CATS is a notation for the realistic subset of tool definitions, not a complete re-encoding of JSON Schema. This section is the reference list of JSON Schema and ecosystem constructs that CATS does not encode, organized by how the converter disposes of each: deferred to raw JSON Schema fallback (§8.1), dropped silently (§8.2), or handled by an existing CATS mechanism (§8.3). The dispositions in §7.1 are authoritative; this section expands them into a keyword-level reference with rationale.

## 8.1 Deferred keywords

A deferred keyword carries semantic content that CATS does not encode. The field — or, where the construct is tool-level, the whole tool — takes the raw JSON Schema fallback path (§7.5). Deferral is lossless on round-trip: the fallback preserves the original construct verbatim, so no validation behavior is lost. The deferred keywords are:


| Keyword / construct                                                                                                                                                         | Why deferred                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `allOf`                                                                                                                                                                     | Schema intersection has no pipe-union analog and is uncommon in tool calling                                                                  |
| `not`                                                                                                                                                                       | Negation has no CATS form, and models reason unreliably over it                                                                               |
| `contains`, `minContains`, `maxContains`                                                                                                                                    | At-least-one-matching validation falls outside the homogeneous `array<T>` model                                                               |
| `prefixItems`, post-`prefixItems` `items`, `unevaluatedItems`                                                                                                               | Positional tuple typing falls outside the homogeneous `array<T>` model                                                                        |
| `additionalItems`                                                                                                                                                           | Pre-2020-12 name for post-`prefixItems` `items`; deferred by structural identity                                                              |
| `if` / `then` / `else`, `dependentRequired`, `dependentSchemas`                                                                                                             | Conditional validation references other fields, and CATS has no cross-field syntax                                                            |
| `minProperties`, `maxProperties`                                                                                                                                            | Aggregate property-count bounds, unenforced by current providers, with no closed-object encoding                                              |
| `contentSchema`                                                                                                                                                             | Describes a shape inside an encoded string; restructure as a typed `object` instead                                                           |
| External `$ref` (cross-document)                                                                                                                                            | Points outside the document; inline the target or fall back                                                                                   |
| Internal `$ref` outside `$defs`                                                                                                                                             | Only `$defs`-targeted references are encoded (§5.7)                                                                                           |
| `$dynamicAnchor`, `$dynamicRef`                                                                                                                                             | Runtime-resolved generic references with no tool-calling use case                                                                             |
| `$anchor`                                                                                                                                                                   | An alternate reference target; not lifted into `$defs`, so a schema using it falls back                                                       |
| `discriminator` (OpenAPI)                                                                                                                                                   | Structurally a discriminated union, which is itself deferred                                                                                  |
| Constraint-only typeless schemas (e.g. `{"minimum": 0}`)                                                                                                                    | The converter does not infer a type from constraints; fix the source schema                                                                   |
| `enum` with a member-checkable constraint not met by every member, or with any non-member-checkable constraint (`format`, `pattern`, `contentEncoding`, `contentMediaType`) | Dropping the constraint would widen the value set, or the constraint cannot be member-checked (§5.6)                                          |
| Boolean schemas (`true`, `false`)                                                                                                                                           | Universal-accept / universal-reject have no document structure to encode                                                                      |
| `propertyNames`                                                                                                                                                             | Validates key names against a sub-schema; dropping it would widen the value set, so it falls back rather than being dropped                   |
| `patternProperties`, `unevaluatedProperties`                                                                                                                                | Property-set semantics with no behavior-preserving closed-object encoding; fall back unless the author restructures as `array<object>` (§5.4) |
| Mixed-type `enum` (members span more than one JSON type, e.g. `[1, "high"]`)                                                                                                | No single `type` to infer for emission alongside `enum`; members must share one JSON type (§5.6)                                              |
| `type` array combined with a sibling constraint (e.g. `{"type": ["string","null"], "format": "date-time"}`)                                                                 | The sibling cannot be attached to a single union branch without changing the value set (§5.5)                                                 |


## 8.2 Dropped keywords

A dropped keyword is discarded during JSON Schema → CATS conversion and never emitted on the way back. Dropping is not recoverable on round-trip, which is acceptable because each dropped keyword carries no validation semantics and no signal the model uses for tool calling. The dropped keywords are:


| Keyword / construct                                                         | Why dropped                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `title` (field-level)                                                       | The field key already serves the label role                                                                                                                                                                                                                                                                                                                                                                                             |
| `title` (tool-level)                                                        | The tool `name` already serves the identifier role                                                                                                                                                                                                                                                                                                                                                                                      |
| `readOnly`, `writeOnly`                                                     | The request/response distinction does not apply; every field is caller-written                                                                                                                                                                                                                                                                                                                                                          |
| `$schema`, `$id`                                                            | Document metadata, unobservable to the model                                                                                                                                                                                                                                                                                                                                                                                            |
| `$comment`                                                                  | Defined as schema-author notes, excluded from validation                                                                                                                                                                                                                                                                                                                                                                                |
| `$vocabulary`                                                               | Appears only in meta-schemas, not instance schemas                                                                                                                                                                                                                                                                                                                                                                                      |
| `summary`, `externalDocs` (OpenAPI, tool-level)                             | Documentation metadata duplicating the description                                                                                                                                                                                                                                                                                                                                                                                      |
| Duplicated parameter-object descriptions                                    | Redundant with the tool description                                                                                                                                                                                                                                                                                                                                                                                                     |
| `examples` (when not folded into prose)                                     | No validation semantics; retained only if the author folds it into the description as prose (author step, §7.1)                                                                                                                                                                                                                                                                                                                         |
| `deprecated` (when not folded into prose)                                   | No validation semantics; retained only if the author surfaces it in the description as prose (author step, §7.1)                                                                                                                                                                                                                                                                                                                        |
| `optional` (property-level sibling key, or parameters-level array of names) | Not a JSON Schema 2020-12 keyword; carries no validation semantics under the spec CATS targets (§1.3) — a conforming validator ignores it. The field's required-ness is determined solely by the standard `required` array, which is preserved normally. Some BFCL tool definitions carry `optional` as a redundant restatement of `required`'s absence; CATS drops it as inert, the same disposition as `title`/`readOnly`/`$comment`. |


## 8.3 Constructs handled by other means

A few constructs appear out of scope but are not, because an existing CATS mechanism already covers their realistic use. They take neither the fallback path nor a silent drop.

`additionalProperties` is not encoded as an explicit construct in CATS. All objects are implicitly closed (§5.4), so the semantics of `"additionalProperties": false` are built into the object model itself; a source object that is already closed encodes with its value set intact. The `true` and typed-open variants have no behavior-preserving CATS form — closing them would narrow the accepted value set — so a source schema that relies on open objects takes the fallback path (§7.5) rather than being silently closed (§7.4). For the omitted-keyword case only, the opt-in `assume_closed` option (§7.7.1) — disabled by default — permits reading the omission as closed; it does not alter the default behavior described here.

`patternProperties` and `unevaluatedProperties` have no CATS syntax and are not silently absorbed. A tool author may restructure the realistic use of `patternProperties` — objects with user-defined keys — as an `array<object>` with `key` and `value` sub-fields (§5.4); absent that manual restructuring, the field falls back to raw JSON Schema (§7.5). `unevaluatedProperties` controls property behavior under `allOf`/`oneOf` composition; since those compositions are themselves deferred (§8.1) and take the fallback path, a schema carrying `unevaluatedProperties` falls back with them rather than having the keyword silently dropped.

Provider-runtime flags — OpenAPI vendor extensions (`x-`*), and API infrastructure fields such as `strict` and `cache_control` — are not JSON Schema constructs and never form part of the model-facing payload. They are consumed by the serving infrastructure before the schema reaches the model, so CATS neither encodes nor drops them; they sit outside its input entirely.

# Appendix A: Complete EBNF grammar

This appendix is the authoritative formalization of CATS syntax. Where an inline grammar fragment in the body and a production here disagree, this appendix governs and the body fragment is to be read as illustrative. The grammar is written in the W3C EBNF dialect of §1.4 (the notation used by the XML and HTML specifications): `::=` introduces a production, `|` alternates, parentheses group, `?` `*` `+` mark optional and repeated terms, and quoted strings are terminals.

A token stream is a **syntactically** well-formed CATS document if and only if it derives from the `document` production. The grammar is deliberately permissive: it admits forms that the prose rules forbid on semantic grounds. Three classes of constraint are intentionally left to prose because a context-free grammar cannot express them faithfully, and encoding them would either be impossible or would multiply the productions without making the notation clearer:

- **Quoting.** When a string value must be quoted is governed by the quote-when-ambiguous rule of §2.6. The grammar admits both the bare and quoted forms of a `string-value`; which one is well-formed in a given position is a prose condition.
- **Annotation/type agreement.** Which annotations may attach to which base type is gated by §6.5. The grammar admits any annotation on any single type; the prose restricts it (a numeric bound on a `string`, for instance, is syntactically derivable but semantically ill-formed).
- **Field-line exclusivity.** A field MUST NOT carry both a required marker and a default (§4.1). The grammar permits the pairing; the prose forbids it.

Productions are ordered top-down — document structure, then field lines, then type expressions, then the annotation chain, then the lexical terminals — the order in which a recursive-descent implementation encounters them.

## A.1 Document structure (§3)

```ebnf
document     ::= defs-block? tool-entry+
tool-entry   ::= tool-block | raw-tool
defs-block   ::= "$defs" NEWLINE INDENT definition+ DEDENT
tool-block   ::= (name | string-literal) description? NEWLINE (INDENT field-line+ DEDENT)?
definition   ::= identifier description? NEWLINE INDENT field-line+ DEDENT
raw-tool     ::= json-object-line NEWLINE
json-object-line ::= <a complete JSON object (RFC 8259) beginning with "{" at column 0, written on one line with no internal NEWLINE>
```

A document is an optional `$defs` block followed by one or more tool entries (§3.1); a document consisting of only a `$defs` block is ill-formed. A `tool-entry` is either a CATS `tool-block` or a `raw-tool` — a fallen-back tool carried as verbatim JSON Schema (§7.5). The two are told apart by the first non-space character of the entry: `{` opens a `raw-tool`, any other character opens a `tool-block` (whose header is a `name` or `string-literal`, never `{`). A `raw-tool` is a single physical line: the whole JSON object is serialized without internal newlines, so the indentation scanner consumes it as one line and never has to match braces or distinguish structural braces from braces inside JSON strings (§7.5). It is given as an opaque terminal for the same reason `json-literal` (A.5) is — reproducing the JSON object grammar inside the CATS grammar would add bulk without changing how the tokenizer treats the line.

A `tool-block` body is optional: a header-only tool with no `INDENT … DEDENT` block is well-formed and encodes a parameterless tool (§3.3). A `definition` body is mandatory. The `tool-block` and `definition` productions are otherwise nearly identical in shape; they differ in position — definitions sit inside the `$defs` block, tool blocks at document scope — and in their header name. A tool block's name is a `name` (hyphen-extended) with a quoted-string escape hatch, because a tool name is contract-bound and cannot always be reshaped to fit the identifier grammar (§3.3); a definition's name is a bare `identifier`, since it is reached through the identifier-only reference syntax `$Name` and admitting other forms would require a less readable quoted reference (§3.2, §5.7). The `$defs` header carries no description (§3.2); a `tool-block` header MAY carry one, the same `description` production used by field lines (§4.6), defined in A.2.

`INDENT` and `DEDENT` are virtual terminals emitted by the indentation scanner when nesting opens and returns to an enclosing level (§3.4); they are defined in A.5.

## A.2 Field lines (§4)

```ebnf
field-line       ::= field-name required-marker? SP type-expression default? description? NEWLINE nested-block?
field-name       ::= name | string-literal
required-marker  ::= "*"
default          ::= SP "=" (value-literal | json-literal)
description      ::= SP "#" SP description-text
nested-block     ::= INDENT field-line+ DEDENT
```

A field line carries, in fixed order, a field name, an optional required marker, exactly one separating space, the type expression, an optional default, an optional description, and an optional nested block (§4.1). The field name and type expression are mandatory; everything from the default onward is optional.

`default` carries its own leading `SP`: a single space precedes `=`, and the value is glued directly to `=` with no following space (§4.5) — `count integer =5`, `config object ={}`. The leading space is unconditional, which keeps the production context-free; its purpose is to prevent the sequence `>=` from arising when the type expression ends in `>` (as in `array<string> =[]`), where a glued `=` would read as a comparison operator. Because the
space is always present, no production needs to inspect the preceding token.

A scalar default is a `value-literal` — a string (quoted per §2.6), number, boolean, or null (§4.5). An object or array default is a `json-literal`: inline JSON syntax such as `={}` or `=[]`, treated as a single opaque contiguous token whose content is governed by JSON's own grammar, not by CATS quoting or identifier rules, and which contains no internal whitespace (§4.5). `json-literal` is defined in A.5.

`description` is the same `SP "#" SP description-text` form used by the block headers of A.1; it is the single description form across the whole notation (§4.6). When `description` follows `default`, the `SP` that opens it separates the two.

*Note (required/default exclusivity, §4.1):* A field line MUST NOT carry both a `required-marker` and a `default`. The grammar permits the combination; §4.1 forbids it as semantically contradictory (a required field should have no value to fall back to).

## A.3 Type expressions (§5)

```ebnf
type-expression   ::= type-union | enum-union | single-value-enum | single-type
single-type       ::= base-type annotation-chain?
base-type         ::= primitive-type | array-type | object-type | reference
primitive-type    ::= "string" | "integer" | "number" | "boolean" | "null" | "any"
array-type        ::= "array" ("<" type-expression ">")?
object-type       ::= "object"
reference         ::= "$" identifier
type-union        ::= single-type ("|" single-type)+
enum-union        ::= value-literal ("|" value-literal)+
single-value-enum ::= value-literal
```

A type expression is a type union, a multi-value enum union, a single-value enum, or a single type (§5.1). An `annotation-chain` attaches at the `single-type` level (§5.1), so each branch of a type union carries its own chain independently and a bare type is the one-branch case of the same rule (§5.5). Neither enum form carries an annotation chain (§5.6).

`single-value-enum` derives a lone value in the type slot — `mode "automatic"` — which encodes JSON Schema's `const` (§5.6, §7.2). Because its body is a `value-literal`, it overlaps in the grammar with a bare `base-type`: the token `string` could derive as the type word or as a single-value string enum. The two are distinguished by the quoting rule of §2.6 — a value colliding with a type word MUST be quoted, so `string` is the type and `"string"` the enum value. This is the same permissive-grammar / prose-disambiguation split that separates `type-union` from `enum-union` (a union is one kind or the other by whether its branches are type words or value literals, §5.6).

*Note (union homogeneity, §5.6):* A pipe-union's branches MUST be either all type expressions or all value literals, never a mix. The grammar admits each kind separately but does not enforce the exclusivity between them.

## A.4 The annotation chain (§6)

```ebnf
annotation-chain ::= format? bounds? mult? length? regex? encoding? media? unique?
format           ::= ":" format-value
format-value     ::= (id-continue | "-")+
bounds           ::= bracket-open number? "," number? bracket-close
mult             ::= "%" number
length           ::= ":length[" number? "," number? "]"
regex            ::= ":regex[" string-literal "]"
encoding         ::= ":encoding[" identifier "]"
media            ::= ":media[" string-literal "]"
unique           ::= ":unique"
bracket-open     ::= "[" | "("
bracket-close    ::= "]" | ")"
```

The chain is a single fixed order of eight optional slots (§6.5); a chain whose slots appear out of this order is ill-formed. `format` is the first slot (§6.1). The grammar is permissive — it admits slot combinations that no single base type carries, and admits chains on `null` and `any` — and which slots are valid is gated by base type in prose (§6.5): a numeric type takes `format`, `bounds`, and `mult`; a string type takes `format`, `length`, `regex`, `encoding`, and `media`; an array takes `bounds` (read as an element count) and `unique`; `format` attaches to any *typed* base but not to `null` or `any` (§5.2). **Annotations must attach to a base type that admits them; this agreement is a prose constraint (§6.5), not a grammatical one. `null` and `any` admit no annotation at all.**

`format-value` is a verbatim passthrough token, not a CATS `identifier`: the converter applies no lookup or policy to it (§6.1), and real format values contain hyphens (`date-time`), digits (`int64`), and other word characters (`semver`, `phone`). It is therefore defined as one or more `id-continue` characters or hyphens, which admits every spec, OpenAPI, and custom format value of §6.1 while excluding the delimiters that begin the next chain slot or end the type expression. The grammar does not exclude the five annotation keywords (`length`, `regex`, `encoding`, `media`, `unique`) from `format-value`; their reservation in format position is a prose rule of §6.1 — a source `format` equal to one of them sends the tool to fallback (§7.5) rather than deriving here.

`bounds` is one production shared between numeric value-range bounds (§6.2) and array element-count bounds (§6.4). The bracket characters carry inclusivity: `[` `]` are inclusive, `(` `)` exclusive, and the four `bracket-open`/`bracket-close` pairings give the four permutations of §6.2. An omitted bound leaves the open bracket on its side, since an unbounded endpoint is never included (§6.2). In the array context both endpoints are always inclusive — `(` and `)` do not appear there — but that restriction is a prose rule of §6.4, not a separate production.

*Note (length brackets, §6.3):* `length` always uses inclusive brackets on both ends; JSON Schema defines no exclusive-length variant. Either bound may be omitted under the same open-bracket convention as `bounds`.

*Note (unconditional quoting, §6.3):* The `regex` and `media` values are always quoted as `string-literal`s regardless of content, because regex and MIME syntax overlap with the notation's structural characters. This is stated by their productions above (both take `string-literal`, never a bare form) and elaborated in §6.3.

## A.5 Lexical terminals (§2)

```ebnf
identifier       ::= id-start id-continue*
id-start         ::= [A-Za-z_]
id-continue      ::= [A-Za-z0-9_]
name             ::= id-start (id-continue | "-")*

string-literal   ::= '"' (string-char | string-escape)* '"'
string-char      ::= <any Unicode code point except U+0022 ("), U+005C (\), U+000A (LF), and U+000D (CR)>
string-escape    ::= '\"' | '\\'

value-literal    ::= string-value | number | "true" | "false" | "null"
string-value     ::= identifier | string-literal

json-literal     ::= <a JSON object or array, written as one contiguous token with no internal whitespace (RFC 8259)>

number           ::= "-"? int frac? exp?
int              ::= "0" | [1-9] [0-9]*
frac             ::= "." [0-9]+
exp              ::= ("e" | "E") ("+" | "-")? [0-9]+

description-text ::= description-bare | string-literal
description-bare ::= <one or more characters, none of which is NEWLINE or "#">

SP               ::= U+0020
NEWLINE          ::= LF | CRLF
INDENT           ::= <virtual terminal: indentation increases one level (§3.4)>
DEDENT           ::= <virtual terminal: indentation returns to an enclosing level (§3.4)>
```

`value-literal` is the shared production for any string, number, boolean, or null value; it serves both the enum branches of §5.6 and the `default` of §4.5. A `string-value` is written bare (an `identifier`) by default and quoted (a `string-literal`) only when the quote-when-ambiguous rule of §2.6 fires — when the bare token would collide with a JSON literal (`"true"`), with a type word (`"string"`), or with a structural delimiter. The grammar admits both forms; §2.6 decides which is well-formed in context.

`name` is `identifier` extended to permit ASCII hyphens after the first character (§2.3); it is the bare form of a tool name (A.1) and a field name (A.2). Every `identifier` is also a `name`, but a hyphenated `name` is not an `identifier` — which is why `reference` (A.3) and definition names stay on `identifier`, keeping the `$Name` reference syntax free of a quoted form. A tool or field name that falls outside `name` (a leading digit or hyphen, or a non-hyphen special character) takes the `string-literal` alternative in its respective production, under §2.6.

`number` reproduces the JSON number grammar (RFC 8259 §6, §2.5). CATS draws no lexical distinction between integer and number literals; the distinction is fixed by the type slot of the enclosing field (§2.5). The `int` production forbids leading zeros, matching JSON.

`json-literal` is the inline-JSON form of an **object or array** default — `{}`, `[]`, or any well-formed JSON object or array written as one contiguous token (§4.5). Scalar defaults (string, number, boolean, null) take the `value-literal` branch of `default` instead, so `json-literal` is restricted to the composite openers `{` and `[`; this keeps the two branches disjoint. Its internal syntax is JSON's, not CATS's: CATS quoting and identifier rules do not apply inside it, and it MUST contain no whitespace, so the whole default remains a single token. It is given as an opaque terminal because reproducing the full JSON object/array grammar inside the CATS grammar would add bulk without changing how a CATS tokenizer treats it — as one unbroken run of non-space characters following `=`.

`description-text` is free-form prose to end of line, written bare by default and quoted as a `string-literal` when it contains `#`, per the third quoting trigger of §2.6 (§4.6). The `description-bare` form therefore excludes `#` and `NEWLINE`; the quoted form admits `#`.

`SP` is a single space (`U+0020`); within a field line exactly one `SP` separates the field name from the type slot, and the `default` and `description` productions each carry their own leading `SP` (§2.2). `NEWLINE` is LF or CRLF, treated as equivalent (§2.1). `INDENT` and `DEDENT` are virtual terminals produced by the indentation scanner from the two-space-per- level rule of §2.2, under the scoping of §3.4; they carry no character content and are listed here only so the structural productions of A.1 and A.2 are complete.

## A.6 Deliberate omissions

The following are rules of CATS but not grammar productions, and are intentionally absent:

- `$defs` and the eight type words are reserved words (§2.4); they appear as terminals in the productions above and need no production of their own.
- The quote-when-ambiguous rule (§2.6) is a prose condition on when a `string-value` or `description-text` takes its quoted form; encoding it in the grammar would over-specify.
- The two-space indentation unit (§2.2) constrains what `INDENT` may emit; it is a lexical constraint, not a production.
- `additionalProperties` has no production: object closure is implicit in the `object` model (§5.4), not a surface construct.
- Annotation/base-type agreement (§6.5) and required-marker/default exclusivity (§4.1) are semantic constraints on otherwise-derivable forms, stated in prose and noted at their productions above.
- Name uniqueness (§3.5) — unique definition names, unique field names within a block, unique tool names within a document — and reference resolution (§5.7) are whole-document properties a context-free grammar cannot check; they are prose well-formedness rules enforced after parsing.
- Mixed-type enum members (§5.6) and a `type` array combined with a sibling constraint (§5.5) are dispositions of the conversion semantics, not grammatical forms; the grammar admits the surface syntax and the converter routes these to fallback.

# Appendix B: Worked Examples

This appendix is non-normative. It illustrates the notation defined in §2–§6 and the conversion semantics of §7 through four complete tool definitions, ordered from minimal to composite. Each example pairs a JSON Schema tool definition with its CATS form, following the labeling convention of §1.4. The JSON side of each pair is written in the canonical output envelope of §7.2.1 — a self-contained tool object carrying its `name` and parameter schema — with keywords in the canonical order of §7.2.

## B.1 A minimal tool

The smallest realistic tool exercises the constructs that appear in nearly every CATS document: a tool header with an inline description (§3.3), the required marker (§4.2), a multi-value enum union (§5.6), defaults (§4.5), and field descriptions (§4.6).

```json
{
  "name": "get-weather",
  "description": "Look up current weather for a location",
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "City name or \"lat,lon\""
    },
    "units": {
      "type": "string",
      "enum": ["celsius", "fahrenheit"],
      "default": "celsius"
    },
    "include_hourly": {
      "type": "boolean",
      "default": false
    }
  },
  "required": ["location"],
  "additionalProperties": false
}
```

```cats
get-weather # Look up current weather for a location
  location* string # City name or "lat,lon"
  units celsius|fahrenheit =celsius
  include_hourly boolean =false
```

Three reading cues carry the structure. The hyphenated tool name `get-weather` is a bare name (§2.3) and needs no quoting. `location*` is the only required parameter — the `required` array of the source becomes the `*` marker, and optionality is the unmarked default (§4.2). The enum union `celsius|fahrenheit` declares no type word; the string type is recovered from the value forms, and the reverse direction re-emits the explicit `"type": "string"` alongside `enum` per §7.2.

## B.2 The annotation vocabulary

This example exercises every annotation family of §6 on one tool: format (§6.1), numeric bounds in inclusive, exclusive, and open forms plus `%` divisibility (§6.2), string length, regex, encoding, and media type (§6.3), and array count bounds with `:unique` (§6.4). Each chain follows the canonical order of §6.5.

```json
{
  "name": "create_account",
  "description": "Register a new user account",
  "type": "object",
  "properties": {
    "username": {
      "type": "string",
      "minLength": 3,
      "maxLength": 20,
      "pattern": "^[a-z0-9_]+$",
      "description": "Lowercase handle"
    },
    "email": {
      "type": "string",
      "format": "email"
    },
    "age": {
      "type": "integer",
      "minimum": 13,
      "description": "Must be 13 or older"
    },
    "balance": {
      "type": "number",
      "default": 0,
      "minimum": 0,
      "multipleOf": 0.01,
      "description": "Starting credit in dollars"
    },
    "discount": {
      "type": "number",
      "exclusiveMinimum": 0,
      "exclusiveMaximum": 1,
      "description": "Fraction strictly between 0 and 1"
    },
    "avatar": {
      "type": "string",
      "contentEncoding": "base64",
      "contentMediaType": "image/png",
      "description": "Profile image"
    },
    "tags": {
      "type": "array",
      "items": { "type": "string" },
      "maxItems": 10,
      "uniqueItems": true,
      "description": "Up to 10 distinct tags"
    }
  },
  "required": ["username", "email"],
  "additionalProperties": false
}
```

```cats
create_account # Register a new user account
  username* string:length[3,20]:regex["^[a-z0-9_]+$"] # Lowercase handle
  email* string:email
  age integer[13,] # Must be 13 or older
  balance number[0,]%0.01 =0 # Starting credit in dollars
  discount number(0,1) # Fraction strictly between 0 and 1
  avatar string:encoding[base64]:media["image/png"] # Profile image
  tags array<string>[,10]:unique # Up to 10 distinct tags
```

The bracket conventions of §6.2 all appear. `age integer[13,]` leaves the upper bound open in the converter's canonical inclusive form — `[13,)` would be an equivalent input spelling, but the converter always emits `[13,]`. `discount number(0,1)` uses exclusive parentheses on both ends, mapping to `exclusiveMinimum`/`exclusiveMaximum`. On `balance`, the chain order of §6.5 places bounds before the `%` divisor, and the default follows the whole type expression with its unconditional leading space (§4.5). The `regex` and `media` values are quoted unconditionally (§6.3); `length`, `encoding`, and the bounds are not. On `tags`, the same bracket notation that means a value range on a number means an element count on the array (§6.4, §6.5), and `:unique` closes the chain.

Every construct in this example is in the directly-encoded set of §7.1 and emits in the canonical forms of §7.2, so the pair above round-trips keyword-identically: converting the JSON to CATS and back recovers the JSON shown, byte-comparable up to whitespace.

## B.3 Structure, composition, and references

This example exercises the document-structure and composition layer: a `$defs` block with a referenced definition (§3.2, §5.7), a reference inside array parameterization with count bounds (§5.3, §6.4), a nested object block with a quoted field name (§4.7, §2.4), a type union with a branch-level annotation (§5.5, §4.4), and a single-value enum encoding `const` (§5.6). It also demonstrates the keyword-level losses of §7.4 on round-trip.

```json
{
  "name": "create_event",
  "description": "Schedule a calendar event",
  "$defs": {
    "Attendee": {
      "type": "object",
      "description": "One person invited to the event",
      "properties": {
        "name": { "type": "string" },
        "email": { "type": "string", "format": "email" },
        "rsvp": {
          "type": "string",
          "enum": ["accepted", "declined", "tentative"],
          "default": "tentative"
        }
      },
      "required": ["name", "email"],
      "additionalProperties": false
    }
  },
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "minLength": 1,
      "maxLength": 120
    },
    "start": {
      "type": "string",
      "format": "date-time",
      "description": "Event start, RFC 3339"
    },
    "end": {
      "type": ["string", "null"],
      "description": "Null for an open-ended event"
    },
    "location": {
      "type": "object",
      "properties": {
        "venue": { "type": "string" },
        "room.number": { "type": "string" }
      },
      "required": ["venue"],
      "additionalProperties": false,
      "description": "Where the event takes place"
    },
    "attendees": {
      "type": "array",
      "items": { "$ref": "#/$defs/Attendee" },
      "minItems": 1,
      "description": "At least one attendee"
    },
    "source": {
      "enum": ["api"],
      "description": "Always the literal string \"api\""
    }
  },
  "required": ["title", "start", "attendees"],
  "additionalProperties": false
}
```

```cats
$defs
  Attendee # One person invited to the event
    name* string
    email* string:email
    rsvp accepted|declined|tentative =tentative

create_event # Schedule a calendar event
  title* string:length[1,120]
  start* string:date-time # Event start, RFC 3339
  end string|null # Null for an open-ended event
  location object # Where the event takes place
    venue* string
    "room.number" string
  attendees* array<$Attendee>[1,] # At least one attendee
  source "api" # Always the literal string "api"
```

The `$defs` block sits at column 0 ahead of the tool blocks (§3.1); `Attendee` is a bare-identifier definition name in the recommended PascalCase (§3.2), reached through the `$`-sigil reference in `array<$Attendee>` (§5.7), with the `[1,]` count bound glued after the parameterization (§6.4). The nested `location` block indents one level below its parent field line, and `"room.number"` takes the quoted-name escape hatch because a dot falls outside the name grammar (§2.4); the quotes are stripped on conversion, so the emitted property key is `room.number`. On `end`, the union's `null` branch is the canonical nullable encoding (§5.5). `source "api"` is a single-value enum: the quotes mark the token as a value rather than a type word or bare name (§2.6, §5.6).

This pair does **not** round-trip keyword-identically; it round-trips behavior-identically with two keyword-level differences of the kind §7.4 documents. The reverse direction emits the source's nullable `end` and one-member `source` enum in the canonical forms of §7.2:

```json
"end": {
  "anyOf": [{ "type": "string" }, { "type": "null" }],
  "description": "Null for an open-ended event"
},
"source": {
  "const": "api",
  "description": "Always the literal string \"api\""
}
```

The recovered `anyOf` validates exactly the values the source's `type: ["string", "null"]` does, and `const: "api"` exactly those of `enum: ["api"]`. The surface keywords differ; the accepted value sets do not.

## B.4 A mixed document with a fallback tool

The final example shows the tool-level fallback of §7.5 in a two-tool document. The first tool is fully encodable. The second routes a payment through one of two mutually exclusive method shapes using `oneOf` — exclusive-match semantics that CATS does not encode (§5.5, §7.4) — so the entire tool is carried verbatim as raw JSON Schema, occupying a top-level position in the document's tool sequence. The reader distinguishes the two forms by the first non-space character of the entry: `{` opens a raw JSON Schema tool, anything else a CATS tool block (§7.5).

```cats
search_products # Find products in the catalog
  query* string
  limit integer[1,50] =10

{"name": "route_payment", "description": "Charge a payment method", "type": "object", "properties": {"method": {"oneOf": [{"type": "object", "properties": {"card_token": {"type": "string"}}, "required": ["card_token"], "additionalProperties": false}, {"type": "object", "properties": {"iban": {"type": "string"}}, "required": ["iban"], "additionalProperties": false}], "description": "Exactly one payment method"}}, "required": ["method"], "additionalProperties": false}
```

In the output envelope of §7.2.1, this document serializes to a two-element array: the CATS-encoded `search_products` schema followed by the `route_payment` schema exactly as written above, byte-for-byte. The fallen-back tool is neither partially encoded nor dropped — no `anyOf` substitution is made for the `oneOf`, because `anyOf` accepts a value matching both branches where `oneOf` rejects it, and that substitution would widen the accepted value set (§7.4). A consumer of the output treats both array entries uniformly as JSON Schema tool definitions; the converter's fallback report records that `route_payment` fell back and that `oneOf` triggered it (§7.5).