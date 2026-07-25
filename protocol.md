# CATS Usage Protocol

*A developer's guide to passing CATS tool definitions to a language model.*

This is a guide, not a spec. The format itself is defined in `spec.md`; this doc just covers how to use it: what goes in the prompt, what to tell the model, and how the model should report a tool call. "Should" and "recommended" here mean "this is what works in practice," not a conformance rule.

**The primer is a recommendation.** The central thing this doc describes — the "primer", a short block of text you put in the system prompt — is not part of CATS and not required by it. It's a suggested way to give the model enough context to read CATS correctly. You can write your own, tweak ours, or skip it if your model already handles CATS well. Everything below is the version we found reliable; treat it as a starting point.

It's written for someone who already has CATS tool definitions — from the converter or hand-written — and needs to wire them into a real request.

---

## 1. Why CATS needs a usage protocol at all

JSON Schema doesn't need a doc like this, and the reason explains everything that follows.

When you call a tool-calling API today, you don't explain JSON Schema to the model. You drop your schema into the provider's `tools` parameter (OpenAI), `tools` block (Anthropic), or `functionDeclarations` (Gemini), and the provider handles two things for you: it frames the tools for the model ("here's what you can call, in a format you know"), and it gives the model a structured channel to emit a call into, which the provider parses back into a clean object for you.

CATS gets neither for free. The provider has never heard of it, so it can't ride the native `tools` parameter — it goes into the system prompt as plain text. And since it's not on the native channel, the model won't be routed into the structured tool-call output either, so you have to tell it how to report a call.

So using CATS means rebuilding, in the prompt, the two things the native API used to hand you: the "here are your tools" framing and the output channel. That's the protocol's whole job, and it breaks into three pieces:

1. **The primer** — tell the model what CATS is and how to read it (§3).
2. **The tool blocks** — put the actual CATS definitions in the prompt (§4).
3. **The output contract** — tell the model how to report a call (§5).

---

## 2. Keep the primer short: it's a fixed cost

One thing to keep in mind before the mechanics. CATS saves tokens on every tool definition, but the primer costs tokens once per request no matter how many tools you send. So the primer is a fixed cost and the savings are per-tool — which means CATS only comes out ahead once you're sending enough tools for the savings to cover the primer. Wrap a 200-token primer around one tiny tool and you've spent more than you saved.

That's the whole reason to keep the primer lean, and the reason it's **calibrated** (§3.2): it only teaches the features actually in *this* request, so it stays as small as possible and breaks even sooner.

(Measured break-even lives in `eval/break_even.py`, not here. Those runs use the same conversion flags as Part 1 — `assume_closed=True`, `map_python_types=True` — on the BFCL converted corpus with the calibrated primer. CATS overtakes raw JSON at roughly **5–8 tools** depending on tokenizer — tiktoken ~6.2, Qwen ~7.8, Anthropic ~5.5 — with **10% total-prompt savings** around N≈9–14. Those numbers are why the primer is built the way it is.)

---

## 3. The primer

The primer goes in the system prompt, ahead of the tool definitions, and teaches the model to read CATS. It has two parts: a **core intro** that's always the same (§3.1), plus a **calibrated body** built from whichever CATS features the request actually uses (§3.2). The generator (§6) assembles the whole thing for you; this section is what it produces and why.

### 3.1 The fixed core

Every primer starts with the same **intro**: what CATS is, how a tool block is shaped (name line → indented parameters), and how to read a parameter line left to right (name, type, optional `#` description). That intro is always emitted:

> The following tools are described in CATS, a compact notation for tool definitions. Read each tool like a typed function signature.
>
> - Each tool starts with its name on its own line, optionally followed by `#` and a description. The indented lines below are its parameters.
> - Each parameter line reads left to right: the parameter name, then its type, then, optionally, a description introduced by `#`.

The **required/optional rule** is calibrated on the whole prompt, not gated like the body clauses. The manifest records `required_uniformity` across all tool parameter lines (§6.2): `mixed`, `all_required`, or `all_optional`. Zero-parameter tool sets count as `all_optional` — no `*` appears anywhere.

**`mixed`** — at least one parameter has `*` and at least one does not. Emit the full rule as a third bullet:

> - A parameter is **required** only if its name is immediately followed by `*`. A parameter with no `*` is optional.

**`all_required`** — every parameter line carries `*`. Replace the mixed rule with:

> - All fields of the below tools are required (indicated by `*`).

**`all_optional`** — no parameter line has `*` (including header-only tools). Omit the `*` rule entirely; the intro is enough.

On OpenAI tiktoken, the core is **112 tokens** in the mixed case, **100** when all required, **84** when all optional. Anything feature-specific — including default values (`=`) — is left to the calibrated body.

### 3.2 The calibrated body: only teach what the model can't guess

The body teaches the features in the request — only those, and within each, only the part the model can't pick up from the examples it's about to read.

That second filter is where most of the savings come from. A lot of CATS features are **self-teaching**: the model sees one example and generalizes, no explanation needed. Spending tokens to explain those is waste. Some features are half self-teaching, and the primer should only cover the half that isn't.

Numeric bounds are the clean example. `limit integer[1,100]` reads as "1 to 100" on sight — interval notation is universal, so it teaches itself. What *doesn't* teach itself is that the bracket style matters: `integer[0,100)` uses a parenthesis to exclude the upper end. So the bounds clause doesn't explain intervals at all — just the bracket rule, and only when an exclusive bracket actually shows up. All-inclusive bounds get no clause.

#### The clause library

Each clause is gated on a feature flag from the manifest (§6). The text shown is what the generator emits verbatim. Clauses appear in this order, after the core. Anything not listed here is self-teaching and gets no clause on purpose.

`**`* required marker** — calibrated in the core (§3.1), not the body. `mixed` emits the full required/optional rule; `all_required` emits a one-line “everything has `*`” rule; `all_optional` omits the rule when no `*` appears anywhere (including zero-parameter tools).

**Default values** — *when any parameter line carries `=value`.*

> `=value` sets a default; omit the parameter to use it.

(Taught only when a default actually appears — `=` reads as assignment on sight, but the "omit the parameter and the default applies" rule is worth stating when defaults are in play.)

**Numeric bounds** — *when any numeric bound is present.* Built from up to three fragments, each gated on its own, so a tool only pays for what it uses.

Base fragment (any numeric bound):

> Numeric ranges use interval notation: `integer[1,100]` accepts 1 to 100.

Inclusivity fragment, three-way on which bracket styles appear:

- exclusive brackets only (e.g. `(0,1)`, no plain `[ ]` range): `A parenthesis excludes that endpoint, so (0,1) is between 0 and 1 with neither included.`
- both styles present: `A square bracket includes the endpoint and a parenthesis excludes it, so [0,100) allows 0 but not 100.`
- inclusive only: no fragment — square brackets teach themselves.

Open-bound fragment (any bound omits an endpoint, e.g. `[1,]` or `[1,)`):

> A missing number means that side is unbounded.

(The split exists because inclusivity and open-endedness are independent: a tool can use one without the other, and the inclusivity wording differs depending on whether the model needs the square-vs-paren contrast or just the parenthesis rule.)

`**%` multipleOf** — *when any type carries `%`.*

> A `%` after a number means the value must be a multiple of what follows it: `integer%5` accepts multiples of 5, `number%0.01` rounds to hundredths.

(Taught whenever present — `%` as "multiple of" is a CATS convention the model won't have seen, and bare it reads like modulo.)

**String length** — *when any type carries `:length[...]`.*

> `:length[1,20]` after a string constrains its length: at least 1 character, at most 20.

Plus the open-bound fragment (shared with numeric bounds), appended only when a length bound omits an endpoint. (Taught because `:length[1,20]` could otherwise read as a value range rather than a character count. Length brackets are always inclusive, so no inclusivity fragment is ever needed.)

**String regex** — *when any type carries `:regex[...]`.*

> `:regex["..."]` after a string means the value must match that regular expression.

**String encoding / media type** — *when any type carries `:encoding[...]` or `:media[...]`.*

> `:encoding[base64]` means the string is encoded that way; `:media["application/pdf"]` gives its MIME type.

**Array element typing** — *when any type uses `array<...>`.*

> `array<T>` elements all have type T (T can be nested).

(Mostly self-teaching, but the "element type can itself be complex" point is worth one sentence so the model doesn't assume elements are always primitives.)

**Array count bounds** — *when any `array<...>` carries `[...]`.*

> Brackets after an array, like `array<string>[1,10]`, constrain how many elements it has — here, 1 to 10. (This is a count of elements, not a range of values.)

(The parenthetical disambiguates: the same bracket notation means a value range on a number but an element count on an array.)

**Array uniqueness** — *when any type carries `:unique`.*

> `:unique` on an array means all its elements must be distinct.

**Type unions** — *when a type slot has `|` joining type words.*

> A `|` between types means any one of them: `string|array<string>` accepts either. `X|null` marks a nullable field.

(The `|` reads as "or" on its own; the `X|null` nullable idiom is worth naming since it's the common case.)

**Multi-value enum** — *when a type slot has `|` joining values.*

> A `|` between values restricts the field to exactly those: `sort relevance|price|newest` allows only those three.

(Same `|` as a type union, different meaning — this clause pins the value reading. If both appear in a request, both clauses are emitted.)

**Single-value enum** — *when a type slot has a single quoted value* (gated independently of multi-value).

> A quoted value like `mode "automatic"` means the field must equal exactly that.

(Split from multi-value so a tool with only `a|b|c` never pays for this sentence, and vice versa.)

`**$defs` references** — *when a type slot has a `$`-prefixed name.*

> A type written `$Name` (for example `home_address $Address`) refers to a reusable shape defined under the `$defs` block at the top of the document. Look there for that shape's fields.

(The `$Name` → `$defs` indirection needs a pointer, or the model treats `$Address` as an opaque type word.)

**Header-only tools** — *when any tool block has a name but no parameter lines.*

> A tool shown with a name but no parameter lines takes no arguments. Call it with an empty arguments object: `{"name": "<tool_name>", "arguments": {}}`.

(Zero-parameter tools encode the empty closed object; the output contract shows a filled `arguments` object by default, so this clause is needed when such a tool is present.)

**Fallback (raw JSON Schema) tools present** — *when any tool entry begins with `{`.*

> Some tools below are written as raw JSON Schema (a block starting with `{`) instead of CATS. Read those exactly as you normally read JSON Schema tool definitions. Every tool is either fully CATS or fully JSON Schema — the two are never mixed inside one tool.

(Only emitted when a fallback tool is actually present — the mixed-document case, §4.2. Most requests won't have it.)

#### What deliberately gets no clause

So the omissions read as choices, not oversights:

- **Bare primitive types** (`string`, `integer`, `number`, `boolean`, `null`, `any`) — the model knows these from JSON Schema; "read it like a typed function signature" covers it.
- **Inclusive-only numeric ranges** (`[1,100]`) — interval notation is universal; only the exclusive and open cases get taught, and only when present.
- **Descriptions** (`# text`) — taught once in the core as part of the field-line order.
- **Nested object blocks** — the indentation is self-evident once the core says indented lines are the block's contents.
- **Tools that always have parameters** — when every tool in the request has at least one parameter line, the empty-arguments clause is omitted.
- **Uniform required/optional prompts** — when every parameter is required or none has `*`, the mixed-case `*` rule and mixed-case contract sentence are omitted in favor of the shorter variants (§3.1, §5.1).

### 3.3 The full-grammar primer (a baseline)

The generator can also emit a **full-grammar primer**: one static block teaching every feature regardless of the request. It's longer and breaks even later, so it's not the default.

The total fixed overhead of that worst-case prompt shell — full-grammar primer prose plus the two `---` section breaks and the output contract (empty tool fence placeholder) — is **676 tokens** on OpenAI tiktoken (GPT-5.X), **710** on Qwen3.5-9B, and **743** on Anthropic (Claude Sonnet 4.6 / Opus 4.6), as measured by `eval/break_even.py`. The primer prose alone is ~590 tokens on tiktoken.

Nearly every real request is much smaller. The mixed-case core is **112 tokens** (OpenAI tiktoken); calibrated primer prose on typical requests usually falls between **112 and ~300 tokens**, before tool blocks. At N=50 tools on the BFCL converted corpus, the mean calibrated primer is ~**41%** of the full-grammar shell size (~276–303 tokens vs 676–743).

### 3.4 Why the per-request primer isn't "tuning the prompt"

Since the calibrated primer changes per request, someone might suspect the prompt is being hand-tuned per example to make CATS look good. It isn't: **the primer is a deterministic function of the feature manifest** (§6). Same features in, same primer out, computed mechanically with no human picking what to teach. The full-grammar primer (§3.3) is kept around precisely as a fixed-prompt comparison point that sidesteps the worry entirely.

---

## 4. Placing the tool blocks

### 4.1 Mechanics

The CATS definitions go in the system prompt after the primer, set off with a `---` separator, in a fenced code block:

```
---

```

get-weather # Look up current weather for a location
  location* string # City name or "lat,lon"
  units celsius|fahrenheit =celsius
  include_hourly boolean =false

```

```

The separator gives the model a clear structural break between the prose primer and the tool definitions — markdown sectioning reads as "new section, shift attention" more reliably than a blank line does. We use plain `---` horizontal rules only (no `##` section headings) to keep framing overhead small. The inner fence sets the CATS apart as one structured unit; a language tag (````cats`) is harmless but does nothing, so plain ````` is fine. Multiple tools go in the same block, blank-line separated, in the order the converter emits (`spec.md` §7.2.1). No need for a fence per tool.

### 4.2 Mixed and all-fallback documents

Some tools can't be expressed in CATS and are carried verbatim as raw JSON Schema (`spec.md` §7.5). Two cases:

**Mixed** — some CATS, some raw JSON. The raw tools sit in the same fenced block, each a `{`-opening JSON object, in declaration order. The fallback clause (§3.2) tells the model a `{` entry is JSON Schema. This is the case that needs both the protocol and the fallback clause.

**All-fallback** — *every* tool fell back, so the converter produced zero CATS. There's nothing to teach and nothing to compress, so don't use CATS here at all — send the JSON Schema tools through the native `tools` channel like normal. A primer would just be overhead around nothing. The generator (§6) detects this (`all_fallback=True`, empty `primer_text`); `build_system_prompt` refuses to assemble a prompt — use the native channel instead.

---

## 5. The output contract

Since CATS bypasses the native tool-call channel (§1), you have to tell the model how to report a call. The shape below is what we recommend — deliberately the `name` + `arguments` shape every native channel already uses, so it's something the model emits reliably.

This protocol does **one call per turn**, which keeps the contract and the parser simple and matches most eval setups. (Parallel calls are a possible extension but out of scope — they'd need an array envelope and a heavier parser.)

### 5.1 Recommended contract text

Goes in the system prompt after the tool blocks, set off with its own `---` separator. The generator (`build_output_contract`, §6) picks the **arguments** sentence from the same `required_uniformity` flag as the core (§3.1).

**Mixed** (`*` and non-`*` parameters both appear):

> To call a tool, respond with only a single fenced JSON block in this exact shape, and nothing else:
>
> ```json
> {"name": "<tool_name>", "arguments": { ... }}
> ```
>
> Arguments are ordinary JSON matching the tool's parameters. Include every required parameter; optional parameters only when you have a value for them. If no tool applies, respond in plain text with no JSON block.

**All required** (every parameter line has `*`):

> … *(same opener and JSON fence)* …
>
> Arguments are ordinary JSON matching the tool's parameters. Include every required parameter. If no tool applies, respond in plain text with no JSON block.

**All optional** (no `*` on any parameter line, including zero-parameter tools):

> … *(same opener and JSON fence)* …
>
> Arguments are ordinary JSON matching the tool's parameters. Include any parameters you have values for. If no tool applies, respond in plain text with no JSON block.

### 5.2 The two rules this is really enforcing

Short as it is, the contract carries two rules that are the usual failure points.

**Read CATS, write JSON.** The arguments are plain JSON — what the tool's JSON Schema would expect — not CATS shorthand. A model fresh off reading CATS may be tempted to answer in the same compact style, so the contract says outright that arguments are ordinary JSON (`spec.md` §1.1). This is the most important sentence in the whole protocol. CATS only ever encodes the outbound definition, never the arguments the model emits — the contract just makes that promise visible to the model.

**Required vs optional in the call.** When the prompt is `mixed`, the contract spells out which parameters to include. When every field is required or every field is optional, that sentence shortens accordingly (§5.1) instead of repeating the `*` rule the core already taught.

**Define the no-call case.** The native channel has a built-in "no tool chosen" state; a text contract doesn't, so you have to say what "no call" looks like — here, plain text with no JSON block. Leave it undefined and the model may force a bad call or emit a half-formed one. Your parser keys off the fenced `json` block: present → parse a call; absent → plain-text response.

### 5.3 Parsing the result

Find the fenced `json` block, parse it as JSON, read `name` and `arguments`. The fence is what lets you find the call even if the model wraps it in prose despite the "nothing else" instruction — worth keeping as a safety margin even though the contract asks for the block alone.

---

## 6. The generator

The protocol above is mechanical, so it's automated. The generator lives in `primer.py`, a sibling of the converter's `cats.py` — it imports from the converter but the converter never imports it.

### 6.1 What it does

Give it tool definitions, get back the primer text for the system prompt — or the full system prompt via `build_system_prompt`, which joins primer prose, fenced tools, and output contract through `assemble_prompt_sections` (`primer` + `---` + tools + `---` + contract). It takes either input form:

- **JSON Schema in** — runs the converter (`cats.convert_with_report_for_tool_calling`, or `cats.convert_with_report` with the same flags), then walks the resulting CATS AST to build a **feature manifest**. `generate_primer_from_json` defaults to `assume_closed=True` and `map_python_types=True`. Plain `cats.convert()` / `cats.convert_with_report()` still default both flags to `False` for strict behavior-preserving conversion.
- **CATS in** — for when you already have CATS and don't want to re-convert. It parses the CATS with the converter's existing parser (`parse_text`) into the same AST, then walks that.

Both paths end up at the same AST and run the same walk, so there's one manifest builder and one primer assembler behind both. (The handoff originally imagined a separate lightweight "feature detector" for the CATS path, but the converter's parser already produces the right AST, so reusing it is less code and can't drift from the converter's own reading.)

### 6.2 The manifest

The manifest is the small struct the AST walk fills in and the assembler reads. It carries enough detail to gate each clause correctly — which means a few features split into independent sub-flags, since some clauses are built from fragments rather than emitted whole. Numeric bounds carry three: `bounds_inclusive` (a `[ ]` bound appears), `bounds_exclusive` (a `( )` bound appears), and `bounds_open` (a bound omits an endpoint). The inclusivity fragment is chosen from the first two (`exclusive` alone → the parenthesis sentence; both → the contrast sentence; inclusive alone → nothing), and the open-bound fragment fires off the third. Enums split into `enum_multi` and `enum_single` the same way. `has_default_value` gates the default-value clause when any parameter line carries `=value`. `has_parameterless_tool` gates the empty-arguments clause when any tool has a header but no parameter lines. `required_uniformity` (`all_required` | `all_optional` | `mixed`) gates both the core's `*` rule (§3.1) and the output contract's arguments sentence (§5.1); it is computed across all tool parameter lines, with zero-parameter prompts treated as `all_optional`. A coarse "bounds present" flag wouldn't be enough — you'd be back to emitting whole clauses and the verbosity the fragments exist to avoid.

### 6.3 A free correctness check

Both paths build the same AST, so for the same underlying schema they must produce the same manifest and the same primer:

> `manifest(json_schema) == manifest(parse_text(cats_of(json_schema)))`

That's a cheap regression test across the corpus. It's nearly tautological now that both paths share the parser — but it still catches the case where the converter's forward direction and `parse_text` disagree on some construct, and it pins down which side to fix.

---

## 7. A full example

Here's a complete system prompt for a one-tool request. The tool uses defaults (`=10`, `=relevance`), a mixed bound (`[1,50)` — inclusive lower, exclusive upper), and a multi-value enum, so the primer is calibrated to exactly those: the default-value clause, the bounds clause with the both-styles fragment, and the enum clause. No `$defs`, no fallback, so those clauses are absent.

````
The following tools are described in CATS, a compact notation for tool definitions. Read each tool like a typed function signature.

- Each tool starts with its name on its own line, optionally followed by `#` and a description. The indented lines below are its parameters.
- Each parameter line reads left to right: the parameter name, then its type, then, optionally, a description introduced by `#`.
- A parameter is **required** only if its name is immediately followed by `*`. A parameter with no `*` is optional.

`=value` sets a default; omit the parameter to use it.

Numeric ranges use interval notation: `integer[1,100]` accepts 1 to 100. A square bracket includes the endpoint and a parenthesis excludes it, so `[0,100)` allows 0 but not 100.

A `|` between values restricts the field to exactly those: `sort relevance|price|newest` allows only those three.

---

```
search # Search the catalog
  query* string
  max_results integer[1,50) =10
  sort relevance|price|newest =relevance
```

---

To call a tool, respond with only a single fenced JSON block in this exact shape, and nothing else:

```json
{"name": "<tool_name>", "arguments": { ... }}
```

Arguments are ordinary JSON matching the tool's parameters. Include every required parameter; optional parameters only when you have a value for them. If no tool applies, respond in plain text with no JSON block.
````

Every line is load-bearing for this request. `required_uniformity` is **mixed** (`query*` required, `max_results` and `sort` optional), so the core carries the full `*` rule and the contract carries the mixed arguments sentence. The default-value clause appears because both optional parameters carry `=`. The bounds clause is the base fragment plus the both-styles fragment, because `[1,50)` mixes an inclusive `[` with an exclusive `)` — and there's no open-bound sentence, since neither endpoint is omitted. The enum clause is the multi-value form only; no single-value enum, so that sentence is absent. Add an open bound or a fixed `const`-style value and the matching fragment appears; make every bound inclusive and the bracket sentence drops out entirely.