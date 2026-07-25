"""
validate.py — the CATS semantic validator (AST -> list of errors).

Third stage of the pipeline:

    text --[lexer]--> tokens --[parser]--> AST --[validate]--> AST --[serializer]--> text

Input  : a finished `Document` AST (nodes.py), as built by parser.parse().
Output : errors from validate(); errors and warnings from validate_with_warnings().
An EMPTY error list means the tree is legal. Warnings are legal-but-noteworthy
observations and never appear in the errors list.

CONTRACT — pure reporting, collect-all, no mutation:
  - Validation READS the tree and never rewrites, normalizes, or "fixes" a node.
  - validate() returns only errors (empty => legal). validate_with_warnings()
    returns (errors, warnings) from the same single pass.
  - It collects EVERY violation in one pass; it does not stop at the first. A
    tool author (and the eval corpus) needs all problems at once.
  - Each error carries a message, a §-citation, and a location when the tree
    offers one. (See the note below: the current AST carries no line/col, so
    locations are absent in practice — the field is here for when nodes gain
    positions, and callers should treat it as optional.)

WHAT THIS DOES NOT CHECK:
  - Base-type annotation gating (§6.5) — a numeric bound on a string, :unique on
    a non-array, annotations on ``null``/``any``, etc. Enforcement lives in
    ``parser._apply_annotations`` (ParseError before a tree exists); the
    validator deliberately does not duplicate those rules.
  - Canonicalization concerns the serializer owns, which are NOT legality:
      * a two-member true|false union/enum (§5.6) — canonicalizes to boolean;
      * an annotation chain written out of canonical order (§6.5 order) — not
        even retained in the tree;
      * oneOf-vs-anyOf and open-vs-closed object distinctions (§5.5/§5.4/§7.4).
    validate.py stays silent on all of these by design.

The rules enforced (each cites its spec section at the error site) are the prose
rules the permissive parser deliberately leaves standing in the tree:
  1. required + default exclusivity            (§4.1 / §4.5)
  2. reference resolution                      (§5.7)
  3. document has >= 1 tool block              (§3.1)
  4. non-empty $defs when present              (§3.2)
  5. definition name is a bare identifier      (§3.2 / §2.3)
  6. union homogeneity (no value branch)       (§5.6)
  7. enum members share one base type          (§5.6)
  8. bounds sanity: lower <= upper             (§6.2 / §6.3 / §6.4)
  9. duplicate field names within a block      (§4.2)
 10. duplicate definition names in $defs       (§3.2 / §3.5)

Warnings (legal trees only; never mixed into the errors list):
  - unused $defs definition                  (§3.2) — defined but never referenced
    by any tool (directly or transitively); mirror of rule 2 (§5.7).
  - duplicate tool names in the document   (§3.5) — SHOULD be unique
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from nodes import (
    NO_DEFAULT,
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

# A bare identifier per §2.3: an ASCII letter/underscore start, then letters,
# digits, and underscores. No hyphens, no quoting — the stricter form a $defs
# definition name must take (§3.2), distinct from the hyphen-permitting `name`.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class ValidationError:
    """One semantic rule violation found in the tree.

    `message`  — human-readable description of what is wrong.
    `section`  — the spec §-citation the rule comes from (e.g. "§4.1").
    `line`/`col` — 1-based source location when available. The current AST nodes
                 carry no position, so these are None in practice; the fields
                 exist so locations can be threaded through later without an API
                 change. Callers must treat them as optional.
    """

    message: str
    section: str
    line: Optional[int] = None
    col: Optional[int] = None

    def __str__(self) -> str:
        where = ""
        if self.line is not None:
            where = f" (line {self.line}"
            where += f", column {self.col})" if self.col is not None else ")"
        return f"{self.message} [{self.section}]{where}"


@dataclass(frozen=True)
class ValidationWarning:
    """One legal-but-noteworthy observation found in the tree.

    Same shape as ValidationError but a separate type so callers can keep
    "errors empty => valid" without treating warnings as violations.
    """

    message: str
    section: str
    line: Optional[int] = None
    col: Optional[int] = None

    def __str__(self) -> str:
        where = ""
        if self.line is not None:
            where = f" (line {self.line}"
            where += f", column {self.col})" if self.col is not None else ")"
        return f"{self.message} [{self.section}]{where}"


def _collect_reference_names(type_node: object, names: set[str]) -> None:
    """Add every `$defs` Reference name reachable inside one type node to `names`.

    Mirrors the serializer's reference walk (§5.7): recurses through Array
    elements, Union branches, and Object field types only.
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
    """Definition names reachable from `fields`, TRANSITIVELY through `$defs` (§5.7).

    Seeds from references in the given fields, then walks into each referenced
    definition's own fields. The `resolved` set is the cycle guard. Names with
    no matching definition are still collected (dangling refs — rule 2 errors).
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
            continue
        for fld in definition.fields:
            _collect_reference_names(fld.type, pending)
    return resolved


def _document_used_definition_names(document: Document) -> set[str]:
    """Every $defs definition name referenced by ANY tool, transitively (§3.2/§5.7)."""
    defs_by_name = {d.name: d for d in (document.defs or [])}
    used: set[str] = set()
    for tool in document.tools:
        used |= _referenced_definition_names(tool.fields, defs_by_name)
    return used


def _value_category(value: Any) -> str:
    """Coarse base-type category of an enum member value (§5.6 / §7.2).

    `integer` and `number` collapse to one "numeric" category because the spec
    treats a mix of ints and floats as a single number enum (`-0.5|0|0.5`, §5.6)
    — that is homogeneous, not a violation. bool is checked before int because
    in Python `bool` is a subclass of `int` and True/False are boolean members.
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "numeric"
    if isinstance(value, str):
        return "string"
    return "other"


class _Validator:
    """Single-pass, read-only walk that accumulates errors and warnings.

    Holds the set of definition names (gathered up front so a reference can be
    resolved no matter where it appears, §5.7). No method writes to the tree.
    """

    def __init__(self, document: Document) -> None:
        self._document = document
        self._definition_names = {d.name for d in (document.defs or [])}
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationWarning] = []

    # --- entry point -------------------------------------------------------

    def run(self) -> tuple[list[ValidationError], list[ValidationWarning]]:
        self._check_document_cardinality()      # rule 3
        self._check_defs_empty()                # rule 4
        self._check_unused_definitions()        # warning: unused $defs
        self._check_duplicate_definition_names()  # rule 10
        self._check_duplicate_tool_names()        # warning: duplicate tools

        for definition in self._document.defs or []:
            self._check_definition_name(definition)   # rule 5
            self._check_duplicate_field_names(definition.fields, "§4.2")
            for fld in definition.fields:
                self._visit_field(fld)

        for tool in self._document.tools:
            if isinstance(tool, RawSchema):
                continue  # §7.5: whole-tool fallback; no CATS field lines to check
            self._check_duplicate_field_names(tool.fields, "§4.2")
            for fld in tool.fields:
                self._visit_field(fld)

        return self.errors, self.warnings

    # --- document-level rules ---------------------------------------------

    def _check_document_cardinality(self) -> None:
        """Rule 3 (§3.1): a document MUST contain at least one tool block."""
        if not self._document.tools:
            self.errors.append(
                ValidationError(
                    "document contains no tool block; at least one is required",
                    "§3.1",
                )
            )

    def _check_defs_empty(self) -> None:
        """Rule 4 (§3.2): a present-but-empty $defs block is ill-formed.

        `Document.defs` is `None` when no `$defs` block was written and `[]`
        when the header appeared with zero definitions — only the latter errors.
        """
        if self._document.defs == []:
            self.errors.append(
                ValidationError(
                    "$defs block is present but contains no definitions",
                    "§3.2",
                )
            )

    def _check_unused_definitions(self) -> None:
        """Warning (§3.2): a $defs definition never referenced by any tool.

        Mirror of rule 2 (§5.7): rule 2 flags referenced-but-not-defined; this
        flags defined-but-never-referenced. Document-level: a definition counts
        as used if ANY tool reaches it, directly or transitively through other
        definitions (a def used only as a dependency of a used def is used).
        """
        if not self._document.defs:
            return
        used = _document_used_definition_names(self._document)
        for definition in self._document.defs:
            if definition.name not in used:
                self.warnings.append(
                    ValidationWarning(
                        f"definition {definition.name!r} is never referenced by any tool",
                        "§3.2",
                    )
                )

    # --- definitions -------------------------------------------------------

    def _check_duplicate_field_names(self, fields: list[Field], section: str) -> None:
        """Rule 9 (§4.2): field names must be unique within one block."""
        seen: set[str] = set()
        for fld in fields:
            if fld.name in seen:
                self.errors.append(
                    ValidationError(
                        f"duplicate field name {fld.name!r} within the same block",
                        section,
                    )
                )
            seen.add(fld.name)

    def _check_duplicate_definition_names(self) -> None:
        """Rule 10 (§3.2 / §3.5): definition names must be unique in $defs."""
        if not self._document.defs:
            return
        seen: set[str] = set()
        for definition in self._document.defs:
            if definition.name in seen:
                self.errors.append(
                    ValidationError(
                        f"duplicate definition name {definition.name!r} in $defs",
                        "§3.2",
                    )
                )
            seen.add(definition.name)

    def _check_duplicate_tool_names(self) -> None:
        """Warning (§3.5): tool names SHOULD be unique across the document."""
        seen: set[str] = set()
        for tool in self._document.tools:
            if isinstance(tool, RawSchema):
                continue
            if tool.name in seen:
                self.warnings.append(
                    ValidationWarning(
                        f"duplicate tool name {tool.name!r} in the document",
                        "§3.5",
                    )
                )
            seen.add(tool.name)

    def _check_definition_name(self, definition: Definition) -> None:
        """Rule 5 (§3.2 / §2.3): a definition name must be a bare identifier.

        Unlike tool and field names (which may be hyphenated or quoted), a
        definition name is reached through `$Name` and so is restricted to the
        identifier grammar of §2.3 — no hyphens, no quoted escape hatch.
        """
        if not _IDENTIFIER_RE.match(definition.name):
            self.errors.append(
                ValidationError(
                    f"definition name {definition.name!r} is not a bare identifier "
                    "(no hyphens or quoting; §2.3)",
                    "§3.2",
                )
            )

    # --- fields ------------------------------------------------------------

    def _visit_field(self, fld: Field) -> None:
        """Check a field, then recurse into its type expression."""
        self._check_required_default(fld)       # rule 1
        self._visit_type(fld.type)

    def _check_required_default(self, fld: Field) -> None:
        """Rule 1 (§4.1 / §4.5): a field cannot be both required and defaulted.

        A required field has no value to fall back to, so the combination is
        semantically contradictory. NO_DEFAULT marks 'no default present'; any
        other value (including a literal null/None) is a real default.
        """
        if fld.required and fld.default is not NO_DEFAULT:
            self.errors.append(
                ValidationError(
                    f"field {fld.name!r} is required but also has a default value; "
                    "the two are mutually exclusive",
                    "§4.1",
                )
            )

    # --- type expressions --------------------------------------------------

    def _visit_type(self, node: object) -> None:
        """Recursively validate a type node and everything nested under it."""
        if isinstance(node, Reference):
            self._check_reference(node)          # rule 2

        elif isinstance(node, Array):
            self._check_bounds(
                node.min_items, node.max_items, "array item count", "§6.4"
            )                                    # rule 8
            if node.element is not None:
                self._visit_type(node.element)

        elif isinstance(node, Object):
            self._check_duplicate_field_names(node.fields, "§4.2")
            for fld in node.fields:
                self._visit_field(fld)

        elif isinstance(node, Union):
            self._check_union_homogeneity(node)  # rule 6
            for branch in node.branches:
                self._visit_type(branch)

        elif isinstance(node, Enum):
            self._check_enum_consistency(node)   # rule 7

        elif isinstance(node, (Integer, Number)):
            self._check_bounds(
                node.minimum, node.maximum, "numeric bound", "§6.2"
            )                                    # rule 8

        elif isinstance(node, String):
            self._check_bounds(
                node.min_length, node.max_length, "string length", "§6.3"
            )                                    # rule 8

        # Boolean, Null, AnyType, Const, RawSchema: nothing nested, no bounds.

    def _check_reference(self, ref: Reference) -> None:
        """Rule 2 (§5.7): a reference must resolve to a $defs definition."""
        if ref.name not in self._definition_names:
            self.errors.append(
                ValidationError(
                    f"reference ${ref.name} does not resolve to any definition "
                    "in $defs",
                    "§5.7",
                )
            )

    def _check_union_homogeneity(self, union: Union) -> None:
        """Rule 6 (§5.6): a union's branches must all be types, not values.

        The parser represents a value-literal branch inside a union as a `Const`
        (and a multi-value enum as `Enum`); a well-formed type union contains
        neither. A union carrying any such branch is the illegal mixed form.
        """
        value_branches = [
            b for b in union.branches if isinstance(b, (Const, Enum))
        ]
        if value_branches:
            self.errors.append(
                ValidationError(
                    "union mixes type branches with value literals; a union must "
                    "be wholly types (a value set is an enum, not a union)",
                    "§5.6",
                )
            )

    def _check_enum_consistency(self, enum: Enum) -> None:
        """Rule 7 (§5.6): an enum's members must share one base type.

        Computed directly from the member values (the source of truth) rather
        than trusting the node's pre-inferred `base_type`. `integer` and `number`
        are compatible (a numeric enum may mix ints and floats, §5.6); a string
        beside a number, or a boolean beside a string, is a genuine mix.
        """
        categories = {_value_category(v) for v in enum.values}
        if len(categories) > 1:
            shown = ", ".join(sorted(categories))
            self.errors.append(
                ValidationError(
                    f"enum members do not share one base type (found: {shown})",
                    "§5.6",
                )
            )

    def _check_bounds(
        self,
        lower: object,
        upper: object,
        label: str,
        section: str,
    ) -> None:
        """Rule 8 (§6.2/§6.3/§6.4): when both endpoints exist, lower <= upper.

        Only fires when BOTH bounds are present; an open endpoint (None) imposes
        no ordering. Equal endpoints are allowed (a single-value range).
        """
        if lower is not None and upper is not None and lower > upper:
            self.errors.append(
                ValidationError(
                    f"{label} lower bound {lower} exceeds upper bound {upper}",
                    section,
                )
            )


def validate_with_warnings(
    document: Document,
) -> tuple[list[ValidationError], list[ValidationWarning]]:
    """Validate a CATS `Document` AST and return errors and warnings together.

    Returns `(errors, warnings)` from a single read-only pass. An empty errors
    list means the tree is legal; warnings may still be present (unused $defs).
    """
    return _Validator(document).run()


def validate(document: Document) -> list[ValidationError]:
    """Validate a CATS `Document` AST and return rule violations only.

    Returns an empty list for a legal tree. Warnings are omitted so existing
    callers keep "errors empty => valid". Use validate_with_warnings() when
    legal-but-noteworthy observations (unused definitions) are needed too.
    """
    errors, _warnings = validate_with_warnings(document)
    return errors
