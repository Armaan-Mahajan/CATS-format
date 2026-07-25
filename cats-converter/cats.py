"""
cats — public API for the CATS converter library.

JSON Schema (draft 2020-12) and CATS text convert through a shared AST; this
module exposes the stable entry points for applications and the eval harness.

The public conversion API operates on **tool definitions** — a single tool schema
(a dict with a top-level ``name`` field) or a list of such schemas (§7.2.1) —
not on standalone JSON Schema type fragments such as ``{"type": "integer"}``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from from_json import (
    AssumedClosedRecord,
    FallbackRecord,
    PythonTypeRenameReport,
    _prepare_tool_schema,
    from_json_with_report,
    load_schema,
)
from nodes import Document, Node, RawSchema, ToolBlock
from parser import parse_text
from to_cats import to_cats
from to_json import to_json
from validate import (
    ValidationError,
    ValidationWarning,
    validate as _validate_document,
    validate_with_warnings,
)

__all__ = [
    "convert",
    "convert_for_tool_calling",
    "convert_with_report",
    "convert_with_report_for_tool_calling",
    "to_json_schema",
    "validate",
    "validate_with_warnings",
    "AssumedClosedRecord",
    "ConversionResult",
    "FallbackRecord",
    "PythonTypeRenameReport",
    "ValidationError",
    "ValidationWarning",
]


@dataclass
class ConversionResult:
    """Outcome of ``convert_with_report``: CATS text plus fallback and warning metadata."""

    cats_text: str
    fallbacks: list[FallbackRecord]
    warnings: list[ValidationWarning]
    assumed_closed: list[AssumedClosedRecord] = field(default_factory=list)
    python_type_renames: PythonTypeRenameReport = field(
        default_factory=PythonTypeRenameReport
    )

    @property
    def fallback_count(self) -> int:
        return len(self.fallbacks)


def _as_document(ast: Node) -> Document | None:
    """Return a ``Document`` when the AST can be validated as a CATS document."""
    if isinstance(ast, Document):
        return ast
    if isinstance(ast, (ToolBlock, RawSchema)):
        return Document(tools=[ast])
    return None


def _is_validatable(document: Document) -> bool:
    """True when every tool is a CATS-encoded block (not a §7.5 RawSchema fallback)."""
    return all(isinstance(tool, ToolBlock) for tool in document.tools)


def _require_tool_definitions(schema: dict[str, Any] | list[Any]) -> None:
    """Reject standalone type fragments; the public API converts tool definitions."""
    if isinstance(schema, list):
        if not schema:
            raise ValueError(
                "cats.convert expects at least one tool definition in the envelope list"
            )
        return
    if not isinstance(schema, dict):
        raise ValueError(
            "cats.convert expects a tool-definition dict or a list of tool schemas"
        )
    prepared, _conflict = _prepare_tool_schema(schema)
    if "name" not in prepared:
        raise ValueError(
            "cats.convert operates on tool definitions (a schema with a top-level "
            "'name' field or a list of such schemas), not standalone type fragments "
            "such as {\"type\": \"integer\", ...}"
        )


def _source_tool_schemas(schema: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """Original JSON tool dicts parallel to ``Document.tools`` declaration order."""
    if isinstance(schema, list):
        return [copy.deepcopy(item) for item in schema if isinstance(item, dict)]
    if isinstance(schema, dict):
        prepared, _conflict = _prepare_tool_schema(schema)
        return [copy.deepcopy(prepared)]
    return []


def _validation_fallback_reason(errors: list[ValidationError]) -> str:
    if not errors:
        return "semantic validation failed"
    first = errors[0]
    return f"semantic validation failed: {first.message} [{first.section}]"


def finalize_ast_for_emission(
    ast: Node,
    source_schema: dict[str, Any] | list[Any],
    fallbacks: list[FallbackRecord],
    warnings: list[ValidationWarning],
) -> Node:
    """Apply semantic validation; fall back invalid CATS tools to raw JSON (§7.5).

  Tools that fail ``validate()`` are replaced with ``RawSchema`` carrying the
  original input JSON verbatim. Each validation error is surfaced as a warning
  naming the affected tool. Document-level validation warnings (e.g. unused
  ``$defs``) are appended without forcing fallback.
    """
    document = _as_document(ast)
    if document is None:
        return ast

    source_tools = _source_tool_schemas(source_schema)
    doc_errors, doc_warnings = validate_with_warnings(document)
    warnings.extend(doc_warnings)

    if not _is_validatable(document):
        return ast

    # Document-level errors (no tool blocks, empty $defs, …) — warn only.
    for err in doc_errors:
        warnings.append(
            ValidationWarning(f"document: {err.message}", err.section)
        )

    new_tools: list[Node] = []
    for index, tool in enumerate(document.tools):
        if isinstance(tool, RawSchema):
            new_tools.append(tool)
            continue

        # Errors only — document-scoped warnings (unused $defs, duplicate tool
        # names) were collected once above; re-running them per mini-doc would
        # falsely flag defs used by sibling tools as unused.
        mini_doc = Document(tools=[tool], defs=document.defs)
        tool_errors = _validate_document(mini_doc)

        if tool_errors:
            raw = (
                source_tools[index]
                if index < len(source_tools)
                else {"name": tool.name, "type": "object", "properties": {}}
            )
            new_tools.append(RawSchema(schema=copy.deepcopy(raw)))
            fallbacks.append(
                FallbackRecord(
                    tool.name,
                    _validation_fallback_reason(tool_errors),
                )
            )
            for err in tool_errors:
                warnings.append(
                    ValidationWarning(
                        f"tool {tool.name!r}: {err.message}",
                        err.section,
                    )
                )
        else:
            new_tools.append(tool)

    return Document(tools=new_tools, defs=document.defs)


def convert_with_report_for_tool_calling(
    schema: dict[str, Any] | list[Any],
) -> ConversionResult:
    """Like :func:`convert_with_report` with Part 1 eval normalization flags.

    Same ``assume_closed=True`` and ``map_python_types=True`` used by the eval
    pipeline, ``generate_primer_from_json``, and the preview UI.
    """
    return convert_with_report(
        schema,
        assume_closed=True,
        map_python_types=True,
    )


def convert_for_tool_calling(schema: dict[str, Any] | list[Any]) -> str:
    """Convert with Part 1 eval flags; returns CATS text only."""
    return convert_with_report_for_tool_calling(schema).cats_text


def convert(
    schema: dict[str, Any] | list[Any],
    *,
    assume_closed: bool = False,
    map_python_types: bool = False,
) -> str:
    """Convert a JSON Schema tool definition to CATS text.

    Expects a tool-definition dict (with a top-level ``name``) or a list of
    such schemas — not a standalone type fragment.

    Example::

        cats.convert({"name": "echo", "type": "object", "properties": {"x": {"type": "string"}}, "additionalProperties": False})
    """
    return convert_with_report(
        schema,
        assume_closed=assume_closed,
        map_python_types=map_python_types,
    ).cats_text


def convert_with_report(
    schema: dict[str, Any] | list[Any],
    *,
    assume_closed: bool = False,
    map_python_types: bool = False,
) -> ConversionResult:
    """Like :func:`convert`, but also returns per-tool fallbacks and validation warnings.

    Example::

        cats.convert_with_report({"name": "t", "type": "object", "properties": {"q": {"not": {}}}, "additionalProperties": False})
    """
    schema = load_schema(schema)
    _require_tool_definitions(schema)
    report = from_json_with_report(
        schema,
        assume_closed=assume_closed,
        map_python_types=map_python_types,
    )
    fallbacks = list(report.fallbacks)
    warnings: list[ValidationWarning] = []
    ast = finalize_ast_for_emission(
        report.ast, schema, fallbacks, warnings
    )
    return ConversionResult(
        cats_text=to_cats(ast),
        fallbacks=fallbacks,
        warnings=warnings,
        assumed_closed=list(report.assumed_closed),
        python_type_renames=report.python_type_renames,
    )


def to_json_schema(cats_text: str) -> dict[str, Any] | list[Any]:
    """Parse CATS text and serialize it to JSON Schema (draft 2020-12).

    Example::

        cats.to_json_schema("echo\\n  x string")
    """
    document = parse_text(cats_text)
    _validate_document(document)
    return to_json(document)


def validate(document: Document) -> list[ValidationError]:
    """Validate a CATS ``Document`` AST and return rule violations only."""
    return _validate_document(document)
