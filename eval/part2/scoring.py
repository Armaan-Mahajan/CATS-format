"""Decode model output and score via BFCL AST checker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval.part2.checker import Part2Condition, score_semantic_correctness
from eval.part2.corpus import CorpusEntry
from eval.part2.output_parser import ParseOutcome, parse_tool_call_response


@dataclass(frozen=True)
class ScoredResult:
    skipped: bool
    skip_reason: str | None
    parse_outcome: ParseOutcome | None
    syntactically_valid: bool
    semantically_valid: bool | None
    checker_result: dict[str, Any] | None
    model_output: list[dict[str, Any]] | None


def _decode_prompt_text(text: str) -> tuple[ParseOutcome, list[dict[str, Any]] | None]:
    parsed = parse_tool_call_response(text)
    if not parsed.syntactically_valid:
        return parsed.outcome, None
    assert parsed.name is not None and parsed.arguments is not None
    return parsed.outcome, [{parsed.name: parsed.arguments}]


def score_extracted_response(
    entry: CorpusEntry,
    *,
    model: str,
    condition: Part2Condition,
    extracted: Any,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> ScoredResult:
    """Score a parsed live provider response."""
    from eval.part2.response_parser import ExtractedModelOutput

    if skipped:
        return ScoredResult(
            skipped=True,
            skip_reason=skip_reason,
            parse_outcome=None,
            syntactically_valid=False,
            semantically_valid=None,
            checker_result=None,
            model_output=None,
        )

    if not isinstance(extracted, ExtractedModelOutput):
        return ScoredResult(
            skipped=False,
            skip_reason=None,
            parse_outcome=None,
            syntactically_valid=False,
            semantically_valid=False,
            checker_result=None,
            model_output=None,
        )

    if extracted.model_output is None:
        return ScoredResult(
            skipped=False,
            skip_reason=None,
            parse_outcome=extracted.parse_outcome,
            syntactically_valid=False,
            semantically_valid=False,
            checker_result=None,
            model_output=None,
        )

    checker_result = score_semantic_correctness(
        entry,
        extracted.model_output,
        part2_model=model,
        condition=condition,
    )
    return ScoredResult(
        skipped=False,
        skip_reason=None,
        parse_outcome=extracted.parse_outcome,
        syntactically_valid=True,
        semantically_valid=bool(checker_result["valid"]),
        checker_result=checker_result,
        model_output=extracted.model_output,
    )
