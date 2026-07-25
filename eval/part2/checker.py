"""Thin wrapper around BFCL's AST checker for Part 2 semantic scoring."""

from __future__ import annotations

from enum import Enum
from typing import Any

from bfcl_eval.constants.enums import Language
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

from eval.part2.corpus import CorpusEntry

# Pinned Part 2 API model strings.
PART2_CLAUDE_MODEL = "claude-sonnet-4-6"
PART2_OPENAI_MODEL = "gpt-5.4-2026-03-05"
PART2_QWEN_MODEL = "qwen/qwen3.5-35b-a3b"

PART2_PINNED_MODELS = frozenset(
    {PART2_CLAUDE_MODEL, PART2_OPENAI_MODEL, PART2_QWEN_MODEL}
)

# Prompt-mode stand-in: all 72 non-FC registry entries have underscore_to_dot=False.
# Safe for conditions (a)/(b) regardless of which real model produced the output.
BFCL_PROMPT_MODE_STANDIN = "claude-sonnet-4-5-20250929"

# FC stand-ins for condition (c). Adjacent BFCL registry entries — not bit-identical to
# pinned strings. Closest matches in bfcl-eval==2026.3.23 (no gpt-5.4 or qwen3.5 entry).
BFCL_FC_STANDIN_BY_PART2_MODEL: dict[str, str] = {
    PART2_CLAUDE_MODEL: "claude-sonnet-4-5-20250929-FC",
  # Closest gpt-5.*-FC entry; no gpt-5.4-2026-03-05 in registry.
    PART2_OPENAI_MODEL: "gpt-5.2-2025-12-11-FC",
  # Closest A3B MoE -FC entry; no qwen3.5-35b-a3b entry in registry.
    PART2_QWEN_MODEL: "qwen3-30b-a3b-instruct-2507-FC",
}

# Part B empirical probe results:
# - Claude Sonnet 4.6: HTTP 400 — rejects dotted name in tools[].name at request time
#   (pattern ^[a-zA-Z0-9_-]{1,128}$). Harness must send uber_ride; response key uber_ride.
# - GPT-5.4: HTTP 400 — rejects dotted name (pattern ^[a-zA-Z0-9_-]+$). Same as Claude.
# - Qwen3.5-35B-A3B (Parasail): HTTP 200 — accepts uber.ride; response key "uber.ride".
# When empirical rename behavior contradicts the FC stand-in's registry underscore_to_dot,
# prompt-mode standin is used instead (see bfcl_model_name_for_checker).
NATIVE_RENAMES_DOTS_TO_UNDERSCORES: dict[str, bool] = {
    PART2_CLAUDE_MODEL: True,
    PART2_OPENAI_MODEL: True,
    PART2_QWEN_MODEL: False,
}


class Part2Condition(str, Enum):
    """Which Part 2 eval condition produced the model output being scored."""

    CATS_IN_PROMPT = "cats_in_prompt"
    JSON_IN_PROMPT = "json_in_prompt"
    NATIVE_TOOLS = "native_tools"

    @property
    def is_prompt_mode(self) -> bool:
        return self in {Part2Condition.CATS_IN_PROMPT, Part2Condition.JSON_IN_PROMPT}


def _registry_underscore_to_dot(model_name: str) -> bool:
    key = model_name.replace("_", "/")
    return MODEL_CONFIG_MAPPING[key].underscore_to_dot


def bfcl_model_name_for_checker(
    part2_model: str,
    condition: Part2Condition,
) -> str:
    """Pick the BFCL registry model_name passed to ``ast_checker``."""
    registry_key = part2_model.replace("_", "/")
    if registry_key in MODEL_CONFIG_MAPPING:
        return part2_model

    if condition.is_prompt_mode:
        return BFCL_PROMPT_MODE_STANDIN

    standin = BFCL_FC_STANDIN_BY_PART2_MODEL.get(part2_model)
    if standin is None:
        raise ValueError(f"no FC stand-in configured for Part 2 model {part2_model!r}")

    empirical_renames = NATIVE_RENAMES_DOTS_TO_UNDERSCORES.get(part2_model)
    if empirical_renames is None:
        return standin

    registry_renames = _registry_underscore_to_dot(standin)
    if empirical_renames == registry_renames:
        return standin

    # Empirical request-side behavior contradicts the FC stand-in's registry flag.
    return BFCL_PROMPT_MODE_STANDIN if not empirical_renames else standin


def score_semantic_correctness(
    entry: CorpusEntry,
    model_output: list[dict[str, Any]],
    *,
    part2_model: str,
    condition: Part2Condition,
) -> dict[str, Any]:
    """Score one decoded tool call against BFCL AST ground truth.

    Returns BFCL's checker dict: ``{"valid": bool, "error": list, "error_type": str}``.
    """
    return ast_checker(
        entry.function,
        model_output,
        entry.ground_truth,
        Language.PYTHON,
        entry.category,
        bfcl_model_name_for_checker(part2_model, condition),
    )


def model_output_from_ground_truth(
    ground_truth: list[dict[str, Any]],
    *,
    native_renames_dots: bool = False,
) -> list[dict[str, Any]]:
    """Build a checker-passing model output from BFCL possible-answer form.

    Ground truth wraps each parameter value in a list of acceptable values.
    Optional parameters may use ``""`` as a sentinel meaning "omit".

    When ``native_renames_dots`` is True, function-name keys use underscores instead
    of dots (native API rename behavior under condition (c)).
    """
    decoded: list[dict[str, Any]] = []
    for call in ground_truth:
        if len(call) != 1:
            raise ValueError(f"expected one function per ground-truth call, got {call!r}")
        func_name, param_lists = next(iter(call.items()))
        if native_renames_dots:
            func_name = func_name.replace(".", "_")
        params: dict[str, Any] = {}
        for param, acceptable in param_lists.items():
            if not isinstance(acceptable, list):
                raise ValueError(
                    f"ground truth for {func_name}.{param} is not a list: {acceptable!r}"
                )
            chosen = next((value for value in acceptable if value != ""), None)
            if chosen is not None:
                params[param] = chosen
        decoded.append({func_name: params})
    return decoded
