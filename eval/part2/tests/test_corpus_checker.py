"""Corpus loader and BFCL AST checker sanity tests (Part C)."""

from __future__ import annotations

import pytest
from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

from eval.part2.checker import (
    BFCL_FC_STANDIN_BY_PART2_MODEL,
    BFCL_PROMPT_MODE_STANDIN,
    PART2_CLAUDE_MODEL,
    PART2_OPENAI_MODEL,
    PART2_QWEN_MODEL,
    Part2Condition,
    bfcl_model_name_for_checker,
    model_output_from_ground_truth,
    score_semantic_correctness,
)
from eval.part2.corpus import (
    EXPECTED_ENTRY_COUNTS,
    load_category_entries,
    load_part2_corpus,
)

UBER_RIDE_ENTRY_ID = "live_simple_2-2-0"


def test_category_counts_match_eval_doc():
    corpus = load_part2_corpus()
    by_category: dict[str, int] = {}
    for entry in corpus:
        by_category[entry.category] = by_category.get(entry.category, 0) + 1
    assert by_category == EXPECTED_ENTRY_COUNTS
    assert len(corpus) == sum(EXPECTED_ENTRY_COUNTS.values())


def test_live_multiple_requires_subsample_filter():
    with pytest.raises(ValueError, match="id_filter"):
        load_category_entries("live_multiple")

    live_multiple = [
        entry for entry in load_part2_corpus() if entry.category == "live_multiple"
    ]
    assert len(live_multiple) == 527


@pytest.mark.parametrize("entry_index", [0, 1, 2, 50, 100])
def test_ast_checker_pass_and_fail_on_live_simple(entry_index: int):
    entry = load_category_entries("live_simple")[entry_index]

    correct = model_output_from_ground_truth(entry.ground_truth)
    pass_result = score_semantic_correctness(
        entry,
        correct,
        part2_model=PART2_CLAUDE_MODEL,
        condition=Part2Condition.CATS_IN_PROMPT,
    )
    assert pass_result["valid"] is True
    assert pass_result["error"] == []

    func_name = next(iter(correct[0]))
    params = correct[0][func_name]

    wrong_name = [{"definitely_not_a_real_tool": params}]
    fail_name = score_semantic_correctness(
        entry,
        wrong_name,
        part2_model=PART2_CLAUDE_MODEL,
        condition=Part2Condition.CATS_IN_PROMPT,
    )
    assert fail_name["valid"] is False

    wrong_params = [{func_name: {**params}}]
    first_param = next(iter(params))
    if isinstance(params[first_param], (int, float)):
        wrong_params[0][func_name][first_param] = params[first_param] + 99999
    else:
        wrong_params[0][func_name][first_param] = "__definitely_wrong__"
    fail_value = score_semantic_correctness(
        entry,
        wrong_params,
        part2_model=PART2_CLAUDE_MODEL,
        condition=Part2Condition.CATS_IN_PROMPT,
    )
    assert fail_value["valid"] is False


def test_native_dot_rename_regression_claude_and_openai():
    """Prompt-mode standin falsely fails uber_ride keys under condition (c)."""
    entry = next(
        e
        for e in load_category_entries("live_simple")
        if e.id == UBER_RIDE_ENTRY_ID
    )
    renamed_output = model_output_from_ground_truth(
        entry.ground_truth, native_renames_dots=True
    )
    assert next(iter(renamed_output[0])) == "uber_ride"

    for part2_model in (PART2_CLAUDE_MODEL, PART2_OPENAI_MODEL):
        old_result = ast_checker(
            entry.function,
            renamed_output,
            entry.ground_truth,
            Language.PYTHON,
            entry.category,
            BFCL_PROMPT_MODE_STANDIN,
        )
        assert old_result["valid"] is False
        assert old_result["error_type"] == "simple_function_checker:wrong_func_name"

        fixed_result = score_semantic_correctness(
            entry,
            renamed_output,
            part2_model=part2_model,
            condition=Part2Condition.NATIVE_TOOLS,
        )
        assert fixed_result["valid"] is True
        assert (
            bfcl_model_name_for_checker(part2_model, Part2Condition.NATIVE_TOOLS)
            == BFCL_FC_STANDIN_BY_PART2_MODEL[part2_model]
        )


def test_native_dot_preserve_regression_qwen():
    """FC standin (underscore_to_dot=True) falsely fails uber.ride keys for Qwen."""
    entry = next(
        e
        for e in load_category_entries("live_simple")
        if e.id == UBER_RIDE_ENTRY_ID
    )
    dotted_output = model_output_from_ground_truth(
        entry.ground_truth, native_renames_dots=False
    )
    assert next(iter(dotted_output[0])) == "uber.ride"

    wrong_fc_result = ast_checker(
        entry.function,
        dotted_output,
        entry.ground_truth,
        Language.PYTHON,
        entry.category,
        BFCL_FC_STANDIN_BY_PART2_MODEL[PART2_QWEN_MODEL],
    )
    assert wrong_fc_result["valid"] is False
    assert wrong_fc_result["error_type"] == "simple_function_checker:wrong_func_name"

    fixed_result = score_semantic_correctness(
        entry,
        dotted_output,
        part2_model=PART2_QWEN_MODEL,
        condition=Part2Condition.NATIVE_TOOLS,
    )
    assert fixed_result["valid"] is True
    assert (
        bfcl_model_name_for_checker(PART2_QWEN_MODEL, Part2Condition.NATIVE_TOOLS)
        == BFCL_PROMPT_MODE_STANDIN
    )
