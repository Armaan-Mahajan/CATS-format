"""Condition (b) system prompt builder — JSON Schema in prompt."""

from __future__ import annotations

import copy
import json
from typing import Any

from eval.part1.pipeline import canonical_tool_json
from primer import RequiredUniformity, build_output_contract

_JSON_INTRO = "The following tools are described using JSON Schema."
_JSON_FENCE_LANG = "json"


def required_uniformity_from_tools(tools: list[dict[str, Any]]) -> RequiredUniformity:
    """Classify required/optional across all tool parameter schemas.

    Mirrors ``primer.py``'s ``_required_uniformity``: walks every property in each
    tool's ``parameters`` object (including nested object properties) and checks
    membership in that object's ``required`` array. Zero-parameter tools contribute
    no flags (``all_optional`` when alone).
    """
    flags: list[bool] = []
    for tool in tools:
        flags.extend(_parameter_required_flags(tool.get("parameters") or {}))

    if not flags:
        return "all_optional"
    if all(flags):
        return "all_required"
    if not any(flags):
        return "all_optional"
    return "mixed"


def _parameter_required_flags(parameters: dict[str, Any]) -> list[bool]:
    properties = parameters.get("properties") or {}
    if not properties:
        return []

    required_names = set(parameters.get("required") or [])
    flags: list[bool] = []
    for prop_name, prop_schema in properties.items():
        flags.append(prop_name in required_names)
        if isinstance(prop_schema, dict) and prop_schema.get("type") == "object":
            flags.extend(_parameter_required_flags(prop_schema))
    return flags


def canonical_tools_array_json(tools: list[dict[str, Any]]) -> str:
    """Serialize tools as a compact JSON array matching Part 1's per-tool baseline."""
    canonical_tools = [
        json.loads(canonical_tool_json(copy.deepcopy(tool))) for tool in tools
    ]
    return json.dumps(canonical_tools, separators=(",", ":"))


def build_tools_block(tools: list[dict[str, Any]]) -> str:
    """Fenced JSON array of tool definitions."""
    payload = canonical_tools_array_json(tools)
    return f"```{_JSON_FENCE_LANG}\n{payload}\n```"


def assemble_json_prompt_sections(
    tools: list[dict[str, Any]],
) -> str:
    """Join intro, tool block, and output contract with ``---`` separators."""
    uniformity = required_uniformity_from_tools(tools)
    return "\n\n".join(
        [
            _JSON_INTRO,
            "---",
            build_tools_block(tools),
            "---",
            build_output_contract(uniformity),
        ]
    )


def build_json_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Assemble the full condition (b) system prompt for one or more tools."""
    if not tools:
        raise ValueError("build_json_system_prompt requires at least one tool")
    return assemble_json_prompt_sections(tools)


def output_contract_section(system_prompt: str) -> str:
    """Return the output-contract section after the final ``---`` separator."""
    parts = system_prompt.split("\n---\n")
    if len(parts) != 3:
        raise ValueError(
            f"expected 3 prompt sections separated by ---, got {len(parts)}"
        )
    return parts[2]
