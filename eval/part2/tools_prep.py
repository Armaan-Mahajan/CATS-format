"""Normalize BFCL tools and apply native API name rules."""

from __future__ import annotations

import copy
from typing import Any

from from_json import normalize_map_python_types


def normalize_entry_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Part 1 normalization (``map_python_types``) for each tool."""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        mapped, _report = normalize_map_python_types(copy.deepcopy(tool))
        normalized.append(mapped)
    return normalized


def native_api_tool_name(name: str, *, renames_dots: bool) -> str:
    if renames_dots:
        return name.replace(".", "_")
    return name


def tool_for_native_api(
    tool: dict[str, Any],
    *,
    renames_dots: bool,
) -> dict[str, Any]:
    """Return a copy of ``tool`` with ``name`` adjusted for native tool APIs."""
    out = copy.deepcopy(tool)
    out["name"] = native_api_tool_name(tool["name"], renames_dots=renames_dots)
    return out


def tools_for_native_api(
    tools: list[dict[str, Any]],
    *,
    renames_dots: bool,
) -> list[dict[str, Any]]:
    return [tool_for_native_api(tool, renames_dots=renames_dots) for tool in tools]
