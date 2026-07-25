"""Pinned Part 2 provider and model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from eval.part2.checker import (
    PART2_CLAUDE_MODEL,
    PART2_OPENAI_MODEL,
    PART2_QWEN_MODEL,
    NATIVE_RENAMES_DOTS_TO_UNDERSCORES,
    Part2Condition,
)


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


# Parasail quantization confirmed via GET /api/v1/models/qwen/qwen3.5-35b-a3b/endpoints
# (provider_name=Parasail, quantization=fp8, tag=parasail/fp8).
OPENROUTER_ROUTING = {
    "provider": {
        "order": ["Parasail"],
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
    }
}


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    model: str
    native_renames_dots: bool


MODEL_CONFIGS: dict[str, ModelConfig] = {
    PART2_OPENAI_MODEL: ModelConfig(
        provider=Provider.OPENAI,
        model=PART2_OPENAI_MODEL,
        native_renames_dots=NATIVE_RENAMES_DOTS_TO_UNDERSCORES[PART2_OPENAI_MODEL],
    ),
    PART2_CLAUDE_MODEL: ModelConfig(
        provider=Provider.ANTHROPIC,
        model=PART2_CLAUDE_MODEL,
        native_renames_dots=NATIVE_RENAMES_DOTS_TO_UNDERSCORES[PART2_CLAUDE_MODEL],
    ),
    PART2_QWEN_MODEL: ModelConfig(
        provider=Provider.OPENROUTER,
        model=PART2_QWEN_MODEL,
        native_renames_dots=NATIVE_RENAMES_DOTS_TO_UNDERSCORES[PART2_QWEN_MODEL],
    ),
}


ALL_MODELS = tuple(MODEL_CONFIGS.keys())
ALL_CONDITIONS = tuple(Part2Condition)
