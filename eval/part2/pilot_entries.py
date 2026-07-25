"""Pilot corpus entry selection for the live orchestration run."""

from __future__ import annotations

# Required special-case entries (do not substitute).
PILOT_ENTRY_ALL_FALLBACK = "live_simple_71-35-0"
PILOT_REQUIRED_ENTRIES: tuple[str, ...] = (
    "live_simple_2-2-0",  # dotted tool, no embedded system
    PILOT_ENTRY_ALL_FALLBACK,  # all-fallback (condition a skip)
    "multiple_0",  # multi-tool decoy selection
    "live_simple_66-30-0",  # embedded system + dotted tool (live_simple)
    "live_multiple_33-10-3",  # embedded system + dotted + multi-tool (live_multiple)
)

# Ordinary spread (no dots, no embedded system, no all-fallback).
PILOT_ORDINARY_ENTRIES: tuple[str, ...] = (
    "live_simple_0-0-0",
    "live_simple_1-1-0",
    "live_simple_4-3-0",
    "multiple_9",
    "multiple_21",
    "live_multiple_7-3-2",
    "live_multiple_23-5-0",
    "live_multiple_25-6-0",
)

PILOT_ENTRY_IDS: tuple[str, ...] = PILOT_REQUIRED_ENTRIES + PILOT_ORDINARY_ENTRIES

PILOT_METADATA_TAG = "cats-part2-pilot"
PILOT_RUN_ID_PREFIX = "pilot"
