"""
oracle.py — the BEHAVIORAL round-trip oracle for the CATS converter.

This is TEST INFRASTRUCTURE, not a test file (pytest will not collect it). It
is the instrument we use to develop from_json.py incrementally: it verifies that
the round trip

    JSON Schema  --[from_json]-->  CATS AST  --[to_json]-->  JSON Schema

preserves MEANING — i.e. the output schema accepts/rejects exactly the same JSON
values as the input schema.

WHY BEHAVIORAL, NOT TEXTUAL (spec §1.2 / §7.4)
----------------------------------------------
The round trip is ALLOWED to rewrite the schema: it canonicalizes oneOf->anyOf,
reorders keywords, closes open objects, normalizes variant spellings. All of
those change the TEXT while preserving the SET OF ACCEPTED VALUES. So we must
NOT assert `output == input` (byte or structural equality) — that contract is
wrong and would fail on legal canonicalization. Instead we assert that, for a
batch of sample JSON instances, the original schema and the round-tripped schema
AGREE on accept/reject for every single instance.

The batch deliberately mixes:
  - instances GENERATED from the original schema (hypothesis-jsonschema), which
    are valid against it — these catch a round trip that wrongly NARROWS; and
  - a fixed pool of assorted probe values plus any caller-supplied instances,
    many of which the schema REJECTS — these catch a round trip that wrongly
    WIDENS. Testing rejection matters as much as testing acceptance.

PUBLIC SURFACE
--------------
  round_trip(schema)                              -> to_json(from_json(schema))
  behaviorally_equivalent(a, b, instances)        -> bool
  behavioral_disagreements(a, b, instances)       -> list[Disagreement] (debug)
  generate_instances(schema, max_examples=...)    -> GenerationResult
  align_comparison_pairs(reference, tripped)          -> list[ComparisonPair]
  assert_round_trip_preserves_meaning(schema, ...) -> RoundTripReport (pytest entry)

SAMPLING CONFIDENCE
-------------------
Instance-based checks only catch meaning changes on values actually sampled. A
pass on two instances is not the same evidence as a pass on fifty. Generation
and round-trip entry points therefore expose counts and a `thin` / `low_confidence`
flag when fewer than THIN_GENERATION_THRESHOLD valid instances were generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from hypothesis import HealthCheck, given, seed, settings
from hypothesis.errors import HypothesisException, Unsatisfiable
from hypothesis_jsonschema import from_schema
from jsonschema import Draft202012Validator

from from_json import _prepare_tool_schema, from_json
from to_json import to_json

JsonSchema = dict[str, Any]


def _canonicalize_for_comparison(obj: Any) -> Any:
    """Recursively sort dict keys for deep structural comparison."""
    if isinstance(obj, dict):
        return {key: _canonicalize_for_comparison(obj[key]) for key in sorted(obj)}
    if isinstance(obj, list):
        return [_canonicalize_for_comparison(item) for item in obj]
    return obj


def schemas_structurally_equal(schema_a: Any, schema_b: Any) -> bool:
    """True when two schema trees match after canonical key ordering."""
    return _canonicalize_for_comparison(schema_a) == _canonicalize_for_comparison(schema_b)


def _semantic_baseline(schema: Any) -> Any:
    """§7.6-normalized schema used as the behavioral reference for comparison.

    OpenAPI ``nullable`` and legacy ``definitions`` are not draft 2020-12 keywords,
    but the converter rewrites them before encoding. The oracle compares against
    this normalized form so round-trip checks reflect intended semantics.
    """
    if isinstance(schema, dict):
        prepared, _conflict = _prepare_tool_schema(schema)
        return prepared
    if isinstance(schema, list):
        return [_semantic_baseline(item) for item in schema]
    return schema

# Fewer than this many Hypothesis-generated *valid* instances => thin generation.
THIN_GENERATION_THRESHOLD = 5

# Boundary probes that stress length/size limits common in JSON Schema.
_BOUNDARY_LONG_STRING = "x" * 100
_BOUNDARY_LARGE_INTEGER = 9_007_199_254_740_991  # 2**53 - 1
_BOUNDARY_DEEP_ARRAY = [[[[[1]]]]]
_BOUNDARY_WIDE_OBJECT = {f"k{i}": i for i in range(20)}


# ---------------------------------------------------------------------------
# The round trip under test
# ---------------------------------------------------------------------------

def round_trip(json_schema: JsonSchema) -> Any:
    """JSON Schema --[from_json]--> CATS AST --[to_json]--> JSON Schema.

    Returns whatever to_json() produces for the reconstructed AST. For the
    current from_json stub (single type nodes) that is a JSON Schema dict; once
    from_json returns a full `Document`, this becomes the §7.2 list-of-tools
    envelope and the comparison layer will be extended accordingly.
    """
    ast = from_json(json_schema)
    return to_json(ast)


# ---------------------------------------------------------------------------
# Behavioral comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Disagreement:
    """One instance the two schemas judge differently — the oracle's evidence."""

    instance: Any
    accepted_by_a: bool
    accepted_by_b: bool


def behavioral_disagreements(
    schema_a: JsonSchema, schema_b: JsonSchema, instances: Sequence[Any]
) -> list[Disagreement]:
    """Every instance on which `schema_a` and `schema_b` disagree (accept/reject).

    Each instance is validated against both schemas with the draft 2020-12
    validator; an instance lands in the result iff exactly one schema accepts it.
    An empty result means the two schemas are indistinguishable on this batch.
    """
    validator_a = Draft202012Validator(schema_a)
    validator_b = Draft202012Validator(schema_b)
    out: list[Disagreement] = []
    for instance in instances:
        accepted_a = validator_a.is_valid(instance)
        accepted_b = validator_b.is_valid(instance)
        if accepted_a != accepted_b:
            out.append(Disagreement(instance, accepted_a, accepted_b))
    return out


def behaviorally_equivalent(
    schema_a: JsonSchema, schema_b: JsonSchema, instances: Sequence[Any]
) -> bool:
    """True iff the two schemas agree on accept/reject for every instance."""
    return not behavioral_disagreements(schema_a, schema_b, instances)


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

# A fixed pool of assorted JSON values spanning every primitive plus containers.
# For any reasonable schema some of these are accepted and some rejected, so the
# pool exercises BOTH the accept and the reject path without us having to know
# which is which per schema. This is what lets the oracle catch a round trip
# that wrongly WIDENS what is accepted.
DEFAULT_PROBE_INSTANCES: tuple[Any, ...] = (
    "hello", "", "  ", "123", "0",
    0, 1, -1, 42, 1_099_511_627_776,
    1.5, -0.0, 3.14,
    True, False, None,
    [], [1, 2, 3], ["a", "b"],
    {}, {"key": "value"}, {"n": 1},
    [[1], [2]], {"nested": {"x": 1}},
    # Boundary probes: catch common length/size widenings without generation.
    _BOUNDARY_LONG_STRING,
    _BOUNDARY_LARGE_INTEGER,
    _BOUNDARY_DEEP_ARRAY,
    _BOUNDARY_WIDE_OBJECT,
)


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of Hypothesis sampling against one schema."""

    instances: list[Any]
    valid_count: int
    thin: bool  # valid_count < THIN_GENERATION_THRESHOLD


def generate_instances(
    json_schema: JsonSchema, max_examples: int = 50
) -> GenerationResult:
    """Draw up to `max_examples` instances that are VALID against `json_schema`.

    Uses hypothesis-jsonschema's `from_schema` strategy, harvested by running a
    throwaway Hypothesis test whose only job is to append each generated value.
    Returns a `GenerationResult` with the instances, `valid_count`, and `thin`
    (True when fewer than THIN_GENERATION_THRESHOLD valid instances were produced).

    Tight schemas (e.g. a narrow regex `pattern`, or mutually exclusive
    constraints) have few or zero valid instances. Hypothesis may then exhaust
    its attempts and raise `Unsatisfiable`; we swallow that and return whatever
    was collected (possibly zero). The oracle still does real work in that case
    via DEFAULT_PROBE_INSTANCES and any caller-supplied instances — those drive
    the reject-side comparison, which needs no valid examples.
    """
    collected: list[Any] = []

    @seed(0)
    @settings(
        max_examples=max_examples,
        deadline=None,
        # Tight/odd schemas trip health checks (heavy filtering, slowness); we
        # want best-effort sampling here, not a hard failure, so suppress them.
        suppress_health_check=list(HealthCheck),
        database=None,
    )
    @given(from_schema(json_schema))
    def _collect(instance: Any) -> None:
        collected.append(instance)

    try:
        _collect()
    except (Unsatisfiable, HypothesisException):
        # Could not satisfy the schema within the attempt budget; return the
        # partial sample. See docstring — this is expected for tight schemas.
        pass
    count = len(collected)
    return GenerationResult(
        instances=collected,
        valid_count=count,
        thin=count < THIN_GENERATION_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Pytest-friendly entry point
# ---------------------------------------------------------------------------

def _format_failure(
    original: JsonSchema, tripped: Any, disagreements: list[Disagreement]
) -> str:
    lines = [
        "round trip changed the MEANING of the schema "
        f"({len(disagreements)} disagreeing instance(s)):",
        f"  original schema : {original!r}",
        f"  round-tripped   : {tripped!r}",
        "  disagreements (instance -> original / round-tripped):",
    ]
    for d in disagreements:
        orig_verdict = "accept" if d.accepted_by_a else "reject"
        trip_verdict = "accept" if d.accepted_by_b else "reject"
        lines.append(f"    {d.instance!r}: original={orig_verdict}, round-tripped={trip_verdict}")
    return "\n".join(lines)


@dataclass(frozen=True)
class RoundTripReport:
    """Outcome of a successful round-trip behavioral check (no meaning change)."""

    sampled_count: int
    probe_count: int
    generated_valid_count: int
    low_confidence: bool
    note: str


def _confidence_note(
    sampled_count: int,
    probe_count: int,
    generated_valid_count: int,
    low_confidence: bool,
) -> str:
    confidence = "LOW CONFIDENCE (thin generation)" if low_confidence else "well-evidenced"
    return (
        f"round-trip passed on {sampled_count} sampled instance(s) "
        f"({probe_count} probes + {generated_valid_count} generated valid); "
        f"confidence: {confidence}"
    )


@dataclass(frozen=True)
class ComparisonPair:
    """One aligned (reference, round-tripped) schema pair for behavioral check."""

    baseline: JsonSchema
    tripped: JsonSchema
    label: str


def _comparison_units(value: Any) -> list[JsonSchema]:
    """Split a schema value into per-unit dicts for pairwise comparison.

    A lone tool dict and a one-element tool list are treated as the same single
    unit so ``Document`` round trips (list envelope) align with dict inputs.
    """
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        if not value:
            return []
        if not all(isinstance(item, dict) for item in value):
            raise AssertionError(
                "oracle cannot compare list envelope: every item must be a dict"
            )
        if len(value) == 1:
            return [value[0]]
        return list(value)
    raise AssertionError(
        f"oracle cannot compare schema of type {type(value).__name__!r}; "
        "expected a JSON Schema dict or list of dicts"
    )


def _pair_comparison_units(
    ref_units: Sequence[JsonSchema],
    trip_units: Sequence[JsonSchema],
) -> list[tuple[JsonSchema, JsonSchema, str]]:
    """Align reference and round-tripped units by tool ``name`` or by index."""
    if len(ref_units) != len(trip_units):
        raise AssertionError(
            f"round trip changed schema unit count: {len(ref_units)} -> {len(trip_units)}"
        )

    ref_named = bool(ref_units) and all("name" in unit for unit in ref_units)
    trip_named = bool(trip_units) and all("name" in unit for unit in trip_units)

    if ref_named and trip_named:
        trip_by_name = {unit["name"]: unit for unit in trip_units}
        if len(trip_by_name) != len(trip_units):
            raise AssertionError(
                "round trip produced duplicate tool names in the envelope"
            )
        pairs: list[tuple[JsonSchema, JsonSchema, str]] = []
        for ref in ref_units:
            name = ref["name"]
            if name not in trip_by_name:
                raise AssertionError(
                    f"round trip dropped tool {name!r}; "
                    f"got tools {sorted(trip_by_name)}"
                )
            pairs.append((ref, trip_by_name[name], name))
        return pairs

    return [
        (ref, trip, f"#{index}" if len(ref_units) > 1 else "schema")
        for index, (ref, trip) in enumerate(zip(ref_units, trip_units))
    ]


def align_comparison_pairs(reference: Any, tripped: Any) -> list[ComparisonPair]:
    """Normalize shapes, apply §7.6 baseline to both sides, and pair for comparison.

    This is the single alignment layer used by ``assert_round_trip_preserves_meaning``.
    Reference and tripped may differ in envelope shape (dict vs one-element list),
    tool order (paired by ``name`` when every unit is a named tool), or legacy
    keywords (``nullable``, ``definitions``) — both sides receive the same
    ``_semantic_baseline`` pass before validation.
    """
    ref_root = _semantic_baseline(reference)
    ref_units = _comparison_units(ref_root)
    trip_units = _comparison_units(tripped)

    aligned: list[ComparisonPair] = []
    for ref_unit, trip_unit, label in _pair_comparison_units(ref_units, trip_units):
        aligned.append(
            ComparisonPair(
                baseline=ref_unit,
                tripped=_semantic_baseline(trip_unit),
                label=label,
            )
        )
    return aligned


def _aggregate_round_trip_reports(
    reports: Sequence[RoundTripReport], *, tool_count: int
) -> RoundTripReport:
    """Combine per-tool reports from a multi-tool envelope check."""
    sampled_count = sum(report.sampled_count for report in reports)
    probe_count = reports[0].probe_count if reports else 0
    generated_valid_count = sum(report.generated_valid_count for report in reports)
    low_confidence = any(report.low_confidence for report in reports)
    confidence = "LOW CONFIDENCE (thin generation)" if low_confidence else "well-evidenced"
    note = (
        f"multi-tool envelope ({tool_count} tools): "
        f"round-trip passed on {sampled_count} sampled instance(s) across all tools "
        f"({probe_count} probes per tool + {generated_valid_count} generated valid "
        f"total); confidence: {confidence}"
    )
    return RoundTripReport(
        sampled_count=sampled_count,
        probe_count=probe_count,
        generated_valid_count=generated_valid_count,
        low_confidence=low_confidence,
        note=note,
    )


def _check_schema_pair_preserves_meaning(
    baseline: JsonSchema,
    tripped: Any,
    *,
    extra_instances: Sequence[Any] = (),
    max_generated: int = 50,
) -> RoundTripReport:
    """Assert one reference schema and one round-tripped schema agree on meaning."""
    # When the round trip is keyword-for-keyword identical to the reference,
    # the two validators cannot disagree. Skip Hypothesis sampling so numeric
    # (and other) paths cannot false-negative on instance-generation quirks.
    # This does NOT resolve the original un-reproduced "meaning not preserved"
    # numeric false-negative — if that recurs, capture the full error block with
    # the disagreeing instance and both accept/reject verdicts before changing
    # this guard.
    if schemas_structurally_equal(baseline, tripped):
        probe_count = len(DEFAULT_PROBE_INSTANCES) + len(extra_instances)
        note = (
            f"round-trip passed (structurally identical to reference); "
            f"{probe_count} probe(s) available but sampling skipped"
        )
        return RoundTripReport(
            sampled_count=0,
            probe_count=probe_count,
            generated_valid_count=0,
            low_confidence=False,
            note=note,
        )

    probe_count = len(DEFAULT_PROBE_INSTANCES) + len(extra_instances)
    generation = generate_instances(baseline, max_examples=max_generated)

    instances: list[Any] = list(DEFAULT_PROBE_INSTANCES)
    instances.extend(extra_instances)
    instances.extend(generation.instances)

    disagreements = behavioral_disagreements(baseline, tripped, instances)
    if disagreements:
        raise AssertionError(_format_failure(baseline, tripped, disagreements))

    sampled_count = len(instances)
    low_confidence = generation.thin
    note = _confidence_note(
        sampled_count, probe_count, generation.valid_count, low_confidence
    )
    return RoundTripReport(
        sampled_count=sampled_count,
        probe_count=probe_count,
        generated_valid_count=generation.valid_count,
        low_confidence=low_confidence,
        note=note,
    )


def assert_round_trip_preserves_meaning(
    json_schema: JsonSchema,
    *,
    extra_instances: Sequence[Any] = (),
    max_generated: int = 50,
) -> RoundTripReport:
    """Round-trip `json_schema` and assert it accepts/rejects the same values.

    Builds the comparison batch from three sources — values GENERATED from the
    original schema (valid examples; catch wrongful narrowing), the fixed probe
    pool, and any `extra_instances` the caller adds (e.g. tricky edge cases or
    values that SHOULD be rejected; catch wrongful widening) — then asserts the
    original and round-tripped schemas agree on every one. On failure the
    AssertionError message lists each disagreeing instance and both verdicts.

    When the input is a §7.2.1 multi-tool list envelope, compares each tool
    schema pairwise; the list itself is never handed to the validator or
    Hypothesis as an instance schema. Named tools are matched by ``name``, not
    list index. A single dict tool and a one-element list round trip align
    automatically.

    On success returns a `RoundTripReport` so callers can tell a thin pass
    (few generated valid instances) from a well-evidenced one. `low_confidence`
    is True when `generated_valid_count < THIN_GENERATION_THRESHOLD`; the `note`
    field repeats the counts for corpus logging.

    Calls the module-level `round_trip`, so a test may monkeypatch
    `oracle.round_trip` to inject a deliberately broken trip and confirm the
    oracle catches it.

    Both sides receive §7.6 input normalization (OpenAPI ``nullable``, legacy
    ``definitions``, provider envelopes, etc.) inside ``align_comparison_pairs``.
    """
    tripped = round_trip(json_schema)
    pairs = align_comparison_pairs(json_schema, tripped)

    if not pairs:
        raise AssertionError("round trip produced no comparable schema units")

    reports: list[RoundTripReport] = []
    for pair in pairs:
        try:
            reports.append(
                _check_schema_pair_preserves_meaning(
                    pair.baseline,
                    pair.tripped,
                    extra_instances=extra_instances,
                    max_generated=max_generated,
                )
            )
        except AssertionError as exc:
            if len(pairs) > 1:
                raise AssertionError(
                    f"round trip changed the MEANING of tool {pair.label!r} "
                    f"in a {len(pairs)}-tool envelope:\n{exc}"
                ) from exc
            raise

    if len(pairs) == 1:
        return reports[0]
    return _aggregate_round_trip_reports(reports, tool_count=len(pairs))
