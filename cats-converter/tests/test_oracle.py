"""Self-tests for the behavioral round-trip oracle (tests/oracle.py).

These prove the ORACLE ITSELF works, before from_json.py is real:

  - a string and an integer schema round-trip through the stub and PASS;
  - a DELIBERATELY BROKEN round trip (drops a constraint, wrongly widening) is
    CAUGHT as non-equivalent — the most important test, since it proves the
    oracle can detect a meaning change rather than rubber-stamp anything;
  - the comparison primitives agree/disagree as expected on hand-built schemas;
  - a RawSchema-only tool's serialized form is carried verbatim and is judged
    behaviorally identical (the §7.5 tool-level fallback path).
"""

from __future__ import annotations

import pytest

import oracle
from nodes import RawSchema
from to_json import to_json


# ---------------------------------------------------------------------------
# The oracle passes when meaning IS preserved (via the real stub round trip)
# ---------------------------------------------------------------------------


class TestOraclePassesOnFaithfulRoundTrip:
    def test_string_schema_round_trips_and_passes(self) -> None:
        # Stub: {"type":"string"} -> String() -> {"type":"string"} (identical).
        oracle.assert_round_trip_preserves_meaning({"type": "string"})

    def test_integer_schema_round_trips_and_passes(self) -> None:
        oracle.assert_round_trip_preserves_meaning({"type": "integer"})

    def test_round_trip_output_equals_stub_serialization(self) -> None:
        assert oracle.round_trip({"type": "string"}) == {"type": "string"}
        assert oracle.round_trip({"type": "integer"}) == {"type": "integer"}


# ---------------------------------------------------------------------------
# THE KEY TEST: a broken round trip must be CAUGHT, not rubber-stamped
# ---------------------------------------------------------------------------


class TestOracleDetectsMeaningChange:
    def test_broken_round_trip_dropping_a_constraint_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Original accepts only strings of length >= 3. The broken trip "loses"
        # the minLength constraint, wrongly WIDENING the schema to all strings.
        schema = {"type": "string", "minLength": 3}

        def broken_trip(_schema: dict) -> dict:
            return {"type": "string"}  # constraint dropped

        monkeypatch.setattr(oracle, "round_trip", broken_trip)

        with pytest.raises(AssertionError) as excinfo:
            oracle.assert_round_trip_preserves_meaning(schema)

        message = str(excinfo.value)
        # The failure must name the changed meaning and surface evidence: a short
        # string the original rejects but the widened round trip accepts.
        assert "meaning" in message.lower()
        assert "disagree" in message.lower()

    def test_disagreements_pinpoints_the_widening_instance(self) -> None:
        narrow = {"type": "string", "minLength": 3}
        widened = {"type": "string"}
        disagreements = oracle.behavioral_disagreements(narrow, widened, ["ab", "abcd"])
        # "ab": rejected by narrow, accepted by widened -> a disagreement.
        # "abcd": accepted by both -> not a disagreement.
        assert [d.instance for d in disagreements] == ["ab"]
        assert disagreements[0].accepted_by_a is False
        assert disagreements[0].accepted_by_b is True

    def test_maxlength_widening_caught_by_probe_pool_alone(self) -> None:
        # Before boundary probes, short strings hid a dropped maxLength. The
        # 100-char probe now disagrees: rejected by maxLength:5, accepted bare.
        narrow = {"type": "string", "maxLength": 5}
        widened = {"type": "string"}
        assert not oracle.behaviorally_equivalent(
            narrow, widened, oracle.DEFAULT_PROBE_INSTANCES
        )


# ---------------------------------------------------------------------------
# The comparison primitives behave as specified
# ---------------------------------------------------------------------------


class TestComparisonPrimitives:
    def test_identical_schemas_are_equivalent(self) -> None:
        schema = {"type": "integer", "minimum": 0}
        assert oracle.behaviorally_equivalent(schema, schema, oracle.DEFAULT_PROBE_INSTANCES)

    def test_keyword_reordering_is_still_equivalent(self) -> None:
        # Same meaning, different key order — must NOT be flagged (behavioral,
        # not textual). Python dict order differs but acceptance is identical.
        a = {"type": "integer", "minimum": 0, "maximum": 10}
        b = {"maximum": 10, "minimum": 0, "type": "integer"}
        assert oracle.behaviorally_equivalent(a, b, oracle.DEFAULT_PROBE_INSTANCES)

    def test_different_types_are_not_equivalent(self) -> None:
        assert not oracle.behaviorally_equivalent(
            {"type": "string"}, {"type": "integer"}, oracle.DEFAULT_PROBE_INSTANCES
        )


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


class TestInstanceGeneration:
    def test_generated_instances_are_valid_against_the_schema(self) -> None:
        from jsonschema import Draft202012Validator

        schema = {"type": "integer", "minimum": 0, "maximum": 5}
        validator = Draft202012Validator(schema)
        result = oracle.generate_instances(schema, max_examples=25)
        assert result.instances, "expected at least one generated instance for a roomy schema"
        assert result.valid_count == len(result.instances)
        assert not result.thin
        assert all(validator.is_valid(i) for i in result.instances)

    def test_tight_schema_degrades_gracefully(self) -> None:
        # Generation must not raise even when Hypothesis struggles; `thin` tracks
        # valid_count against THIN_GENERATION_THRESHOLD regardless of outcome.
        tight = {"type": "string", "pattern": "^x{200}$"}
        result = oracle.generate_instances(tight, max_examples=25)
        assert result.valid_count == len(result.instances)
        assert result.thin == (result.valid_count < oracle.THIN_GENERATION_THRESHOLD)

    def test_thin_flag_follows_valid_count(self) -> None:
        assert oracle.GenerationResult([], 0, True).thin
        assert oracle.GenerationResult(["a"], 1, True).thin
        assert not oracle.GenerationResult(["a"] * 5, 5, False).thin


class TestSamplingConfidence:
    def test_well_evidenced_pass_reports_counts_and_not_low_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the behavioral path even though a faithful integer round trip is
        # structurally identical to its reference.
        monkeypatch.setattr(oracle, "schemas_structurally_equal", lambda _a, _b: False)
        report = oracle.assert_round_trip_preserves_meaning({"type": "integer"})
        assert not report.low_confidence
        assert report.generated_valid_count >= oracle.THIN_GENERATION_THRESHOLD
        assert report.sampled_count >= report.probe_count + report.generated_valid_count
        assert "well-evidenced" in report.note

    def test_thin_generation_yields_low_confidence_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def thin_generation(
            _schema: dict, max_examples: int = 50
        ) -> oracle.GenerationResult:
            return oracle.GenerationResult(instances=["ok"], valid_count=1, thin=True)

        monkeypatch.setattr(oracle, "schemas_structurally_equal", lambda _a, _b: False)
        monkeypatch.setattr(oracle, "generate_instances", thin_generation)
        report = oracle.assert_round_trip_preserves_meaning({"type": "string"})
        assert report.low_confidence
        assert report.generated_valid_count == 1
        assert "LOW CONFIDENCE" in report.note
        assert report.sampled_count < 50  # distinguishable from a full pass

    def test_structurally_identical_round_trip_skips_sampling(self) -> None:
        probe = {"type": "integer", "minimum": 1, "exclusiveMaximum": 3}
        report = oracle.assert_round_trip_preserves_meaning(probe)
        assert report.sampled_count == 0
        assert report.generated_valid_count == 0
        assert not report.low_confidence
        assert "structurally identical" in report.note


class TestMultiToolEnvelope:
    """§7.2.1 list envelopes are compared tool-by-tool, not as a list schema."""

    TWO_TOOL_SHARED_DEFS = [
        {
            "name": "use_foo",
            "type": "object",
            "$defs": {
                "Foo": {
                    "type": "object",
                    "properties": {"s": {"type": "string"}},
                    "additionalProperties": False,
                },
                "Bar": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "additionalProperties": False,
                },
            },
            "properties": {"x": {"$ref": "#/$defs/Foo"}},
            "additionalProperties": False,
        },
        {
            "name": "use_bar",
            "type": "object",
            "$defs": {
                "Foo": {
                    "type": "object",
                    "properties": {"s": {"type": "string"}},
                    "additionalProperties": False,
                },
                "Bar": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "additionalProperties": False,
                },
            },
            "properties": {"y": {"$ref": "#/$defs/Bar"}},
            "additionalProperties": False,
        },
    ]

    def test_two_tool_list_with_pruned_defs_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from converter_demo import pipeline_round_trip as _pipeline_round_trip

        monkeypatch.setattr(oracle, "round_trip", _pipeline_round_trip)
        report = oracle.assert_round_trip_preserves_meaning(
            self.TWO_TOOL_SHARED_DEFS
        )
        assert "multi-tool envelope (2 tools)" in report.note
        assert not oracle.schemas_structurally_equal(
            oracle._semantic_baseline(self.TWO_TOOL_SHARED_DEFS),  # noqa: SLF001
            _pipeline_round_trip(self.TWO_TOOL_SHARED_DEFS),
        )

    def test_two_tool_list_catches_per_tool_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        strict = {
            "name": "strict",
            "type": "object",
            "properties": {"x": {"type": "string", "minLength": 3}},
            "required": ["x"],
            "additionalProperties": False,
        }
        loose = {
            "name": "loose",
            "type": "object",
            "properties": {"y": {"type": "integer"}},
            "required": ["y"],
            "additionalProperties": False,
        }
        envelope = [strict, loose]

        def broken_trip(_schema: dict) -> list[dict]:
            widened = dict(strict)
            widened["properties"] = {"x": {"type": "string"}}
            return [widened, loose]

        monkeypatch.setattr(oracle, "round_trip", broken_trip)

        with pytest.raises(AssertionError) as excinfo:
            oracle.assert_round_trip_preserves_meaning(
                envelope, extra_instances=[{"x": "ab"}]
            )

        message = str(excinfo.value)
        assert "strict" in message
        assert "meaning" in message.lower()


class TestAlignmentLayer:
    """Shape alignment: dict vs list-of-one, public API, name-based pairing."""

    TOOL_WITH_DEFS = {
        "name": "t",
        "type": "object",
        "$defs": {
            "A": {
                "type": "object",
                "properties": {"s": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "properties": {"x": {"$ref": "#/$defs/A"}},
        "additionalProperties": False,
    }

    def test_dict_tool_with_defs_aligns_with_list_round_trip(self) -> None:
        tripped = oracle.round_trip(self.TOOL_WITH_DEFS)
        assert isinstance(tripped, list) and len(tripped) == 1
        pairs = oracle.align_comparison_pairs(self.TOOL_WITH_DEFS, tripped)
        assert len(pairs) == 1
        oracle.assert_round_trip_preserves_meaning(self.TOOL_WITH_DEFS)

    def test_public_api_round_trip_via_alignment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import copy

        import cats

        def public_trip(schema: object) -> object:
            text = cats.convert(copy.deepcopy(schema))
            return cats.to_json_schema(text)

        monkeypatch.setattr(oracle, "round_trip", public_trip)
        report = oracle.assert_round_trip_preserves_meaning(self.TOOL_WITH_DEFS)
        assert "structurally identical" in report.note or report.sampled_count > 0

    def test_named_tools_paired_by_name_not_index(self) -> None:
        strict = {
            "name": "strict",
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        loose = {
            "name": "loose",
            "type": "object",
            "properties": {"y": {"type": "integer"}},
            "additionalProperties": False,
        }
        reference = [strict, loose]
        tripped = [loose, strict]
        pairs = oracle.align_comparison_pairs(reference, tripped)
        assert [pair.label for pair in pairs] == ["strict", "loose"]
        assert pairs[0].baseline["name"] == "strict"
        assert pairs[0].tripped["name"] == "strict"


class TestNumericProbeRegression:
    """Regression probes for integer/number paths (bounds, format, or both)."""

    @pytest.fixture()
    def pipeline_round_trip(self, monkeypatch: pytest.MonkeyPatch):
        from test_to_cats import _pipeline_round_trip

        monkeypatch.setattr(oracle, "round_trip", _pipeline_round_trip)
        return _pipeline_round_trip

    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "integer", "minimum": 1, "exclusiveMaximum": 3},
            {"type": "integer", "format": "int64"},
            {
                "type": "integer",
                "format": "int64",
                "minimum": 1,
                "maximum": 3,
            },
            {"type": "number", "minimum": 1, "exclusiveMaximum": 3},
        ],
    )
    def test_pipeline_round_trip_preserves_meaning(
        self, schema: dict, pipeline_round_trip
    ) -> None:
        assert oracle.schemas_structurally_equal(schema, pipeline_round_trip(schema))
        report = oracle.assert_round_trip_preserves_meaning(schema)
        assert "structurally identical" in report.note


# ---------------------------------------------------------------------------
# §7.5 tool-level fallback: a RawSchema is carried verbatim and stays equivalent
# ---------------------------------------------------------------------------


class TestRawSchemaFallback:
    def test_rawschema_is_serialized_verbatim_and_behaviorally_identical(self) -> None:
        # A construct CATS cannot encode (`not`) — the kind of thing that, in a
        # real tool, becomes a whole-tool RawSchema fallback (§7.5). Serializing
        # the RawSchema must reproduce the input verbatim, hence be behaviorally
        # identical. (from_json does not yet BUILD RawSchema, so we exercise the
        # serializer side directly; the full from_json path is tested later.)
        raw = {"not": {"type": "string"}}
        tripped = to_json(RawSchema(schema=raw))

        assert tripped == raw  # carried verbatim
        assert oracle.behaviorally_equivalent(
            raw, tripped, list(oracle.DEFAULT_PROBE_INSTANCES)
        )
