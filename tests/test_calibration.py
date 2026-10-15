"""Tests for belief confidence calibration."""

import pytest

from cognix.perception.beliefs import BeliefStore
from cognix.perception.calibrate import CalibrationLedger
from cognix.runtime import CognitiveRuntime
from cognix.config import Config


def make_ledger(**kwargs):
    return CalibrationLedger(**kwargs)


def test_no_outcomes_passes_prediction_through():
    ledger = make_ledger()
    assert ledger.adjusted("the sky is green", 0.9) == 0.9


def test_too_few_samples_passes_through():
    ledger = make_ledger(min_samples=5)
    for _ in range(4):
        ledger.record_outcome("the cache is warm", 0.9, False)
    assert ledger.adjusted("the cache is warm", 0.9) == 0.9


def test_chronic_overconfidence_dampens_toward_hit_rate():
    ledger = make_ledger(min_samples=5, strength=0.5)
    for _ in range(6):
        ledger.record_outcome("fridays break staging", 0.9, False)
    # hit rate 0.0, weight 6/10=0.6: 0.9 - 0.5*0.6*0.9 = 0.63
    assert ledger.adjusted("fridays break staging", 0.9) == pytest.approx(0.63)


def test_partial_hit_rate_pulls_partway():
    ledger = make_ledger(min_samples=5, strength=1.0)
    for i in range(10):
        ledger.record_outcome("deploys are safe", 0.9, i % 2 == 0)
    # hit rate 0.5, weight 1.0: 0.9 - 1.0*1.0*0.4 = 0.5
    assert ledger.adjusted("deploys are safe", 0.9) == pytest.approx(0.5)


def test_predictions_below_hit_rate_untouched():
    ledger = make_ledger(min_samples=5)
    for _ in range(6):
        ledger.record_outcome("the build passes", 0.9, True)
    assert ledger.adjusted("the build passes", 0.6) == 0.6


def test_adjusted_result_clamped():
    ledger = make_ledger(min_samples=1, strength=1.0)
    ledger.record_outcome("x is y", 1.0, False)
    out = ledger.adjusted("x is y", 1.0)
    assert 0.0 <= out <= 1.0


def test_hit_rate_none_without_records():
    ledger = make_ledger()
    assert ledger.hit_rate("nothing asserted") is None
    ledger.record_outcome("the build passes", 0.7, True)
    assert ledger.hit_rate("the build passes") == 1.0


def test_stats_brier_and_bins():
    ledger = make_ledger(min_samples=1)
    ledger.record_outcome("a is b", 0.8, True)
    ledger.record_outcome("c is d", 0.8, False)
    stats = ledger.stats()
    assert stats["outcomes"] == 2
    assert stats["propositions"] == 2
    # brier: ((0.8-1)^2 + (0.8-0)^2) / 2 = (0.04 + 0.64) / 2
    assert stats["brier"] == pytest.approx(0.34)
    binned = [r for r in stats["bins"] if r["range"] == "0.8-1.0"]
    assert len(binned) == 1
    assert binned[0]["n"] == 2
    assert binned[0]["hit_rate"] == pytest.approx(0.5)


def test_ledger_roundtrip():
    ledger = make_ledger(min_samples=3, strength=0.7)
    ledger.record_outcome("a is b", 0.8, True)
    ledger.adjusted("a is b", 0.9)
    clone = CalibrationLedger.from_dict(ledger.to_dict())
    assert clone.min_samples == 3
    assert clone.strength == pytest.approx(0.7)
    assert clone.hit_rate("a is b") == 1.0
    assert clone.stats()["dampened"] == ledger.stats()["dampened"]


def test_store_dampens_new_belief_after_failures():
    store = BeliefStore(now=lambda: 0.0)
    for _ in range(6):
        store.record_outcome("fridays break staging", False, predicted=0.9)
    belief = store.assert_belief("fridays break staging", 0.9, "test")
    assert belief.confidence < 0.9


def test_store_leaves_untested_belief_alone():
    store = BeliefStore(now=lambda: 0.0)
    belief = store.assert_belief("the sky is blue", 0.85, "test")
    assert belief.confidence == 0.85


def test_store_record_outcome_defaults_to_belief_confidence():
    store = BeliefStore(now=lambda: 0.0)
    store.assert_belief("the cache is warm", 0.7, "test")
    assert store.record_outcome("the cache is warm", True) == 1
    assert store.calibration.hit_rate("the cache is warm") == 1.0


def test_store_record_outcome_fails_with_no_belief_and_no_predicted():
    store = BeliefStore(now=lambda: 0.0)
    assert store.record_outcome("never asserted", True) is False


def test_store_emits_calibrated_event():
    store = BeliefStore(now=lambda: 0.0)
    seen = []
    store._emit = lambda etype, **kw: seen.append((etype, kw))
    for _ in range(6):
        store.record_outcome("fridays break staging", False, predicted=0.9)
    store.assert_belief("fridays break staging", 0.9, "test")
    kinds = [e[0] for e in seen]
    assert "belief.calibrated" in kinds
    calibrated = [e for e in seen if e[0] == "belief.calibrated"][0]
    assert calibrated[1]["raw"] == 0.9
    assert calibrated[1]["calibrated"] < 0.9


def test_store_disabled_calibration():
    store = BeliefStore(now=lambda: 0.0)
    store.calibration_enabled = False
    for _ in range(6):
        store.record_outcome("fridays break staging", False, predicted=0.9)
    belief = store.assert_belief("fridays break staging", 0.9, "test")
    assert belief.confidence == 0.9


def test_store_persistence_keeps_calibration():
    store = BeliefStore(now=lambda: 0.0)
    for _ in range(6):
        store.record_outcome("fridays break staging", False, predicted=0.9)
    clone = BeliefStore.from_dict(store.to_dict(), now=lambda: 0.0)
    assert clone.calibration.stats()["outcomes"] == 6
    assert clone.calibration_enabled is True
    belief = clone.assert_belief("fridays break staging", 0.9, "test")
    assert belief.confidence < 0.9


def test_runtime_wires_config_into_beliefs():
    config = Config.defaults().merge({"beliefs": {"calibration_strength": 0.2}})
    rt = CognitiveRuntime(config=config, workspace_dir="/tmp")
    assert rt.beliefs.calibration.strength == pytest.approx(0.2)
    assert rt.beliefs.calibration_enabled is True


def test_runtime_record_belief_outcome():
    rt = CognitiveRuntime(config=Config.defaults(), workspace_dir="/tmp")
    rt.beliefs.assert_belief("the cache is warm", 0.8, "test")
    assert rt.record_belief_outcome("the cache is warm", True) == 1
    assert rt.beliefs.calibration.hit_rate("the cache is warm") == 1.0


def test_runtime_save_load_keeps_calibration(tmp_path):
    rt = CognitiveRuntime(config=Config.defaults(), workspace_dir="/tmp")
    for _ in range(6):
        rt.record_belief_outcome("fridays break staging", False, predicted=0.9)
    path = str(tmp_path / "state.json")
    rt.save(path)
    rt2 = CognitiveRuntime.load(path, workspace_dir="/tmp")
    assert rt2.beliefs.calibration.stats()["outcomes"] == 6
    belief = rt2.beliefs.assert_belief("fridays break staging", 0.9, "test")
    assert belief.confidence < 0.9
