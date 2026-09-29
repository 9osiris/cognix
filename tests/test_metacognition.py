"""Tests for the metacognitive strategy ledger."""

import pytest

from cognix.agent.goals import Goal
from cognix.agent.metacognition import MetaCognition


def make_goal(description):
    return Goal(description=description)


def test_win_rate_defaults_to_half():
    meta = MetaCognition()
    assert meta.strategy_win_rate("decompose") == 0.5
    assert meta.best_strategy() is None


def test_record_outcome_tracks_wins_and_tries():
    meta = MetaCognition()
    meta.record_outcome("reactive", True)
    meta.record_outcome("reactive", False)
    meta.record_outcome("reactive", True)
    assert meta.strategy_win_rate("reactive") == pytest.approx(2 / 3)
    assert meta.ledger()["reactive"] == (2, 3)


def test_best_strategy_picks_highest_win_rate():
    meta = MetaCognition()
    meta.record_outcome("reactive", True)
    meta.record_outcome("decompose", False)
    meta.record_outcome("decompose", False)
    assert meta.best_strategy() == "reactive"


def test_ledger_prefers_proven_winner():
    meta = MetaCognition()
    for _ in range(4):
        meta.record_outcome("reactive", True)
    goal = make_goal("calculate 2 * 3 then write a note called math saying done")
    assert meta.choose_strategy(goal, None) == "reactive"


def test_ledger_ignored_until_enough_evidence():
    meta = MetaCognition()
    meta.record_outcome("reactive", True)
    meta.record_outcome("reactive", True)
    goal = make_goal("calculate 2 * 3 then write a note called math saying done")
    assert meta.choose_strategy(goal, None) == "decompose"


def test_ledger_ignored_when_win_rate_low():
    meta = MetaCognition()
    meta.record_outcome("reactive", True)
    meta.record_outcome("reactive", False)
    meta.record_outcome("reactive", False)
    meta.record_outcome("reactive", False)
    goal = make_goal("calculate 2 * 3 then write a note called math saying done")
    assert meta.choose_strategy(goal, None) == "decompose"


def test_heuristics_still_win_for_target_states():
    meta = MetaCognition()
    for _ in range(5):
        meta.record_outcome("reactive", True)
    goal = make_goal("make sure the backup is complete")
    assert meta.choose_strategy(goal, None) == "means_ends"


def test_failures_still_force_reactive():
    meta = MetaCognition()
    for _ in range(5):
        meta.record_outcome("decompose", True)
    goal = make_goal("calculate 2 * 3 then write a note called math saying done")
    assert meta.choose_strategy(goal, None, past_failures=2) == "reactive"


def test_ledger_persists_across_save_load(tmp_path):
    from cognix.runtime import CognitiveRuntime
    import tempfile
    rt = CognitiveRuntime(workspace_dir=tempfile.mkdtemp())
    goal = rt.add_goal("calculate 2 * 3")
    rt.run_goal(goal.id)
    assert rt.meta.ledger()
    path = str(tmp_path / "state.json")
    rt.save(path)
    rt2 = CognitiveRuntime.load(path, workspace_dir=rt.workspace_dir)
    assert rt2.meta.ledger() == rt.meta.ledger()
    assert rt2.meta.strategy_win_rate("decompose") == rt.meta.strategy_win_rate("decompose")


def test_ledger_serialize_roundtrip():
    from cognix.agent.metacognition import MetaCognition
    m = MetaCognition()
    m.record_outcome("decompose", True)
    m.record_outcome("decompose", False)
    m2 = MetaCognition()
    m2.ledger_from_dict(m.ledger_to_dict())
    assert m2.ledger() == m.ledger()
    assert m2.strategy_win_rate("decompose") == 0.5
