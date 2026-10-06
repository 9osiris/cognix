"""Tests for the sleep cycle: consolidate, then dream, one event."""

import tempfile

import pytest

from cognix.config import Config
from cognix.runtime import CognitiveRuntime


def make_runtime(seed=None):
    workspace = tempfile.mkdtemp(prefix="cognix_sleep_")
    config = Config.defaults()
    if seed is not None:
        config.set("memory.dream_seed", seed)
    return CognitiveRuntime(config=config, workspace_dir=workspace,
                            now=lambda: 1000.0)


def seed_memories(rt, n=6):
    topics = [
        ("the atlas server is running hot", ("server", "alert")),
        ("the garden irrigation needs a timer", ("garden",)),
        ("mercury-04 runs the staging database", ("server", "db")),
        ("the herons migrate at dawn", ("nature",)),
        ("the cache hit rate dropped overnight", ("cache", "alert")),
        ("sourdough starter needs feeding twice daily", ("food",)),
    ]
    for summary, tags in topics[:n]:
        rt.episodic.record(summary=summary, tags=tags, salience=0.7)


def test_sleep_returns_combined_stats():
    rt = make_runtime()
    seed_memories(rt)
    stats = rt.sleep()
    assert "episodes_added" in stats
    assert "replay" in stats
    dream_stats = stats["dream"]
    assert dream_stats["dreams"] >= 1
    assert dream_stats["insights"] >= 1


def test_sleep_emits_all_three_events():
    rt = make_runtime()
    seed_memories(rt)
    rt.sleep()
    for event_type in ("memory.consolidated", "memory.replayed",
                       "memory.dreamed", "memory.slept"):
        hits = rt.events.recent(event_type)
        assert hits, "missing %s" % event_type
    slept = rt.events.recent("memory.slept")[0]["payload"]
    assert slept["dreams"] >= 1


def test_sleep_records_dream_traces():
    rt = make_runtime()
    seed_memories(rt)
    stats = rt.sleep()
    dreams = [e for e in rt.episodic.recent(30) if "dream" in e.tags]
    assert len(dreams) == stats["dream"]["dreams"] >= 1
    dream_ids = set(stats["dream"]["dream_ids"])
    assert dream_ids == set(e.id for e in dreams)


def test_sleep_on_empty_memory_is_quiet():
    rt = make_runtime()
    stats = rt.sleep()
    assert stats["dream"]["dreams"] == 0
    assert stats["dream"]["insights"] == 0
    slept = rt.events.recent("memory.slept")
    assert slept and slept[0]["payload"]["dreams"] == 0


def test_sleep_consolidates_working_items_first():
    rt = make_runtime()
    report = rt.observe("URGENT: the database is on fire, fix it now")
    assert report["attended"] is True
    before = len(rt.episodic)
    rt.working.rehearse(report["working_id"])
    rt.working.rehearse(report["working_id"])
    stats = rt.sleep()
    assert stats["episodes_added"] >= 1
    assert len(rt.episodic) >= before + 1


def test_sleep_dreams_are_seeded():
    def run_once():
        rt = make_runtime(seed=7)
        seed_memories(rt)
        stats = rt.sleep()
        return [rt.episodic.get(eid).summary for eid in stats["dream"]["dream_ids"]]

    assert run_once() == run_once()
