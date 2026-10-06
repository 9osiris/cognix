"""Tests for offline replay."""

import pytest

from cognix.memory.episodic import EpisodicMemory
from cognix.memory.semantic import SemanticMemory
from cognix.memory.working import WorkingMemory
from cognix.memory.replay import (
    ReplayPolicy,
    _keyword_overlap,
    _tag_overlap,
    replay,
    select_episodes,
)


def make_episodic():
    mem = EpisodicMemory()
    mem.record(summary="the atlas server is running hot",
               tags=("server", "alert"), salience=0.9)
    mem.record(summary="the atlas server needs a reboot",
               tags=("server", "alert"), salience=0.8)
    mem.record(summary="lunch was a sandwich",
               tags=("food",), salience=0.2)
    return mem


def test_policy_roundtrip():
    policy = ReplayPolicy(budget=3, min_salience=0.5)
    clone = ReplayPolicy.from_dict(policy.to_dict())
    assert clone.budget == 3
    assert clone.min_salience == 0.5


def test_select_episodes_prefers_salient_and_recent():
    mem = make_episodic()
    chosen = select_episodes(mem, ReplayPolicy(budget=2, min_salience=0.4))
    assert len(chosen) == 2
    assert all(e.salience >= 0.4 for e in chosen)
    assert chosen[0].salience >= chosen[1].salience


def test_select_episodes_respects_budget():
    mem = make_episodic()
    chosen = select_episodes(mem, ReplayPolicy(budget=1, min_salience=0.0))
    assert len(chosen) == 1


def test_select_episodes_empty_when_nothing_salient():
    mem = make_episodic()
    chosen = select_episodes(mem, ReplayPolicy(min_salience=0.99))
    assert chosen == []


def test_tag_overlap():
    mem = make_episodic()
    eps = mem.recent(10)
    servers = [e for e in eps if "server" in e.tags]
    food = [e for e in eps if "food" in e.tags]
    assert _tag_overlap(servers[0], servers[1]) == 1.0
    assert _tag_overlap(servers[0], food[0]) == 0.0


def test_tag_overlap_ignores_bare_intent_tags():
    mem = EpisodicMemory()
    mem.record(summary="the server is hot", tags=("inform",), salience=0.9)
    mem.record(summary="the garden needs water", tags=("inform",), salience=0.9)
    eps = mem.recent(10)
    assert _tag_overlap(eps[0], eps[1]) == 0.0
    mem.record(summary="the db is slow", tags=("inform", "db"), salience=0.9)
    mem.record(summary="the db needs indexes", tags=("inform", "db"), salience=0.9)
    eps = mem.recent(10)
    assert _tag_overlap(eps[0], eps[1]) == 1.0


def test_replay_does_not_link_on_intent_tag_alone():
    mem = EpisodicMemory()
    mem.record(summary="the server is hot", tags=("inform",), salience=0.9)
    mem.record(summary="the garden needs water", tags=("inform",), salience=0.9)
    sem = SemanticMemory()
    stats = replay(mem, sem, policy=ReplayPolicy(budget=5, min_salience=0.4))
    assert stats["links_added"] == 0


def test_keyword_overlap():
    mem = make_episodic()
    eps = mem.recent(10)
    servers = [e for e in eps if "server" in e.tags]
    assert _keyword_overlap(servers[0], servers[1]) > 0.0


def test_replay_links_related_episodes():
    mem = make_episodic()
    sem = SemanticMemory()
    stats = replay(mem, sem, policy=ReplayPolicy(budget=5, min_salience=0.4))
    assert stats["replayed"] == 2
    assert stats["links_added"] == 1
    assert stats["relations_strengthened"] >= 1


def test_replay_ignores_lone_episodes():
    mem = EpisodicMemory()
    mem.record(summary="a single lonely event", tags=("solo",), salience=0.9)
    sem = SemanticMemory()
    stats = replay(mem, sem)
    assert stats["replayed"] == 1
    assert stats["links_added"] == 0


def test_replay_rehearses_matching_working_items():
    mem = make_episodic()
    sem = SemanticMemory()
    working = WorkingMemory()
    item = working.add("the atlas server status report", kind="inform",
                       activation=0.4)
    before = item.activation
    stats = replay(mem, sem, working, policy=ReplayPolicy(min_salience=0.4))
    assert stats["items_rehearsed"] >= 1
    assert working.get(item.id).activation > before


def test_replay_stats_shape():
    mem = make_episodic()
    stats = replay(mem, SemanticMemory())
    for key in ("replayed", "links_added", "relations_strengthened",
                "items_rehearsed", "pairs_examined"):
        assert key in stats


def test_replay_without_working_memory():
    mem = make_episodic()
    stats = replay(mem, SemanticMemory(), working=None)
    assert stats["replayed"] == 2
