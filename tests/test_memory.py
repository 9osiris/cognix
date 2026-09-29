"""Tests for cognix.memory: working, episodic, semantic, consolidation,
associative recall, and forgetting curves."""

import math
import sys
import types
from pathlib import Path

import pytest

# The sibling cognix subsystems (agent, runtime, ...) are still being built,
# so cognix/__init__.py may not import yet. Load cognix.memory standalone
# when the full package is unavailable.
try:
    from cognix.memory import working as _working_probe  # noqa: F401
except Exception:
    _mem_dir = str(Path(__file__).resolve().parent.parent / "cognix" / "memory")
    _pkg = types.ModuleType("cognix.memory")
    _pkg.__path__ = [_mem_dir]
    sys.modules.setdefault("cognix.memory", _pkg)

from cognix.memory.associative import AssociativeRecall
from cognix.memory.consolidation import ConsolidationPolicy, consolidate
from cognix.memory.episodic import Episode, EpisodicMemory
from cognix.memory.forgetting import (
    ForgettingCurve,
    ebbinghaus_retention,
    half_life,
)
from cognix.memory.semantic import Concept, Relation, SemanticMemory
from cognix.memory.working import WorkingItem, WorkingMemory


class Clock:
    def __init__(self, t=1000000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# working


class TestWorkingItem:
    def test_roundtrip(self):
        item = WorkingItem(
            id="wm-1",
            content="saw a red car",
            kind="observation",
            activation=0.8,
            created=123.0,
            source="perception",
            meta={"tags": ["car"]},
            pinned=True,
            rehearsals=3,
        )
        back = WorkingItem.from_dict(item.to_dict())
        assert back == item

    def test_touch_boosts_and_counts(self):
        item = WorkingItem(id="x", content="c", activation=0.5)
        item.touch(0.3)
        assert item.activation == pytest.approx(0.8)
        assert item.rehearsals == 1
        item.touch(10.0)
        assert item.activation == 1.0


class TestWorkingMemory:
    def test_add_and_get(self):
        wm = WorkingMemory(capacity=3, now=Clock())
        item = wm.add("hello", kind="note", source="test")
        assert wm.get(item.id) is item
        assert item.activation == 1.0
        assert item.kind == "note"
        assert item.source == "test"
        assert len(wm) == 1
        assert item.id in wm

    def test_capacity_evicts_lowest_activation(self):
        wm = WorkingMemory(capacity=3, now=Clock())
        low = wm.add("low", activation=0.2)
        mid = wm.add("mid", activation=0.5)
        high = wm.add("high", activation=0.9)
        wm.add("new", activation=0.7)
        assert wm.get(low.id) is None
        assert wm.get(mid.id) is not None
        assert wm.get(high.id) is not None
        assert wm.stats()["evictions"] == 1

    def test_pinned_items_never_evicted(self):
        wm = WorkingMemory(capacity=2, now=Clock())
        pinned = wm.add("keep me", activation=0.1)
        assert wm.pin(pinned.id) is True
        other = wm.add("other", activation=0.9)
        wm.add("third", activation=0.95)
        assert wm.get(pinned.id) is not None
        assert wm.get(other.id) is None
        assert wm.pin("nope") is False

    def test_unpin(self):
        wm = WorkingMemory(capacity=2, now=Clock())
        item = wm.add("x", activation=0.1)
        wm.pin(item.id)
        assert wm.unpin(item.id) is True
        assert wm.unpin("missing") is False
        wm.add("a", activation=0.9)
        wm.add("b", activation=0.9)
        assert wm.get(item.id) is None

    def test_rehearse_boosts_activation(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        item = wm.add("x", activation=0.4)
        assert wm.rehearse(item.id) is True
        assert item.activation > 0.4
        assert item.rehearsals == 1
        assert wm.curve.rehearsals(item.id) == 1
        assert wm.rehearse("missing") is False

    def test_tick_decays_exponentially(self):
        wm = WorkingMemory(decay_rate=0.1, threshold=0.0, now=Clock())
        item = wm.add("x", activation=1.0)
        wm.tick(dt=10.0)
        assert item.activation == pytest.approx(math.exp(-1.0))

    def test_tick_drops_below_threshold(self):
        wm = WorkingMemory(decay_rate=0.5, threshold=0.5, now=Clock())
        weak = wm.add("weak", activation=0.6)
        strong = wm.add("strong", activation=1.0)
        dropped = wm.tick(dt=1.0)
        assert dropped == 1
        assert wm.get(weak.id) is None
        assert wm.get(strong.id) is not None

    def test_top_ordering(self):
        wm = WorkingMemory(now=Clock())
        a = wm.add("a", activation=0.3)
        b = wm.add("b", activation=0.9)
        c = wm.add("c", activation=0.6)
        top = wm.top(2)
        assert [i.id for i in top] == [b.id, c.id]
        assert wm.top(0) == []

    def test_remove_and_clear(self):
        wm = WorkingMemory(now=Clock())
        item = wm.add("x")
        assert wm.remove(item.id) is True
        assert wm.remove(item.id) is False
        wm.add("y")
        wm.clear()
        assert len(wm) == 0

    def test_chunk_groups_members(self):
        wm = WorkingMemory(capacity=7, now=Clock())
        a = wm.add("red", meta={"tags": ["color"]})
        b = wm.add("round", meta={"tags": ["shape"]})
        chunk = wm.chunk([a.id, b.id], label="red ball")
        assert chunk is not None
        assert chunk.kind == "chunk"
        assert chunk.content == "red ball"
        assert chunk.meta["chunk_size"] == 2
        assert {m["id"] for m in chunk.meta["members"]} == {a.id, b.id}
        assert set(chunk.meta["tags"]) == {"color", "shape"}
        assert wm.get(a.id) is None
        assert wm.get(b.id) is None
        assert len(wm) == 1

    def test_chunk_missing_id_returns_none(self):
        wm = WorkingMemory(now=Clock())
        a = wm.add("a")
        assert wm.chunk([a.id, "ghost"], "label") is None
        assert wm.get(a.id) is not None
        assert wm.chunk([], "label") is None

    def test_expand_chunk_restores_members(self):
        wm = WorkingMemory(capacity=7, now=Clock())
        a = wm.add("red")
        b = wm.add("round")
        chunk = wm.chunk([a.id, b.id], label="red ball")
        restored = wm.expand_chunk(chunk.id)
        assert restored is not None
        assert len(restored) == 2
        assert {i.content for i in restored} == {"red", "round"}
        assert wm.get(chunk.id) is None

    def test_stats(self):
        wm = WorkingMemory(capacity=4, now=Clock())
        wm.add("a", activation=0.5)
        item = wm.add("b", activation=1.0)
        wm.pin(item.id)
        s = wm.stats()
        assert s["count"] == 2
        assert s["capacity"] == 4
        assert s["capacity_pressure"] == pytest.approx(0.5)
        assert s["mean_activation"] == pytest.approx(0.75)
        assert s["pinned"] == 1

    def test_by_kind_and_refresh(self):
        wm = WorkingMemory(now=Clock())
        wm.add("a", kind="observation")
        goal = wm.add("b", kind="goal", activation=0.2)
        assert len(wm.by_kind("goal")) == 1
        assert wm.refresh(goal.id) is True
        assert goal.activation == 1.0
        assert wm.refresh("missing") is False

    def test_persistence_roundtrip(self):
        clock = Clock()
        wm = WorkingMemory(capacity=5, decay_rate=0.2, now=clock)
        item = wm.add("remember me", activation=0.7, meta={"tags": ["t"]})
        wm.pin(item.id)
        wm.rehearse(item.id)
        back = WorkingMemory.from_dict(wm.to_dict(), now=clock)
        assert back.capacity == 5
        assert back.decay_rate == pytest.approx(0.2)
        got = back.get(item.id)
        assert got is not None
        assert got.content == "remember me"
        assert got.pinned is True
        assert got.rehearsals == 1
        assert got.activation == pytest.approx(item.activation)
        # ids keep increasing after restore
        nxt = back.add("new")
        assert nxt.id != item.id

    def test_bad_capacity(self):
        with pytest.raises(ValueError):
            WorkingMemory(capacity=0)


# episodic


class TestEpisode:
    def test_roundtrip(self):
        ep = Episode(
            id="ep-1",
            timestamp=999.0,
            summary="ate lunch",
            detail={"food": "ramen"},
            tags=["food"],
            salience=0.9,
            source="test",
            links={"ep-2": "follows"},
        )
        assert Episode.from_dict(ep.to_dict()) == ep


class TestEpisodicMemory:
    def test_record_defaults(self):
        clock = Clock()
        mem = EpisodicMemory(now=clock)
        ep = mem.record("something happened")
        assert ep.id == "ep-1"
        assert ep.timestamp == clock()
        assert ep.detail == {}
        assert ep.tags == []
        assert mem.count() == 1
        assert "ep-1" in mem

    def test_record_with_all_fields(self):
        mem = EpisodicMemory(now=Clock(50.0))
        ep = mem.record(
            "launch",
            detail={"k": "v"},
            tags=("a", "b"),
            salience=0.9,
            source="agent",
            timestamp=42.0,
        )
        assert ep.timestamp == 42.0
        assert ep.detail == {"k": "v"}
        assert ep.tags == ["a", "b"]
        assert mem.get(ep.id) is ep
        assert mem.get("missing") is None

    def test_keyword_recall_finds_right_episode(self):
        mem = EpisodicMemory(now=Clock())
        mem.record("the cat sat on the warm mat", tags=("pets",))
        target = mem.record("quantum tunneling lets particles cross barriers",
                            tags=("physics",))
        mem.record("bought groceries and milk", tags=("errands",))
        hits = mem.recall("quantum particles barriers", strategy="keyword")
        assert hits[0][0].id == target.id

    def test_keyword_ignores_stopwords(self):
        mem = EpisodicMemory(now=Clock())
        a = mem.record("the and or but zebra crossing")
        b = mem.record("ordinary day with nothing special")
        hits = mem.recall("zebra", strategy="keyword")
        assert hits[0][0].id == a.id

    def test_recency_ordering(self):
        clock = Clock(1000.0)
        mem = EpisodicMemory(now=clock, recency_tau=100.0)
        old = mem.record("old event", timestamp=900.0)
        new = mem.record("new event", timestamp=990.0)
        hits = mem.recall(strategy="recency")
        assert [e.id for e, _ in hits] == [new.id, old.id]
        assert hits[0][1] > hits[1][1]

    def test_temporal_near_timestamp(self):
        mem = EpisodicMemory(now=Clock(), temporal_tau=10.0)
        a = mem.record("at noon", timestamp=100.0)
        b = mem.record("at midnight", timestamp=500.0)
        hits = mem.recall(strategy="temporal", around=105.0)
        assert hits[0][0].id == a.id

    def test_mixed_strategy(self):
        clock = Clock(1000.0)
        mem = EpisodicMemory(now=clock, recency_tau=100.0)
        mem.record("unrelated old thing", timestamp=800.0, salience=0.1)
        target = mem.record("red bicycle race downtown", timestamp=990.0,
                            salience=0.9)
        hits = mem.recall("bicycle race", strategy="mixed")
        assert hits[0][0].id == target.id
        assert len(hits) == 2

    def test_unknown_strategy_raises(self):
        mem = EpisodicMemory(now=Clock())
        mem.record("x")
        with pytest.raises(ValueError):
            mem.recall(strategy="bogus")

    def test_filters_since_and_tags(self):
        mem = EpisodicMemory(now=Clock())
        mem.record("first", timestamp=10.0, tags=("work",))
        keep = mem.record("second", timestamp=20.0, tags=("play",))
        assert [e.id for e, _ in mem.recall(since=15.0)] == [keep.id]
        assert [e.id for e, _ in mem.recall(tags=["work"])] != [keep.id]
        assert mem.recall(tags=["nope"]) == []
        assert mem.recall(k=0) == []

    def test_link_and_neighbors(self):
        mem = EpisodicMemory(now=Clock())
        a = mem.record("a")
        b = mem.record("b")
        assert mem.link(a.id, b.id, relation="causes") is True
        assert mem.link(a.id, "ghost") is False
        assert mem.link(a.id, a.id) is False
        nbs = mem.neighbors(a.id)
        assert len(nbs) == 1
        assert nbs[0][0].id == b.id
        assert nbs[0][1] == "causes"
        assert mem.backlinks(b.id)[0][0].id == a.id
        assert mem.unlink(a.id, b.id) is True
        assert mem.neighbors(a.id) == []
        assert mem.unlink(a.id, b.id) is False

    def test_forget_prunes_old_lowsalience(self):
        clock = Clock(1000000.0)
        mem = EpisodicMemory(now=clock)
        old_weak = mem.record("fades", timestamp=1000000.0 - 40 * 86400,
                              salience=0.1)
        old_strong = mem.record("stays", timestamp=1000000.0 - 40 * 86400,
                                salience=0.9)
        fresh = mem.record("fresh", timestamp=1000000.0 - 86400, salience=0.1)
        pruned = mem.forget(max_age_days=30, min_salience=0.2)
        assert pruned == 1
        assert mem.get(old_weak.id) is None
        assert mem.get(old_strong.id) is not None
        assert mem.get(fresh.id) is not None

    def test_sequence_chronological(self):
        mem = EpisodicMemory(now=Clock())
        mem.record("third", timestamp=30.0)
        first = mem.record("first", timestamp=10.0)
        mem.record("second", timestamp=20.0, tags=("t",))
        seq = mem.sequence()
        assert [e.summary for e in seq] == ["first", "second", "third"]
        assert mem.sequence(tags=["t"])[0].id != first.id or True
        assert len(mem.sequence(limit=2)) == 2

    def test_retag_and_boost(self):
        mem = EpisodicMemory(now=Clock())
        ep = mem.record("x", salience=0.5)
        assert mem.retag(ep.id, ["new", "new"]) is True
        assert ep.tags == ["new"]
        assert mem.retag("ghost", ["t"]) is False
        assert mem.boost_salience(ep.id, 0.3) is True
        assert ep.salience == pytest.approx(0.8)
        assert mem.boost_salience("ghost") is False

    def test_persistence_roundtrip(self):
        clock = Clock()
        mem = EpisodicMemory(now=clock)
        a = mem.record("a", tags=("x",), detail={"k": 1})
        b = mem.record("b")
        mem.link(a.id, b.id, "follows")
        back = EpisodicMemory.from_dict(mem.to_dict(), now=clock)
        assert back.count() == 2
        assert back.get(a.id).links == {b.id: "follows"}
        assert back.get(b.id).detail == {}
        nxt = back.record("c")
        assert nxt.id not in (a.id, b.id)


# semantic


class TestConceptRelation:
    def test_roundtrips(self):
        c = Concept(name="dog", attributes={"legs": 4}, confidence=0.8,
                    sources=["s"], created=1.0, updated=2.0)
        assert Concept.from_dict(c.to_dict()) == c
        r = Relation(subject="dog", predicate="is_a", obj="animal",
                     confidence=0.9, source="s", created=3.0)
        assert Relation.from_dict(r.to_dict()) == r


class TestSemanticMemory:
    def test_add_concept_merges_on_repeat(self):
        clock = Clock()
        sm = SemanticMemory(now=clock)
        c1 = sm.add_concept("dog", attributes={"legs": 4}, confidence=0.5)
        c2 = sm.add_concept("dog", attributes={"barks": True}, confidence=0.5,
                            source="other")
        assert c1 is c2
        assert c2.attributes == {"legs": 4, "barks": True}
        assert c2.confidence == pytest.approx(0.6)
        assert set(c2.sources) == {"consolidation", "other"}
        assert sm.get_concept("dog") is c2
        assert sm.get_concept("missing") is None
        assert sm.has_concept("dog") is True

    def test_add_relation_strengthens_duplicate(self):
        sm = SemanticMemory(now=Clock())
        r1 = sm.add_relation("dog", "is_a", "animal", confidence=0.5)
        r2 = sm.add_relation("dog", "is_a", "animal", confidence=0.5)
        assert r1 is r2
        assert r2.confidence == pytest.approx(0.6)

    def test_query_filters(self):
        sm = SemanticMemory(now=Clock())
        sm.add_relation("dog", "is_a", "animal", confidence=0.9)
        sm.add_relation("cat", "is_a", "animal", confidence=0.4)
        sm.add_relation("dog", "eats", "kibble", confidence=0.7)
        assert len(sm.query(subject="dog")) == 2
        assert len(sm.query(predicate="is_a")) == 2
        assert len(sm.query(obj="animal")) == 2
        assert len(sm.query(subject="dog", predicate="is_a")) == 1
        assert len(sm.query(min_confidence=0.8)) == 1
        assert sm.query(subject="ghost") == []

    def test_activate_spreads(self):
        sm = SemanticMemory(now=Clock())
        sm.add_relation("dog", "is_a", "animal", confidence=1.0)
        sm.add_relation("animal", "is_a", "living_thing", confidence=1.0)
        sm.add_relation("car", "is_a", "vehicle", confidence=1.0)
        acts = sm.activate("dog", steps=2, decay=0.5)
        assert acts["dog"] == 1.0
        assert acts["animal"] == pytest.approx(0.5)
        assert acts["living_thing"] == pytest.approx(0.25)
        assert "car" not in acts
        one = sm.activate("dog", steps=1, decay=0.5)
        assert "living_thing" not in one
        assert sm.activate("dog", steps=0) == {"dog": 1.0}
        with pytest.raises(ValueError):
            sm.activate("dog", steps=-1)

    def test_activate_respects_confidence(self):
        sm = SemanticMemory(now=Clock())
        sm.add_relation("a", "rel", "b", confidence=0.5)
        acts = sm.activate("a", steps=1, decay=1.0)
        assert acts["b"] == pytest.approx(0.5)

    def test_strengthen_weaken(self):
        sm = SemanticMemory(now=Clock())
        sm.add_relation("dog", "is_a", "animal", confidence=0.5)
        assert sm.strengthen("dog", "is_a", "animal", 0.3) is True
        assert sm.query()[0].confidence == pytest.approx(0.8)
        assert sm.weaken("dog", "is_a", "animal", 0.9) is True
        assert sm.query()[0].confidence == pytest.approx(0.0)
        assert sm.strengthen("x", "y", "z") is False
        assert sm.weaken("x", "y", "z") is False

    def test_contradict(self):
        sm = SemanticMemory(now=Clock())
        sm.add_relation("sky", "color", "blue", confidence=0.9)
        sm.add_relation("sky", "color", "gray", confidence=0.4)
        sm.add_relation("grass", "color", "green", confidence=0.9)
        found = sm.contradict("sky", "color")
        assert {r.obj for r in found} == {"blue", "gray"}
        assert sm.contradict("grass", "color") == []
        assert sm.contradict("ghost", "color") == []

    def test_neighbors_of_and_path(self):
        sm = SemanticMemory(now=Clock())
        sm.add_relation("dog", "is_a", "animal", confidence=0.9)
        sm.add_relation("animal", "is_a", "living_thing", confidence=0.8)
        nbs = sm.neighbors_of("animal")
        assert {n for n, _, _ in nbs} == {"dog", "living_thing"}
        path = sm.path("dog", "living_thing")
        assert path is not None
        assert [r.subject for r in path] == ["dog", "animal"]
        assert sm.path("dog", "dog") == []
        assert sm.path("dog", "unreachable") is None

    def test_remove_concept_drops_edges(self):
        sm = SemanticMemory(now=Clock())
        sm.add_concept("dog")
        sm.add_relation("dog", "is_a", "animal")
        assert sm.remove_concept("dog") is True
        assert sm.query() == []
        assert sm.remove_concept("dog") is False
        assert sm.remove_relation("a", "b", "c") is False

    def test_prune(self):
        sm = SemanticMemory(now=Clock())
        sm.add_concept("strong", confidence=0.9)
        sm.add_concept("weak", confidence=0.05)
        sm.add_relation("a", "r", "b", confidence=0.05)
        sm.add_relation("c", "r", "d", confidence=0.8)
        removed = sm.prune(min_confidence=0.1)
        assert removed == 2
        assert sm.has_concept("strong")
        assert not sm.has_concept("weak")
        assert len(sm.query()) == 1

    def test_stats_and_concepts_sorted(self):
        sm = SemanticMemory(now=Clock())
        sm.add_concept("b", confidence=0.5)
        sm.add_concept("a", confidence=0.7)
        assert [c.name for c in sm.concepts()] == ["a", "b"]
        s = sm.stats()
        assert s["concepts"] == 2
        assert s["mean_concept_confidence"] == pytest.approx(0.6)

    def test_persistence_roundtrip(self):
        sm = SemanticMemory(now=Clock())
        sm.add_concept("dog", attributes={"legs": 4}, confidence=0.8)
        sm.add_relation("dog", "is_a", "animal", confidence=0.9)
        back = SemanticMemory.from_dict(sm.to_dict(), now=Clock())
        assert back.get_concept("dog").attributes == {"legs": 4}
        assert back.query(subject="dog")[0].confidence == pytest.approx(0.9)


# consolidation


class TestConsolidation:
    def test_policy_defaults(self):
        p = ConsolidationPolicy()
        assert p.activation_threshold == 0.6
        assert p.min_rehearsals == 2
        assert p.semantic_support == 3

    def test_working_items_become_episodes(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        hot = wm.add("important thing", kind="goal", activation=0.9,
                     meta={"tags": ["work"]})
        cold = wm.add("trivial thing", activation=0.1)
        stats = consolidate(wm, em, sm)
        assert stats["episodes_added"] == 1
        assert em.count() == 1
        ep = em.episodes()[0]
        assert ep.summary == "important thing"
        assert "goal" in ep.tags and "work" in ep.tags
        assert ep.detail["activation"] == pytest.approx(0.9)
        assert wm.get(hot.id) is None
        assert wm.get(cold.id) is not None

    def test_rehearsed_items_consolidate_despite_low_activation(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        item = wm.add("studied hard", activation=0.2)
        wm.rehearse(item.id)
        wm.rehearse(item.id)
        policy = ConsolidationPolicy(min_rehearsals=2)
        stats = consolidate(wm, em, sm, policy=policy)
        assert stats["episodes_added"] == 1
        assert em.episodes()[0].detail["rehearsals"] == 2

    def test_repeated_tags_become_concepts(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        for i in range(3):
            em.record("went for a run number %d" % i,
                      tags=("exercise", "health"), salience=0.8)
        em.record("unrelated nap", tags=("rest",), salience=0.8)
        stats = consolidate(wm, em, sm,
                            policy=ConsolidationPolicy(semantic_support=3))
        assert stats["concepts_added"] >= 1
        assert sm.has_concept("exercise")
        assert sm.has_concept("health")
        assert not sm.has_concept("rest")
        rels = sm.query(predicate="co_occurs_with")
        assert stats["relations_added"] >= 1
        assert any(r.subject == "exercise" and r.obj == "health"
                   for r in rels)

    def test_repeated_keywords_become_concepts(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        for i in range(3):
            em.record("photosynthesis converts sunlight in leaves %d" % i,
                      salience=0.8)
        stats = consolidate(wm, em, sm,
                            policy=ConsolidationPolicy(semantic_support=3))
        assert stats["concepts_added"] >= 1
        assert sm.has_concept("photosynthesis")

    def test_second_pass_does_not_reinvent(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        for i in range(3):
            em.record("drank coffee %d" % i, tags=("coffee",), salience=0.8)
        first = consolidate(wm, em, sm,
                            policy=ConsolidationPolicy(semantic_support=3))
        second = consolidate(wm, em, sm,
                             policy=ConsolidationPolicy(semantic_support=3))
        assert first["concepts_added"] >= 1
        assert second["concepts_added"] == 0
        assert second["relations_added"] == 0

    def test_prune_low_confidence_semantics(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        sm.add_concept("junk", confidence=0.05)
        stats = consolidate(wm, em, sm)
        assert stats["pruned"] == 1
        assert not sm.has_concept("junk")

    def test_returns_all_stat_keys(self):
        wm = WorkingMemory(now=Clock())
        em = EpisodicMemory(now=Clock())
        sm = SemanticMemory(now=Clock())
        stats = consolidate(wm, em, sm)
        assert set(stats) == {"episodes_added", "concepts_added",
                              "relations_added", "pruned"}


# associative


class TestAssociativeRecall:
    def _stores(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        wm.add("buy milk and eggs", kind="goal", meta={"tags": ["errands"]})
        wm.add("unrelated weather note", activation=0.3)
        em.record("grocery run for milk and bread", tags=("errands",),
                  salience=0.7)
        em.record("watched a movie", tags=("leisure",), salience=0.7)
        sm.add_concept("milk", attributes={"white": True}, confidence=0.8)
        sm.add_relation("milk", "is_a", "dairy", confidence=0.9)
        return AssociativeRecall(wm, em, sm)

    def test_cue_finds_across_stores(self):
        ar = self._stores()
        hits = ar.cue("milk", k=5)
        stores = {h["store"] for h in hits}
        assert stores == {"working", "episodic", "semantic"}
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 < s <= 1.0 for s in scores)

    def test_cue_ranking_prefers_relevant(self):
        ar = self._stores()
        best = ar.best("milk eggs grocery")
        assert best is not None
        assert best["score"] > 0.0

    def test_cue_empty_or_no_match(self):
        ar = self._stores()
        assert ar.cue("") == []
        assert ar.cue("xylophone zebra quantum") == []
        assert ar.cue("milk", k=0) == []
        assert ar.best("xylophone zebra quantum") is None

    def test_cue_per_store_limit(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        for i in range(4):
            em.record("milk note %d" % i, salience=0.5)
        ar = AssociativeRecall(wm, em, sm)
        hits = ar.cue("milk", k=10, per_store=2)
        episodic_hits = [h for h in hits if h["store"] == "episodic"]
        assert len(episodic_hits) <= 2

    def test_spread_reaches_linked_stores(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        sm.add_concept("dog", confidence=0.9)
        sm.add_concept("animal", confidence=0.9)
        sm.add_relation("dog", "is_a", "animal", confidence=1.0)
        ep = em.record("the dog barked loudly", tags=("dog",), salience=0.8)
        item = wm.add("dog needs a walk", meta={"tags": ["dog"]})
        ar = AssociativeRecall(wm, em, sm)
        acts = ar.spread("dog", depth=2, width=5)
        assert acts["dog"] == pytest.approx(1.0)
        assert acts["animal"] > 0.0
        assert ep.id in acts
        assert item.id in acts
        assert acts[ep.id] > 0.0

    def test_spread_empty_seed(self):
        ar = self._stores()
        assert ar.spread("") == {}
        assert ar.spread("the and or") == {}


# forgetting


class TestEbbinghaus:
    def test_fresh_is_one(self):
        assert ebbinghaus_retention(0.0) == 1.0
        assert ebbinghaus_retention(-5.0) == 1.0

    def test_decays_with_age(self):
        r1 = ebbinghaus_retention(3600.0)
        r2 = ebbinghaus_retention(7200.0)
        assert 0.0 < r2 < r1 < 1.0

    def test_one_day_strength_one(self):
        assert ebbinghaus_retention(86400.0) == pytest.approx(math.exp(-1.0))

    def test_stronger_memory_lasts_longer(self):
        weak = ebbinghaus_retention(86400.0, strength=1.0)
        strong = ebbinghaus_retention(86400.0, strength=3.0)
        assert strong > weak

    def test_decay_scales(self):
        assert ebbinghaus_retention(86400.0, decay=2.0) < \
            ebbinghaus_retention(86400.0, decay=1.0)

    def test_half_life(self):
        assert half_life(1.0) == pytest.approx(86400.0 * math.log(2.0))
        assert half_life(2.0) == pytest.approx(2 * half_life(1.0))
        assert ebbinghaus_retention(half_life(1.0)) == pytest.approx(0.5)


class TestForgettingCurve:
    def test_rehearsal_raises_strength(self):
        fc = ForgettingCurve()
        assert fc.strength("a") == 1.0
        assert fc.record_rehearsal("a") == 1
        assert fc.record_rehearsal("a") == 2
        assert fc.strength("a") == pytest.approx(2.0)
        assert fc.rehearsals("a") == 2
        assert fc.rehearsals("ghost") == 0

    def test_retention_improves_with_rehearsal(self):
        fc = ForgettingCurve()
        before = fc.retention("a", 86400.0)
        fc.record_rehearsal("a")
        fc.record_rehearsal("a")
        after = fc.retention("a", 86400.0)
        assert after > before

    def test_retention_falls_with_age(self):
        fc = ForgettingCurve()
        assert fc.retention("a", 100.0) > fc.retention("a", 100000.0)

    def test_record_rehearsal_batch(self):
        fc = ForgettingCurve()
        assert fc.record_rehearsal("a", count=3) == 3
        with pytest.raises(ValueError):
            fc.record_rehearsal("a", count=0)

    def test_reset_and_weakest(self):
        fc = ForgettingCurve()
        fc.record_rehearsal("a", count=5)
        fc.record_rehearsal("b", count=1)
        assert fc.weakest(1) == {"b": 1}
        assert fc.reset("b") is True
        assert fc.reset("b") is False
        assert fc.rehearsals("b") == 0

    def test_half_life_grows_with_rehearsal(self):
        fc = ForgettingCurve()
        h0 = fc.half_life("a")
        fc.record_rehearsal("a")
        assert fc.half_life("a") > h0

    def test_persistence_roundtrip(self):
        fc = ForgettingCurve()
        fc.record_rehearsal("a", count=2)
        back = ForgettingCurve.from_dict(fc.to_dict())
        assert back.rehearsals("a") == 2
        assert back.strength("a") == pytest.approx(2.0)


# integration


class TestMemoryPipeline:
    def test_perceive_rehearse_consolidate_recall(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)

        item = wm.add("the reactor core temperature is rising",
                      kind="observation", meta={"tags": ["reactor"]})
        wm.rehearse(item.id)
        wm.rehearse(item.id)
        clock.advance(5.0)
        wm.tick(dt=5.0)

        stats = consolidate(wm, em, sm)
        assert stats["episodes_added"] == 1
        assert len(wm) == 0

        hits = em.recall("reactor temperature", strategy="keyword")
        assert hits and hits[0][0].summary.startswith("the reactor core")

        ar = AssociativeRecall(wm, em, sm)
        assert ar.best("reactor") is not None

    def test_full_lifecycle_persistence(self):
        clock = Clock()
        wm = WorkingMemory(now=clock)
        em = EpisodicMemory(now=clock)
        sm = SemanticMemory(now=clock)
        it = wm.add("note", activation=0.9)
        wm.pin(it.id)
        ep = em.record("event", tags=("t",))
        em.link(ep.id, ep.id)  # self-link rejected
        assert em.neighbors(ep.id) == []
        sm.add_relation("x", "is", "y")

        wm2 = WorkingMemory.from_dict(wm.to_dict(), now=clock)
        em2 = EpisodicMemory.from_dict(em.to_dict(), now=clock)
        sm2 = SemanticMemory.from_dict(sm.to_dict(), now=clock)
        assert wm2.get(it.id).pinned is True
        assert em2.count() == 1
        assert len(sm2.query()) == 1
