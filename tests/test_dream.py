"""Tests for the dream cycle."""

import pytest

from cognix.memory.dream import DreamPolicy, dream
from cognix.memory.episodic import EpisodicMemory
from cognix.memory.semantic import SemanticMemory
from cognix.perception.beliefs import BeliefStore


def make_memories(n=6):
    mem = EpisodicMemory()
    sem = SemanticMemory()
    beliefs = BeliefStore()
    topics = [
        ("the atlas server is running hot", ("server", "alert")),
        ("the garden irrigation needs a timer", ("garden",)),
        ("mercury-04 runs the staging database", ("server", "db")),
        ("the herons migrate at dawn", ("nature",)),
        ("the cache hit rate dropped overnight", ("cache", "alert")),
        ("sourdough starter needs feeding twice daily", ("food",)),
    ]
    for summary, tags in topics[:n]:
        mem.record(summary=summary, tags=tags, salience=0.7)
    return mem, sem, beliefs


def test_policy_roundtrip():
    policy = DreamPolicy(budget=6, dreams=2, seed=7)
    clone = DreamPolicy.from_dict(policy.to_dict())
    assert clone.budget == 6
    assert clone.dreams == 2
    assert clone.seed == 7


def test_dream_records_dream_episodes():
    mem, sem, beliefs = make_memories()
    before = len(mem)
    stats = dream(mem, sem, beliefs, DreamPolicy(seed=1))
    assert stats["dreams"] >= 1
    assert len(mem) == before + stats["dreams"]
    for ep_id in stats["dream_ids"]:
        ep = mem.get(ep_id)
        assert "dream" in ep.tags
        assert ep.source == "dream"


def test_dream_links_unlinked_pairs():
    mem, sem, beliefs = make_memories()
    stats = dream(mem, sem, beliefs, DreamPolicy(dreams=2, seed=2))
    assert stats["links_added"] >= 1
    dreamed = [mem.get(i) for i in stats["dream_ids"]]
    for ep in dreamed:
        pair = ep.detail["pair"]
        assert any(o.id == pair[1] for o, _ in mem.neighbors(pair[0]))


def test_dream_insights_are_low_confidence_hypotheses():
    mem, sem, beliefs = make_memories()
    stats = dream(mem, sem, beliefs, DreamPolicy(seed=3))
    assert stats["insights"] >= 1
    for belief in beliefs.strongest(50):
        if "dream" in belief.sources:
            assert belief.confidence <= 0.5 + 1e-9
            assert belief.proposition.startswith("dream insight:")
    rels = sem.query(predicate="dream-related-to")
    assert len(rels) == stats["relations_proposed"]


def test_dream_is_deterministic_with_seed():
    mem_a, sem_a, beliefs_a = make_memories()
    mem_b, sem_b, beliefs_b = make_memories()
    stats_a = dream(mem_a, sem_a, beliefs_a, DreamPolicy(seed=42))
    stats_b = dream(mem_b, sem_b, beliefs_b, DreamPolicy(seed=42))
    assert stats_a["dream_ids"] != []
    a_summaries = sorted(mem_a.get(i).summary for i in stats_a["dream_ids"])
    b_summaries = sorted(mem_b.get(i).summary for i in stats_b["dream_ids"])
    assert a_summaries == b_summaries


def test_dream_skips_already_linked_pairs():
    mem, sem, beliefs = make_memories(n=2)
    eps = mem.episodes()
    mem.link(eps[0].id, eps[1].id, "dreamed-with")
    stats = dream(mem, sem, beliefs, DreamPolicy(dreams=3, seed=4))
    assert stats["dreams"] == 0
    assert stats["insights"] == 0


def test_dream_needs_at_least_two_episodes():
    mem, sem, beliefs = make_memories(n=1)
    stats = dream(mem, sem, beliefs)
    assert stats["dreams"] == 0


def test_dream_without_beliefs_still_records_traces():
    mem, sem, _ = make_memories()
    stats = dream(mem, sem, None, DreamPolicy(seed=5))
    assert stats["dreams"] >= 1
    assert stats["insights"] == 0
    assert stats["relations_proposed"] == 0


def test_dream_respects_dreams_cap():
    mem, sem, beliefs = make_memories(n=6)
    stats = dream(mem, sem, beliefs, DreamPolicy(dreams=1, seed=6))
    assert stats["dreams"] == 1


def test_runtime_dream_hook():
    from cognix.runtime import CognitiveRuntime
    rt = CognitiveRuntime(workspace_dir="/tmp/cognix_dream_test")
    for text, tags in [
        ("the atlas server is running hot", ("server",)),
        ("the garden irrigation needs a timer", ("garden",)),
    ]:
        rt.episodic.record(summary=text, tags=tags, salience=0.8)
    stats = rt.dream()
    assert stats["dreams"] >= 1
    assert len(rt.beliefs) >= 1
    path = rt.save("/tmp/cognix_dream_test/state.json")
    loaded = CognitiveRuntime.load(path)
    assert loaded.dream_policy.to_dict() == rt.dream_policy.to_dict()
