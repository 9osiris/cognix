"""Tests for the attention subsystem: salience, focus, arousal."""

import math
from types import SimpleNamespace

import pytest

from cognix.attention import (
    WEIGHTS,
    Arousal,
    Focus,
    SalienceSignal,
    batch_score,
    novelty_score,
    relevance_score,
    score_observation,
    surprise_score,
    urgency_score,
)


# small stubs standing in for episodic memory and belief stores


class FakeEpisodic:
    def __init__(self, texts):
        self._texts = list(texts)

    def recent(self, limit=50):
        return [SimpleNamespace(text=text) for text in self._texts[-limit:]]


class FakeBeliefs:
    def __init__(self, items):
        self._items = [
            SimpleNamespace(proposition=prop, confidence=conf)
            for prop, conf in items
        ]

    def query(self, keyword):
        word = keyword.lower()
        return [item for item in self._items if word in item.proposition.lower()]


def make_goal(description):
    return SimpleNamespace(description=description)


class TestSalienceSignal:
    def test_clamps_score_to_unit_range(self):
        assert SalienceSignal("x", 2.0).score == 1.0
        assert SalienceSignal("x", -0.5).score == 0.0
        assert SalienceSignal("x", 0.4).score == 0.4

    def test_dict_roundtrip(self):
        signal = SalienceSignal("novelty", 0.75, "very new")
        restored = SalienceSignal.from_dict(signal.to_dict())
        assert restored == signal
        assert restored.reason == "very new"

    def test_repr_mentions_name_and_score(self):
        assert "novelty" in repr(SalienceSignal("novelty", 0.5))


class TestNovelty:
    def test_novel_text_outscores_repeated_text(self):
        episodic = FakeEpisodic(["the cat sat on the mat", "dogs are loyal animals"])
        novel = novelty_score("quantum entanglement enables teleportation", episodic)
        repeated = novelty_score("the cat sat on the mat", episodic)
        assert novel > repeated
        assert novel == pytest.approx(1.0)
        assert repeated == pytest.approx(0.0)

    def test_partial_overlap_gives_partial_novelty(self):
        episodic = FakeEpisodic(["the cat sat on the mat"])
        score = novelty_score("the cat sat on the rug", episodic)
        assert 0.0 < score < 1.0

    def test_no_episodic_memory_is_fully_novel(self):
        assert novelty_score("anything at all", None) == pytest.approx(1.0)
        assert novelty_score("anything at all", FakeEpisodic([])) == pytest.approx(1.0)

    def test_empty_text_has_no_novelty(self):
        assert novelty_score("", FakeEpisodic(["something"])) == pytest.approx(0.0)

    def test_only_last_fifty_episodes_count(self):
        old = ["cats are cute"]
        fillers = ["filler sentence number %d here" % i for i in range(60)]
        episodic = FakeEpisodic(old + fillers)
        # "cats are cute" fell out of the 50-episode window
        assert novelty_score("cats are cute", episodic) == pytest.approx(1.0)
        assert novelty_score("cats are cute", FakeEpisodic(old)) == pytest.approx(0.0)

    def test_accepts_plain_iterables_and_dicts(self):
        assert novelty_score("cats are cute", [{"text": "cats are cute"}]) == pytest.approx(0.0)
        assert novelty_score("cats are cute", ["cats are cute"]) == pytest.approx(0.0)


class TestRelevance:
    def test_goal_relevant_text_outscores_irrelevant(self):
        goals = [make_goal("learn spanish vocabulary")]
        relevant = relevance_score("i want to learn spanish vocabulary daily", goals)
        irrelevant = relevance_score("the toaster needs repair", goals)
        assert relevant > irrelevant
        assert relevant == pytest.approx(1.0)
        assert irrelevant == pytest.approx(0.0)

    def test_best_goal_wins(self):
        goals = [make_goal("bake sourdough bread"), make_goal("learn spanish")]
        score = relevance_score("learn spanish verbs", goals)
        assert score == pytest.approx(1.0)

    def test_no_goals_means_zero(self):
        assert relevance_score("learn spanish", []) == pytest.approx(0.0)
        assert relevance_score("learn spanish", None) == pytest.approx(0.0)

    def test_partial_overlap(self):
        goals = [make_goal("red green blue yellow")]
        assert relevance_score("i like red and blue", goals) == pytest.approx(0.5)


class TestSurprise:
    def test_believed_text_is_not_surprising(self):
        beliefs = FakeBeliefs([("the sky is blue", 0.9)])
        calm = surprise_score("the sky is blue today", beliefs)
        strange = surprise_score("quantum foam fluctuates wildly", beliefs)
        assert calm < strange
        assert calm == pytest.approx(0.1)
        assert strange == pytest.approx(1.0)

    def test_strongest_matching_belief_sets_surprise(self):
        beliefs = FakeBeliefs([("the sky is blue", 0.9), ("the sky is vast", 0.4)])
        assert surprise_score("the sky is blue", beliefs) == pytest.approx(0.1)

    def test_no_beliefs_is_fully_surprising(self):
        assert surprise_score("anything", None) == pytest.approx(1.0)


class TestUrgency:
    def test_urgent_text_outscores_calm_text(self):
        urgent = urgency_score("URGENT: the server is down, fix this asap!")
        calm = urgency_score("the server seems fine today")
        assert urgent > calm
        assert urgent == pytest.approx(1.0)

    def test_markers_add_up_and_clamp(self):
        assert urgency_score("urgent critical emergency asap!!!") == pytest.approx(1.0)
        assert urgency_score("this is now due") > urgency_score("this is due")
        assert urgency_score("nothing special here") == pytest.approx(0.0)

    def test_date_mentions_count(self):
        assert urgency_score("submit the report by friday") > 0.0
        assert urgency_score("meeting on 2026-10-01") > 0.0

    def test_marker_matching_is_word_boundaried(self):
        # "nowhere" must not trigger the "now" marker
        assert urgency_score("nowhere to be found") == pytest.approx(0.0)


class TestScoreObservation:
    def test_returns_total_and_four_signals(self):
        total, signals = score_observation("the sky is blue")
        assert isinstance(total, float)
        assert 0.0 <= total <= 1.0
        assert [signal.name for signal in signals] == [
            "novelty",
            "relevance",
            "surprise",
            "urgency",
        ]
        assert all(isinstance(signal, SalienceSignal) for signal in signals)
        assert all(signal.reason for signal in signals)

    def test_total_is_weighted_sum(self):
        text = "URGENT: learn spanish verbs now"
        goals = [make_goal("learn spanish")]
        beliefs = FakeBeliefs([("spanish is hard", 0.5)])
        episodic = FakeEpisodic(["unrelated filler sentence"])
        total, _ = score_observation(text, goals=goals, beliefs=beliefs, episodic=episodic)
        expected = (
            WEIGHTS["novelty"] * novelty_score(text, episodic)
            + WEIGHTS["relevance"] * relevance_score(text, goals)
            + WEIGHTS["surprise"] * surprise_score(text, beliefs)
            + WEIGHTS["urgency"] * urgency_score(text)
        )
        assert total == pytest.approx(expected)

    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_max_clamps_to_one(self):
        total, _ = score_observation(
            "URGENT ASAP CRITICAL learn spanish",
            goals=[make_goal("learn spanish")],
            beliefs=None,
            episodic=None,
        )
        assert total == pytest.approx(1.0)

    def test_known_combination(self):
        # novelty 1, relevance 0, surprise 1, urgency 0 -> 0.5
        total, _ = score_observation("hello world")
        assert total == pytest.approx(0.5)

    def test_accepts_now_callable(self):
        total, signals = score_observation("hello", now=lambda: 123.0)
        assert 0.0 <= total <= 1.0
        assert len(signals) == 4


class TestFocus:
    def test_selects_top_k_above_threshold(self):
        focus = Focus(bandwidth=2, threshold=0.3)
        candidates = [("a", 0.9), ("b", 0.5), ("c", 0.1), ("d", 0.8)]
        assert focus.select(candidates) == ["a", "d"]

    def test_threshold_filters(self):
        focus = Focus(bandwidth=5, threshold=0.3)
        assert focus.select([("a", 0.29), ("b", 0.3), ("c", 0.31)]) == ["c", "b"]

    def test_ties_keep_input_order(self):
        focus = Focus(bandwidth=3, threshold=0.0)
        assert focus.select([("a", 0.5), ("b", 0.5), ("c", 0.5)]) == ["a", "b", "c"]

    def test_empty_candidates(self):
        assert Focus().select([]) == []

    def test_defaults(self):
        focus = Focus()
        assert focus.bandwidth == 3
        assert focus.threshold == pytest.approx(0.3)

    def test_interrupt_when_candidate_exceeds_by_margin(self):
        focus = Focus()
        assert focus.interrupt(0.5, 0.8) is True
        assert focus.interrupt(0.5, 0.8, margin=0.1) is True

    def test_no_interrupt_within_margin(self):
        focus = Focus()
        assert focus.interrupt(0.5, 0.6) is False
        assert focus.interrupt(0.5, 0.75) is False  # exactly at margin, strict
        assert focus.interrupt(0.9, 0.5) is False

    def test_dict_roundtrip(self):
        focus = Focus(bandwidth=2, threshold=0.4)
        restored = Focus.from_dict(focus.to_dict())
        assert restored.bandwidth == 2
        assert restored.threshold == pytest.approx(0.4)
        assert restored.select([("a", 0.9), ("b", 0.1)]) == ["a"]


class TestArousal:
    def test_starts_at_baseline(self):
        arousal = Arousal(baseline=0.5)
        assert arousal.level() == pytest.approx(0.5)
        assert arousal.processing_depth() == 2

    def test_update_raises_level_and_clamps(self):
        arousal = Arousal(baseline=0.5, rise=0.4)
        arousal.update(1.0)
        assert arousal.level() == pytest.approx(0.9)
        arousal.update(1.0)
        assert arousal.level() == pytest.approx(1.0)

    def test_update_scales_with_salience(self):
        arousal = Arousal(baseline=0.5, rise=0.4)
        arousal.update(0.5)
        assert arousal.level() == pytest.approx(0.7)
        before = arousal.level()
        arousal.update(0.0)
        assert arousal.level() == pytest.approx(before)

    def test_tick_decays_toward_baseline(self):
        arousal = Arousal(baseline=0.5, rise=0.4, decay_rate=0.1)
        arousal.update(1.0)
        arousal.tick(10.0)
        assert arousal.level() == pytest.approx(0.5 + 0.4 * math.exp(-1.0))

    def test_tick_never_crosses_baseline(self):
        arousal = Arousal(baseline=0.5, rise=0.4, decay_rate=0.1)
        arousal.update(1.0)
        arousal.tick(10000.0)
        assert arousal.level() == pytest.approx(0.5)

    def test_tick_with_zero_dt_changes_nothing(self):
        arousal = Arousal(baseline=0.5, rise=0.4)
        arousal.update(1.0)
        arousal.tick(0.0)
        assert arousal.level() == pytest.approx(0.9)

    def test_processing_depth_mapping(self):
        assert Arousal(baseline=0.2).processing_depth() == 1
        assert Arousal(baseline=0.34).processing_depth() == 1
        assert Arousal(baseline=0.35).processing_depth() == 2
        assert Arousal(baseline=0.69).processing_depth() == 2
        assert Arousal(baseline=0.7).processing_depth() == 3
        hot = Arousal(baseline=0.5, rise=0.4)
        hot.update(1.0)
        assert hot.processing_depth() == 3

    def test_now_callable_tracks_activity(self):
        clock = [100.0]
        arousal = Arousal(now=lambda: clock[0])
        assert arousal.last_active == pytest.approx(100.0)
        clock[0] = 250.0
        arousal.update(0.5)
        assert arousal.last_active == pytest.approx(250.0)

    def test_dict_roundtrip_preserves_level(self):
        arousal = Arousal(baseline=0.5, rise=0.4, decay_rate=0.2)
        arousal.update(1.0)
        arousal.tick(5.0)
        restored = Arousal.from_dict(arousal.to_dict())
        assert restored.level() == pytest.approx(arousal.level())
        assert restored.baseline == pytest.approx(0.5)
        assert restored.rise == pytest.approx(0.4)
        assert restored.decay_rate == pytest.approx(0.2)


class TestBatchScore:
    def test_orders_best_first(self):
        texts = ["the sky is blue", "URGENT ASAP CRITICAL server down"]
        scored = batch_score(texts)
        assert scored[0][0] == "URGENT ASAP CRITICAL server down"
        assert scored[0][1] >= scored[1][1]
        assert len(scored[0]) == 3  # text, total, signals

    def test_empty_input(self):
        assert batch_score([]) == []

    def test_passes_stores_through(self):
        scored = batch_score(
            ["learn spanish verbs"],
            goals=[make_goal("learn spanish")],
            beliefs=FakeBeliefs([]),
            episodic=FakeEpisodic([]),
        )
        assert scored[0][1] == pytest.approx(0.3 + 0.3 + 0.2)


class TestFocusRanked:
    def test_ranked_returns_pairs_above_threshold(self):
        focus = Focus(bandwidth=2, threshold=0.3)
        assert focus.ranked([("a", 0.9), ("b", 0.1), ("c", 0.5)]) == [
            ("a", 0.9),
            ("c", 0.5),
        ]

    def test_select_respects_bandwidth_over_ranked(self):
        focus = Focus(bandwidth=1, threshold=0.0)
        assert focus.select([("a", 0.2), ("b", 0.8)]) == ["b"]


class TestArousalReset:
    def test_reset_returns_to_baseline(self):
        arousal = Arousal(baseline=0.5, rise=0.4)
        arousal.update(1.0)
        assert arousal.level() > 0.5
        arousal.reset()
        assert arousal.level() == pytest.approx(0.5)
        assert arousal.processing_depth() == 2


class TestAttentionPerceptionIntegration:
    def test_full_loop(self):
        from cognix.perception import (
            BeliefStore,
            observation_to_beliefs,
            parse_observation,
        )

        store = BeliefStore()
        arousal = Arousal()
        focus = Focus(bandwidth=2, threshold=0.2)
        texts = [
            "the sky is blue",
            "URGENT: the server is down, fix it asap",
            "Bob likes pizza",
        ]
        scored = []
        for text in texts:
            parsed = parse_observation(text)
            for proposition, confidence in observation_to_beliefs(parsed):
                store.assert_belief(proposition, confidence, "perception")
            total, _ = score_observation(text, beliefs=store)
            scored.append((text, total))
            arousal.update(total)
        chosen = focus.select(scored)
        assert "URGENT: the server is down, fix it asap" in chosen
        assert arousal.level() > 0.5
        assert arousal.processing_depth() >= 2
        assert store.get("bob likes pizza") is not None

    def test_surprise_drops_as_beliefs_accumulate(self):
        from cognix.perception import BeliefStore

        store = BeliefStore()
        before = surprise_score("the sky is blue today", store)
        store.assert_belief("the sky is blue", 0.9, "perception")
        after = surprise_score("the sky is blue today", store)
        assert after < before
        assert after == pytest.approx(0.1)
