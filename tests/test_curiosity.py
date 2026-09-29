"""Tests for the curiosity tracker."""

import pytest

from cognix.attention.curiosity import CuriosityTracker


def test_novel_text_scores_high_novelty():
    tracker = CuriosityTracker()
    assert tracker.novelty("quantum routers dream of electric sheep") > 0.9


def test_repeated_exposure_lowers_novelty():
    tracker = CuriosityTracker()
    text = "the atlas server is running hot"
    first = tracker.expose(text)
    tracker.expose(text)
    tracker.expose(text)
    assert tracker.novelty(text) < first


def test_bonus_is_small_and_positive():
    tracker = CuriosityTracker()
    bonus = tracker.bonus("something entirely unprecedented here")
    assert 0.0 < bonus <= tracker.bonus_scale


def test_familiarity_grows_with_exposure():
    tracker = CuriosityTracker()
    text = "the backup completed successfully"
    assert tracker.familiarity(text) == 0.0
    for _ in range(4):
        tracker.expose(text)
    assert tracker.familiarity(text) > 0.5


def test_empty_text_has_no_familiarity():
    tracker = CuriosityTracker()
    assert tracker.familiarity("") == 0.0
    assert tracker.novelty("") == 1.0


def test_boredom_after_repetition():
    tracker = CuriosityTracker(boredom_window=4)
    for _ in range(10):
        tracker.expose("the same old status report again")
    assert tracker.is_bored()


def test_no_boredom_with_variety():
    tracker = CuriosityTracker(boredom_window=4)
    tracker.expose("the atlas server is running hot")
    tracker.expose("quantum routers dream of electric sheep")
    tracker.expose("the backup completed successfully")
    tracker.expose("a heron lands on the frozen lake")
    assert not tracker.is_bored()


def test_boredom_needs_full_window():
    tracker = CuriosityTracker(boredom_window=10)
    for _ in range(3):
        tracker.expose("the same old status report again")
    assert not tracker.is_bored()


def test_top_familiar_ranks_exposed_topics():
    tracker = CuriosityTracker()
    for _ in range(3):
        tracker.expose("atlas server atlas server atlas")
    tracker.expose("lonely heron")
    top = tracker.top_familiar(2)
    assert "atlas" in top


def test_reset_clears_everything():
    tracker = CuriosityTracker()
    tracker.expose("the atlas server is running hot")
    tracker.reset()
    assert tracker.familiarity("the atlas server is running hot") == 0.0
    assert tracker.boredom() == 0.0


def test_roundtrip_preserves_state():
    tracker = CuriosityTracker()
    tracker.expose("the atlas server is running hot")
    tracker.expose("the atlas server is running hot")
    clone = CuriosityTracker.from_dict(tracker.to_dict())
    assert clone.familiarity("the atlas server is running hot") > 0.0
    assert clone.bonus_scale == tracker.bonus_scale


def test_decay_forgets_old_exposures():
    moment = [1000.0]
    tracker = CuriosityTracker(decay_rate=0.5, now=lambda: moment[0])
    tracker.expose("the atlas server is running hot")
    assert tracker.familiarity("the atlas server is running hot") > 0.0
    moment[0] += 100.0
    assert tracker.familiarity("the atlas server is running hot") == 0.0
