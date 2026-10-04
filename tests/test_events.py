"""Tests for the event bus."""

import json

import pytest

from cognix.events import EventBus


def make_bus(**kwargs):
    return EventBus(max_log=kwargs.pop("max_log", 50), now=lambda: 1234.0, **kwargs)


def test_emit_calls_subscriber_with_event():
    bus = make_bus()
    seen = []
    bus.subscribe("goal.added", seen.append)
    event = bus.emit("goal.added", goal_id="g1")
    assert len(seen) == 1
    assert seen[0]["type"] == "goal.added"
    assert seen[0]["payload"]["goal_id"] == "g1"
    assert seen[0]["at"] == 1234.0
    assert event is seen[0]


def test_wildcard_gets_everything():
    bus = make_bus()
    seen = []
    bus.subscribe("*", seen.append)
    bus.emit("a", x=1)
    bus.emit("b", x=2)
    assert [e["type"] for e in seen] == ["a", "b"]


def test_unmatched_types_do_not_fire():
    bus = make_bus()
    seen = []
    bus.subscribe("goal.added", seen.append)
    bus.emit("goal.completed", goal_id="g1")
    assert seen == []


def test_unsubscribe_stops_delivery():
    bus = make_bus()
    seen = []
    bus.subscribe("x", seen.append)
    bus.unsubscribe("x", seen.append)
    bus.emit("x")
    assert seen == []
    # unsubscribing something never added is a no-op
    bus.unsubscribe("x", seen.append)


def test_no_double_subscription():
    bus = make_bus()
    seen = []
    bus.subscribe("x", seen.append)
    bus.subscribe("x", seen.append)
    bus.emit("x")
    assert len(seen) == 1


def test_seq_numbers_increase():
    bus = make_bus()
    e1 = bus.emit("x")
    e2 = bus.emit("y")
    assert e2["seq"] == e1["seq"] + 1


def test_log_bounded_by_max_log():
    bus = make_bus(max_log=3)
    for i in range(6):
        bus.emit("x", i=i)
    assert len(bus.log) == 3
    assert [e["payload"]["i"] for e in bus.log] == [3, 4, 5]


def test_recent_filters_by_type_and_n():
    bus = make_bus()
    bus.emit("a")
    bus.emit("b")
    bus.emit("a")
    bus.emit("b")
    recent_a = bus.recent("a")
    assert len(recent_a) == 2
    assert all(e["type"] == "a" for e in recent_a)
    assert len(bus.recent(n=1)) == 1
    assert bus.recent("nope") == []


def test_counts():
    bus = make_bus()
    bus.emit("a")
    bus.emit("a")
    bus.emit("b")
    assert bus.counts() == {"a": 2, "b": 1}


def test_disabled_bus_emits_nothing():
    bus = make_bus(enabled=False)
    seen = []
    bus.subscribe("*", seen.append)
    result = bus.emit("x")
    assert result is None
    assert seen == []
    assert len(bus.log) == 0
    bus.enabled = True
    bus.emit("x")
    assert len(seen) == 1


def test_failing_handler_does_not_break_others():
    bus = make_bus()
    order = []

    def bad(event):
        order.append("bad")
        raise RuntimeError("boom")

    def good(event):
        order.append("good")

    bus.subscribe("x", bad)
    bus.subscribe("x", good)
    bus.emit("x")
    assert order == ["bad", "good"]
    assert len(bus.errors) == 1
    assert bus.errors[0][0] == "x"


def test_roundtrip_serializes():
    bus = make_bus()
    bus.emit("goal.added", goal_id="g1", priority=0.5)
    bus.emit("tool.step_finished", step_id="s1", ok=True)
    data = bus.to_dict()
    json.dumps(data)  # must be json-safe
    clone = EventBus.from_dict(data)
    assert clone.counts() == bus.counts()
    assert clone.recent("goal.added")[0]["payload"]["goal_id"] == "g1"
    assert clone.recent()[-1]["seq"] == bus.recent()[-1]["seq"]


def test_clear():
    bus = make_bus()
    bus.emit("x")
    bus.clear()
    assert bus.recent() == []
    assert bus.counts() == {}
