"""tests for plugins, config, persistence, tracing, and inspection."""
import json
import os
import time

import pytest

from cognix import plugins
from cognix.config import Config
from cognix.introspection import Tracer, query_state, summarize, summary
from cognix.persistence import SAVE_VERSION, StateError, StateStore, load_state, save_state
from cognix.plugins import PluginError, PluginManager
from cognix.tools import Tool, ToolRegistry

SAMPLE_PLUGIN = '''
from cognix.tools import Tool

def register_tools():
    def shout(text):
        return text.upper()
    return [Tool(
        name="shout",
        description="shout text",
        params={"text": {"type": "string", "description": "text", "required": True}},
        func=shout,
    )]
'''

BROKEN_PLUGIN = "this is not valid python (((\n"

NO_FACTORY_PLUGIN = "VALUE = 42\n"


def _write_plugin(plugin_dir, name, source):
    path = os.path.join(plugin_dir, name + ".py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    return path


# plugins


def test_discover_finds_py_files(tmp_path):
    _write_plugin(str(tmp_path), "alpha", SAMPLE_PLUGIN)
    _write_plugin(str(tmp_path), "beta", SAMPLE_PLUGIN)
    _write_plugin(str(tmp_path), "_private", SAMPLE_PLUGIN)
    with open(os.path.join(str(tmp_path), "notes.txt"), "w") as handle:
        handle.write("not a plugin")
    manager = PluginManager(str(tmp_path))
    assert manager.discover() == ["alpha", "beta"]


def test_discover_missing_dir_is_empty(tmp_path):
    manager = PluginManager(str(tmp_path / "nope"))
    assert manager.discover() == []


def test_load_returns_tools(tmp_path):
    _write_plugin(str(tmp_path), "alpha", SAMPLE_PLUGIN)
    manager = PluginManager(str(tmp_path))
    tools = manager.load("alpha")
    assert len(tools) == 1
    assert isinstance(tools[0], Tool)
    assert tools[0].name == "shout"


def test_load_bad_name_and_missing(tmp_path):
    manager = PluginManager(str(tmp_path))
    with pytest.raises(PluginError, match="not found"):
        manager.load("ghost")
    with pytest.raises(PluginError, match="bad plugin name"):
        manager.load("../evil")


def test_load_wraps_import_errors(tmp_path):
    _write_plugin(str(tmp_path), "broken", BROKEN_PLUGIN)
    manager = PluginManager(str(tmp_path))
    with pytest.raises(PluginError, match="failed to import"):
        manager.load("broken")


def test_load_requires_register_tools(tmp_path):
    _write_plugin(str(tmp_path), "nofactory", NO_FACTORY_PLUGIN)
    manager = PluginManager(str(tmp_path))
    with pytest.raises(PluginError, match="register_tools"):
        manager.load("nofactory")


def test_load_all_into_registry(tmp_path):
    _write_plugin(str(tmp_path), "alpha", SAMPLE_PLUGIN)
    _write_plugin(str(tmp_path), "broken", BROKEN_PLUGIN)
    _write_plugin(str(tmp_path), "_ignored", SAMPLE_PLUGIN)
    manager = PluginManager(str(tmp_path))
    registry = ToolRegistry()
    report = manager.load_all(registry)
    assert report["alpha"] == "loaded 1 tools"
    assert report["broken"].startswith("error:")
    assert registry.get("shout") is not None
    out = registry.call("shout", {"text": "hey"})
    assert out["ok"] is True and out["result"] == "HEY"


def test_load_all_duplicate_tool_names_reported(tmp_path):
    _write_plugin(str(tmp_path), "alpha", SAMPLE_PLUGIN)
    _write_plugin(str(tmp_path), "beta", SAMPLE_PLUGIN)
    manager = PluginManager(str(tmp_path))
    registry = ToolRegistry()
    report = manager.load_all(registry)
    assert report["alpha"] == "loaded 1 tools"
    assert report["beta"].startswith("error:")


# config


def test_config_defaults():
    cfg = Config.defaults()
    assert cfg.get("memory.working_capacity") == 7
    assert cfg.get("memory.decay_rate") == 0.08
    assert cfg.get("attention.bandwidth") == 3
    assert cfg.get("agent.max_retries") == 2
    assert cfg.get("tools.shell_enabled") is True
    assert cfg.get("runtime.consolidate_every") == 10
    assert cfg.validate() == []


def test_config_required_sections_present():
    cfg = Config.defaults()
    for section, keys in {
        "memory": ["working_capacity", "decay_rate", "consolidation_threshold"],
        "attention": ["bandwidth", "salience_threshold"],
        "agent": ["max_retries", "step_timeout", "default_strategy"],
        "tools": ["workspace_dir", "shell_enabled"],
        "runtime": ["autosave", "save_path", "consolidate_every"],
    }.items():
        for key in keys:
            assert cfg.get(section + "." + key) is not None, (section, key)


def test_config_load_from_json(tmp_path):
    path = str(tmp_path / "cfg.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"memory": {"working_capacity": 9}}, handle)
    cfg = Config.load(path)
    assert cfg.get("memory.working_capacity") == 9
    assert cfg.get("memory.decay_rate") == 0.08


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("COGNIX_MEMORY__WORKING_CAPACITY", "12")
    monkeypatch.setenv("COGNIX_TOOLS__SHELL_ENABLED", "false")
    monkeypatch.setenv("COGNIX_AGENT__STEP_TIMEOUT", "2.5")
    monkeypatch.setenv("UNRELATED_VAR", "zzz")
    cfg = Config.from_env()
    assert cfg.get("memory.working_capacity") == 12
    assert cfg.get("tools.shell_enabled") is False
    assert cfg.get("agent.step_timeout") == 2.5


def test_config_merge():
    cfg = Config.defaults()
    merged = cfg.merge({"attention": {"bandwidth": 5}})
    assert merged.get("attention.bandwidth") == 5
    # original untouched
    assert cfg.get("attention.bandwidth") == 3


def test_config_set_and_get():
    cfg = Config.defaults()
    cfg.set("agent.default_strategy", "reactive")
    assert cfg.get("agent.default_strategy") == "reactive"
    assert cfg.get("nope.missing", "fallback") == "fallback"


def test_config_validate_catches_problems():
    cfg = Config({"memory": {"working_capacity": -1, "decay_rate": 5.0},
                  "agent": {"default_strategy": "teleport"}})
    problems = cfg.validate()
    assert any("memory.working_capacity" in p for p in problems)
    assert any("memory.decay_rate" in p for p in problems)
    assert any("default_strategy" in p for p in problems)


def test_config_to_from_dict_roundtrip():
    cfg = Config({"runtime": {"autosave": True}})
    data = cfg.to_dict()
    back = Config.from_dict(data)
    assert back.to_dict() == data
    # mutating the dict does not leak into the config
    data["runtime"]["autosave"] = False
    assert cfg.get("runtime.autosave") is True


# persistence


def test_save_load_roundtrip(tmp_path):
    state = {"memory": {"working": [1, 2]}, "turn": 7}
    path = str(tmp_path / "deep" / "save.json")
    save_state(state, path)
    assert os.path.isfile(path)
    loaded = load_state(path)
    assert loaded == state


def test_save_writes_version_envelope(tmp_path):
    path = str(tmp_path / "s.json")
    save_state({"a": 1}, path)
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw["version"] == SAVE_VERSION
    assert raw["state"] == {"a": 1}


def test_load_corrupt_file_errors(tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    with pytest.raises(StateError, match="corrupt"):
        load_state(path)


def test_load_missing_file_errors(tmp_path):
    with pytest.raises(StateError, match="not found"):
        load_state(str(tmp_path / "nope.json"))


def test_load_rejects_newer_version(tmp_path):
    path = str(tmp_path / "new.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"version": SAVE_VERSION + 5, "state": {}}, handle)
    with pytest.raises(StateError, match="newer"):
        load_state(path)


def test_load_runs_migration_from_version_0(tmp_path):
    # version 0 saves were bare state dicts without an envelope
    path = str(tmp_path / "old.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"legacy": True}, handle)
    assert load_state(path) == {"legacy": True}


def test_state_store_ops(tmp_path):
    store = StateStore(str(tmp_path / "saves"))
    assert store.latest() is None
    assert store.list_saves() == []
    store.save("first", {"n": 1})
    time.sleep(0.01)
    store.save("second", {"n": 2})
    assert store.load("first") == {"n": 1}
    assert store.latest() == "second"
    names = [name for name, _ in store.list_saves()]
    assert names == ["second", "first"]
    assert store.delete("first") is True
    assert store.delete("first") is False
    with pytest.raises(StateError, match="no save"):
        store.load("first")


def test_state_store_rejects_bad_names(tmp_path):
    store = StateStore(str(tmp_path))
    with pytest.raises(StateError):
        store.save("../evil", {})


# tracing


def test_tracer_nesting_and_durations():
    tracer = Tracer()
    with tracer.start("task", kind="demo"):
        time.sleep(0.01)
        with tracer.start("step"):
            pass
    roots = tracer.roots()
    assert len(roots) == 1
    task = roots[0]
    assert task.name == "task"
    assert task.attrs["kind"] == "demo"
    assert task.duration is not None and task.duration >= 0.01
    assert len(task.children) == 1
    assert task.children[0].name == "step"
    assert task.children[0].duration is not None


def test_span_duration_none_while_open():
    tracer = Tracer()
    span = tracer.start("open")
    span.__enter__()
    try:
        assert span.duration is None
    finally:
        span.__exit__(None, None, None)
    assert span.duration is not None


def test_tracer_events():
    tracer = Tracer()
    with tracer.start("task"):
        tracer.event("started", note="go")
        tracer.event("done")
    events = tracer.events()
    assert [e["name"] for e in events] == ["started", "done"]
    assert events[0]["attrs"] == {"note": "go"}
    assert events[0]["depth"] == 1


def test_tracer_timeline_output():
    tracer = Tracer()
    with tracer.start("task"):
        time.sleep(0.01)
        with tracer.start("child"):
            pass
    text = tracer.timeline()
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("task")
    assert "ms]" in lines[0]
    assert lines[1].startswith("  child")
    assert "ms]" in lines[1]


def test_tracer_clear():
    tracer = Tracer()
    with tracer.start("task"):
        tracer.event("e")
    tracer.clear()
    assert tracer.roots() == []
    assert tracer.events() == []
    assert tracer.timeline() == ""


def test_tracer_to_from_dict_roundtrip():
    tracer = Tracer()
    with tracer.start("task", kind="x"):
        with tracer.start("child"):
            pass
        tracer.event("mid")
    data = tracer.to_dict()
    assert data["spans"][0]["name"] == "task"
    assert data["spans"][0]["children"][0]["name"] == "child"
    back = Tracer.from_dict(data)
    assert back.timeline() == tracer.timeline()
    assert len(back.events()) == 1


# inspection


def _sample_state():
    return {
        "memory": {
            "working": ["a", "b", "c"],
            "episodic": [{"id": 1}, {"id": 2}],
            "semantic": {
                "concepts": {"cat": {}, "dog": {}},
                "relations": [("cat", "is", "animal")],
            },
        },
        "beliefs": ["sky is blue"],
        "goals": [
            {"name": "g1", "status": "active"},
            {"name": "g2", "status": "done"},
            {"name": "g3", "status": "active"},
        ],
        "tools": ["calc", "echo"],
    }


def test_summarize_counts():
    counts = summarize(_sample_state())
    assert counts["working_items"] == 3
    assert counts["episodes"] == 2
    assert counts["concepts"] == 2
    assert counts["relations"] == 1
    assert counts["beliefs"] == 1
    assert counts["goals"]["total"] == 3
    assert counts["goals"]["by_status"] == {"active": 2, "done": 1}
    assert counts["tools"] == 2


def test_summarize_tolerates_missing_sections():
    counts = summarize({})
    assert counts["working_items"] == 0
    assert counts["goals"] == {"total": 0, "by_status": {}}
    with pytest.raises(ValueError):
        summarize("not a dict")


def test_summary_one_liner():
    line = summary(_sample_state())
    assert "working=3" in line
    assert "episodes=2" in line
    assert "concepts=2" in line
    assert "goals=3(active=2, done=1)" in line
    assert "tools=2" in line
    assert "\n" not in line


def test_query_state_dotted_lookup():
    state = _sample_state()
    assert query_state(state, "memory.semantic.concepts") == {"cat": {}, "dog": {}}
    assert query_state(state, "goals.0.name") == "g1"
    assert query_state(state, "tools.1") == "echo"


def test_query_state_helpful_key_errors():
    state = _sample_state()
    with pytest.raises(KeyError) as exc:
        query_state(state, "memory.nonexistent")
    assert "nonexistent" in str(exc.value)
    assert "semantic" in str(exc.value)
    with pytest.raises(KeyError, match="out of range"):
        query_state(state, "goals.9")
    with pytest.raises(KeyError, match="integer"):
        query_state(state, "goals.name")
    with pytest.raises(KeyError, match="leaf"):
        query_state(state, "tools.0.deeper")


# extra plugin edge cases


def test_load_plugin_returning_non_tools_errors(tmp_path):
    _write_plugin(str(tmp_path), "weird", "def register_tools():\n    return ['not', 'tools']\n")
    manager = PluginManager(str(tmp_path))
    with pytest.raises(PluginError, match="list of Tool"):
        manager.load("weird")


def test_load_plugin_factory_raising_errors(tmp_path):
    _write_plugin(str(tmp_path), "raiser", "def register_tools():\n    raise RuntimeError('nope')\n")
    manager = PluginManager(str(tmp_path))
    with pytest.raises(PluginError, match="register_tools"):
        manager.load("raiser")


def test_load_all_empty_dir(tmp_path):
    manager = PluginManager(str(tmp_path))
    assert manager.load_all(ToolRegistry()) == {}


def test_plugin_error_is_exception():
    assert issubclass(PluginError, Exception)


# extra config edge cases


def test_config_save_then_load(tmp_path):
    cfg = Config({"memory": {"working_capacity": 11}, "runtime": {"autosave": True}})
    path = str(tmp_path / "c.json")
    cfg.save(path)
    back = Config.load(path)
    assert back.get("memory.working_capacity") == 11
    assert back.get("runtime.autosave") is True


def test_config_from_dict_tolerates_unknown_keys():
    cfg = Config.from_dict({"bogus": {"x": 1}, "memory": {"working_capacity": 8}})
    assert cfg.get("memory.working_capacity") == 8
    # unknown sections are carried along, not dropped
    assert cfg.get("bogus.x") == 1


def test_config_validate_rejects_bool_as_int():
    cfg = Config({"memory": {"working_capacity": True}})
    problems = cfg.validate()
    assert any("memory.working_capacity" in p for p in problems)


def test_config_merge_accepts_new_sections():
    # merge deep-merges; unknown sections are carried along
    cfg = Config.defaults()
    merged = cfg.merge({"custom": {"x": 1}})
    assert merged.get("custom.x") == 1
    assert cfg.get("custom", None) is None


def test_config_set_writes_values():
    cfg = Config.defaults()
    cfg.set("memory.working_capacity", 21)
    assert cfg.get("memory.working_capacity") == 21
    # set also creates nested paths on demand
    cfg.set("custom.flag", True)
    assert cfg.get("custom.flag") is True


# extra persistence edge cases


def test_save_leaves_no_tmp_files(tmp_path):
    path = str(tmp_path / "s.json")
    save_state({"a": 1}, path)
    leftovers = [e for e in os.listdir(str(tmp_path)) if e.endswith(".tmp")]
    assert leftovers == []


def test_load_state_explicit_version_1(tmp_path):
    path = str(tmp_path / "v1.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"version": 1, "state": {"ok": True}}, handle)
    assert load_state(path) == {"ok": True}


def test_save_state_rejects_non_dict(tmp_path):
    with pytest.raises(StateError, match="must be a dict"):
        save_state([1, 2], str(tmp_path / "s.json"))


def test_state_store_list_ignores_non_json(tmp_path):
    directory = str(tmp_path / "saves")
    os.makedirs(directory)
    with open(os.path.join(directory, "junk.txt"), "w") as handle:
        handle.write("x")
    with open(os.path.join(directory, ".hidden.json"), "w") as handle:
        handle.write("{}")
    store = StateStore(directory)
    store.save("real", {"a": 1})
    assert [name for name, _ in store.list_saves()] == ["real"]


def test_state_store_save_overwrites(tmp_path):
    store = StateStore(str(tmp_path))
    store.save("n", {"v": 1})
    store.save("n", {"v": 2})
    assert store.load("n") == {"v": 2}


# extra tracer edge cases


def test_tracer_multiple_roots():
    tracer = Tracer()
    with tracer.start("one"):
        pass
    with tracer.start("two"):
        pass
    assert [s.name for s in tracer.roots()] == ["one", "two"]
    assert len(tracer.timeline().splitlines()) == 2


def test_tracer_event_outside_span_depth_zero():
    tracer = Tracer()
    tracer.event("lonely")
    assert tracer.events()[0]["depth"] == 0


def test_tracer_active_span():
    tracer = Tracer()
    assert tracer.active() is None
    with tracer.start("outer"):
        assert tracer.active().name == "outer"
        with tracer.start("inner"):
            assert tracer.active().name == "inner"
            assert tracer.active().is_running is True
        assert tracer.active().name == "outer"
    assert tracer.active() is None


def test_tracer_timeline_shows_attrs():
    tracer = Tracer()
    with tracer.start("task", kind="demo"):
        pass
    assert "kind='demo'" in tracer.timeline()


def test_span_from_dict_roundtrip_fields():
    tracer = Tracer()
    with tracer.start("s", x=1):
        pass
    data = tracer.to_dict()["spans"][0]
    assert data["attrs"] == {"x": 1}
    assert data["duration"] is not None
    back = Tracer.from_dict(tracer.to_dict())
    assert back.roots()[0].attrs == {"x": 1}


# extra inspection edge cases


def test_summarize_goal_without_status():
    counts = summarize({"goals": [{"name": "g"}]})
    assert counts["goals"]["by_status"] == {"unknown": 1}


def test_summary_empty_state():
    line = summary({})
    assert line == "working=0 episodes=0 concepts=0 relations=0 beliefs=0 goals=0 tools=0"


def test_query_state_on_root_list():
    state = [{"a": 1}, {"a": 2}]
    assert query_state(state, "1.a") == 2


def test_query_state_rejects_bad_path():
    with pytest.raises(KeyError):
        query_state({"a": 1}, "")
    with pytest.raises(KeyError, match="leaf"):
        query_state({"a": 1}, "a.b")
