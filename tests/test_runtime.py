"""Integration tests for the cognitive runtime: cycles, goals, persistence."""

import json
import os
import tempfile

import pytest

from cognix.config import Config
from cognix.runtime import CognitiveRuntime


@pytest.fixture
def rt():
    workspace = tempfile.mkdtemp(prefix="cognix_test_")
    return CognitiveRuntime(workspace_dir=workspace)


def test_observe_attends_salient(rt):
    report = rt.observe("URGENT: the database is on fire, fix it now")
    assert report["attended"] is True
    assert report["salience"] > 0.3
    assert len(rt.working.top(5)) >= 1


def test_observe_empty_is_ignored(rt):
    report = rt.observe("   ")
    assert report["attended"] is False


def test_observe_builds_beliefs(rt):
    rt.observe("the cache server is down")
    found = rt.beliefs.query("cache")
    assert len(found) >= 1


def test_repeated_observation_loses_salience(rt):
    first = rt.observe("the build finished successfully")
    second = rt.observe("the build finished successfully")
    assert second["salience"] <= first["salience"]


def test_run_goal_with_calc(rt):
    goal = rt.add_goal("calculate 6 * 7", priority=0.9)
    result = rt.run_goal(goal.id)
    assert result["ok"] is True
    assert rt.goals.get(goal.id).status == "done"


def test_run_goal_no_goal(rt):
    result = rt.run_goal()
    assert result["ok"] is False


def test_run_goal_impossible_tool_fails(rt):
    goal = rt.add_goal("use the nosuchtool to levitate things", priority=0.9)
    result = rt.run_goal(goal.id)
    assert result["ok"] is False
    assert rt.goals.get(goal.id).status == "failed"


def test_cycle_and_consolidate(rt):
    rt.observe("the raven likes shiny things")
    rt.observe("the raven collects shiny things")
    for item in rt.working.top(20):
        rt.working.rehearse(item.id)
        rt.working.rehearse(item.id)
    stats = rt.consolidate()
    assert stats["episodes_added"] >= 1


def test_recall_finds_observation(rt):
    rt.observe("the vault code is 7-3-9")
    for item in rt.working.top(20):
        rt.working.rehearse(item.id)
    rt.consolidate()
    hits = rt.recall("vault code", k=5)
    texts = []
    for hit in hits:
        item = hit["item"]
        texts.append(getattr(item, "summary", None) or getattr(item, "content", ""))
    assert any("7-3-9" in t for t in texts)


def test_save_load_roundtrip(rt):
    rt.observe("the lighthouse keeper is named mara")
    rt.add_goal("remember mara", priority=0.4)
    tmp = os.path.join(tempfile.mkdtemp(), "state.json")
    rt.save(tmp)
    assert os.path.exists(tmp)
    rt2 = CognitiveRuntime.load(tmp, workspace_dir=rt.workspace_dir)
    assert len(rt2.beliefs.query("mara")) >= 1
    assert len(rt2.goals.by_status("active")) >= 1


def test_saved_state_is_versioned_json(rt):
    tmp = os.path.join(tempfile.mkdtemp(), "state.json")
    rt.save(tmp)
    with open(tmp, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["version"] >= 1
    assert "working" in data["state"] and "episodic" in data["state"] \
        and "semantic" in data["state"]


def test_trace_timeline_mentions_run(rt):
    goal = rt.add_goal("calculate 2 + 2", priority=0.9)
    rt.run_goal(goal.id)
    timeline = rt.trace_timeline()
    assert "run_goal" in timeline


def test_config_flows_into_runtime():
    cfg = Config.defaults()
    cfg.set("memory.working_capacity", 3)
    workspace = tempfile.mkdtemp(prefix="cognix_test_")
    rt = CognitiveRuntime(config=cfg, workspace_dir=workspace)
    assert rt.working.capacity == 3


def test_arousal_rises_on_urgent(rt):
    before = rt.arousal.level()
    rt.observe("URGENT critical failure in production, act now")
    assert rt.arousal.level() >= before


def test_batch_script(tmp_path):
    from cognix.batch import run_batch_file
    workspace = tempfile.mkdtemp(prefix="cognix_test_")
    rt = CognitiveRuntime(workspace_dir=workspace)
    script = tmp_path / "demo.cx"
    script.write_text(
        "observe: the demo server is called atlas\n"
        "goal: 0.8 calculate 5 * 5\n"
        "run\n"
        "assert: goal-done\n",
        encoding="utf-8",
    )
    results = run_batch_file(rt, str(script))
    assert all(ok for _, ok, _ in results), results


def test_expand_goal_splits_into_subgoals(rt):
    goal = rt.add_goal("calculate 2 * 3 and write a note called math saying done")
    children = rt.expand_goal(goal.id)
    assert len(children) == 2
    assert rt.goals.get(goal.id).status == "suspended"
    assert all(c.parent_id == goal.id for c in children)


def test_expand_goal_single_clause_returns_empty(rt):
    goal = rt.add_goal("calculate 2 * 3")
    assert rt.expand_goal(goal.id) == []
    assert rt.goals.get(goal.id).status == "active"


def test_expand_goal_unknown_id(rt):
    assert rt.expand_goal("goal-999") == []


def test_run_all_goals_settles_expanded_parent(rt):
    goal = rt.add_goal("calculate 2 * 3 and write a note called math saying done")
    children = rt.expand_goal(goal.id)
    assert len(children) == 2
    results = rt.run_all_goals()
    assert rt.goals.get(goal.id).status == "done"
    assert all(rt.goals.get(c.id).terminal for c in children)
    assert any(r.get("settled_parent") for r in results)


def test_run_all_goals_runs_in_priority_order(rt):
    low = rt.add_goal("calculate 1 * 1", priority=0.2)
    high = rt.add_goal("calculate 2 * 2", priority=0.9)
    order = []

    def fake_run(gid=None, **kwargs):
        order.append(gid)
        rt.goals.complete(gid, outcome="mocked")
        return {"ok": True, "goal_id": gid}

    original = rt.run_goal
    rt.run_goal = fake_run
    try:
        rt.run_all_goals()
    finally:
        rt.run_goal = original
    assert order[0] == high.id
    assert order[1] == low.id


def test_run_all_goals_empty(rt):
    assert rt.run_all_goals() == []


def test_run_all_goals_respects_max(rt):
    rt.add_goal("calculate 1 * 1", priority=0.2)
    rt.add_goal("calculate 2 * 2", priority=0.9)
    results = rt.run_all_goals(max_goals=1)
    assert len(results) == 1


def test_curiosity_bonus_applied_on_observe(rt):
    report = rt.observe("an entirely unprecedented quantum heron event")
    assert "curiosity_bonus" in report
    assert report["curiosity_bonus"] > 0


def test_curiosity_persists_across_save_load(rt, tmp_path):
    rt.observe("the atlas server is running hot")
    path = str(tmp_path / "state.json")
    rt.save(path)
    rt2 = CognitiveRuntime.load(path, workspace_dir=rt.workspace_dir)
    assert rt2.curiosity.familiarity("the atlas server is running hot") > 0


def test_consolidate_runs_replay(rt):
    rt.observe("the atlas server is running hot")
    rt.observe("the atlas server needs a reboot")
    for item in rt.working.top(20):
        rt.working.rehearse(item.id)
    stats = rt.consolidate()
    assert "replay" in stats
    assert stats["replay"]["replayed"] >= 1


def test_file_search_tool(rt):
    rt.tools.call("file_write", {"path": "search_a.txt", "content": "hello world\n"})
    rt.tools.call("file_write", {"path": "search_b.txt", "content": "nothing\n"})
    result = rt.tools.call("file_search", {"pattern": "hello"})
    assert result["ok"]
    hits = result["result"]
    assert len(hits) == 1
    assert hits[0]["path"] == "search_a.txt"
    assert hits[0]["line"] == 1


def test_file_search_bad_pattern_is_error(rt):
    result = rt.tools.call("file_search", {"pattern": "(["})
    assert not result["ok"]


def test_file_search_stays_in_workspace(rt):
    result = rt.tools.call("file_search", {"pattern": "x", "dir": ".."})
    assert not result["ok"]


def test_repl_new_commands(rt, capsys):
    from cognix.repl import REPL
    repl = REPL(rt)
    repl.dispatch("/curiosity")
    repl.dispatch("/ledger")
    repl.dispatch("/expand")
    repl.dispatch("/runall")
    out = capsys.readouterr().out
    assert "boredom=" in out
    assert "no strategy outcomes" in out
    assert "usage: /expand <goal_id>" in out


def test_repl_replay_and_expand(rt, capsys):
    from cognix.repl import REPL
    repl = REPL(rt)
    goal = rt.add_goal("calculate 2 * 3 and write a note called m saying done")
    repl.dispatch("/expand %s" % goal.id)
    out = capsys.readouterr().out
    assert "calculate 2 * 3" in out
    repl.dispatch("/replay")
    out = capsys.readouterr().out
    assert "replayed=" in out


def test_settle_parent_fails_when_child_fails(rt):
    goal = rt.add_goal("calculate 2 * 3 and levitate the database with nosuchtool")
    children = rt.expand_goal(goal.id)
    assert len(children) == 2
    rt.run_all_goals()
    parent = rt.goals.get(goal.id)
    assert parent.status == "failed"
    assert "subgoals failed" in (parent.outcome or "")


def test_parent_not_settled_while_child_active(rt):
    goal = rt.add_goal("calculate 2 * 3 and calculate 3 * 4")
    children = rt.expand_goal(goal.id)
    assert not rt._settle_resumed_parent(goal)
    assert rt.goals.get(goal.id).status == "suspended"


def test_cli_runall(tmp_path, capsys):
    from cognix.cli import main
    rc = main(["runall", "calculate 1 * 1", "calculate 2 * 2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("ok=True") == 2


def test_cli_runall_bad_goal_fails(capsys):
    from cognix.cli import main
    rc = main(["runall", "levitate the database with nosuchtool"])
    assert rc == 1


def test_repl_curiosity_with_text(rt, capsys):
    from cognix.repl import REPL
    REPL(rt).dispatch("/curiosity the quantum heron landed")
    out = capsys.readouterr().out
    assert "novelty=" in out
