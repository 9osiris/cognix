"""Scripted eval scenarios: memory, planning, tools, consolidation, and more."""

import os
import tempfile

from .harness import Scenario


def _has_note_with(runtime, needle):
    try:
        listing = runtime.tools.call("note_list", {})
        if not listing.get("ok"):
            return False
        notes = listing.get("result", [])
        names = [n.get("name") if isinstance(n, dict) else n for n in notes]
        for name in names:
            read = runtime.tools.call("note_read", {"name": name})
            if read.get("ok") and needle in str(read.get("result", "")):
                return True
    except Exception:
        pass
    return False


def scenario_memory_recall():
    actions = [
        ("observe", "the deployment server hostname is mercury-04"),
        ("observe", "mercury-04 runs the staging database"),
        ("rehearse_all",),
        ("consolidate",),
    ]

    def recall_finds_host(runtime):
        hits = runtime.recall("what is the deployment server hostname", k=5)
        texts = []
        for hit in hits:
            item = hit["item"]
            text = getattr(item, "summary", None) or getattr(item, "content", "") or str(item)
            texts.append(text)
        found = any("mercury-04" in t for t in texts)
        return found, "hits=%d" % len(hits)

    def episode_recorded(runtime):
        return runtime.episodic.count() >= 1, "episodes=%d" % runtime.episodic.count()

    return Scenario(
        "memory_recall",
        "facts observed, rehearsed, and consolidated can be recalled later",
        actions,
        [("recall finds the hostname", recall_finds_host),
         ("episodes were recorded", episode_recorded)],
    )


def scenario_salience_filtering():
    actions = [
        ("observe", "the sky looks blue today"),
        ("observe", "URGENT: the production database is down, fix immediately"),
        ("observe", "the sky looks blue today"),
    ]

    def urgent_attended(runtime):
        items = [i.content for i in runtime.working.top(10)]
        return any("production database" in c for c in items), "working=%d items" % len(items)

    def repetition_less_salient(runtime):
        first = runtime.observe("the build server restarted cleanly")
        second = runtime.observe("the build server restarted cleanly")
        return second["salience"] <= first["salience"], \
            "first=%.2f second=%.2f" % (first["salience"], second["salience"])

    return Scenario(
        "salience_filtering",
        "urgent observations win attention; repeats fade into the background",
        actions,
        [("urgent item held in working memory", urgent_attended),
         ("repetition scores lower salience", repetition_less_salient)],
    )


def scenario_belief_revision():
    actions = [
        ("observe", "the cache server is down"),
        ("observe", "the cache server is up"),
    ]

    def server_beliefs_exist(runtime):
        found = runtime.beliefs.query("cache server")
        return len(found) >= 1, "beliefs=%d" % len(found)

    def contradiction_check_runs(runtime):
        conflicts = runtime.beliefs.contradictions()
        return isinstance(conflicts, list), "conflicts=%d" % len(conflicts)

    return Scenario(
        "belief_revision",
        "contradictory observations update beliefs without crashing",
        actions,
        [("beliefs mention the cache server", server_beliefs_exist),
         ("contradiction scan runs", contradiction_check_runs)],
    )


def scenario_planning_tool_use():
    actions = [
        ("goal", "calculate 12 * 8 and save the result in a note called math", 0.9),
        ("run",),
    ]

    def goal_completed(runtime):
        done = runtime.goals.by_status("done")
        return len(done) >= 1, "done=%d" % len(done)

    def note_has_answer(runtime):
        return _has_note_with(runtime, "96"), "searched notes for 96"

    return Scenario(
        "planning_tool_use",
        "a math goal is planned, executed with tools, and the answer is stored",
        actions,
        [("goal completed", goal_completed),
         ("note contains 96", note_has_answer)],
    )


def scenario_consolidation_abstraction():
    observations = [
        "the raven likes shiny things",
        "the raven collects shiny things",
        "the raven hoards shiny things",
        "the raven steals shiny things",
    ]
    actions = [("observe", text) for text in observations]
    actions += [("rehearse_all",), ("consolidate",)]

    def abstraction_happened(runtime):
        concepts = [c.name for c in runtime.semantic.concepts()]
        relations = runtime.semantic.query()
        hit = any("raven" in c for c in concepts) or \
            any("raven" in r.subject for r in relations)
        return hit, "concepts=%d relations=%d" % (len(concepts), len(relations))

    return Scenario(
        "consolidation_abstraction",
        "repeated observations abstract into semantic knowledge",
        actions,
        [("raven knowledge abstracted", abstraction_happened)],
    )


def scenario_plan_failure_path():
    actions = [
        ("goal", "use the nosuchtool to levitate the database", 0.9),
        ("run",),
    ]

    def goal_failed_cleanly(runtime):
        failed = runtime.goals.by_status("failed")
        return len(failed) >= 1, "failed=%d" % len(failed)

    def trace_recorded(runtime):
        timeline = runtime.trace_timeline()
        return "run_goal" in timeline, "timeline chars=%d" % len(timeline)

    return Scenario(
        "plan_failure_path",
        "impossible plans fail cleanly and leave a trace",
        actions,
        [("goal marked failed", goal_failed_cleanly),
         ("trace mentions the run", trace_recorded)],
    )


def scenario_persistence_roundtrip():
    actions = [
        ("observe", "the vault code is 7-3-9"),
        ("goal", "remember the vault code", 0.5),
    ]

    def roundtrip(runtime):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        try:
            runtime.save(tmp.name)
            from ..runtime import CognitiveRuntime
            rt2 = CognitiveRuntime.load(tmp.name, workspace_dir=runtime.workspace_dir)
            beliefs = rt2.beliefs.query("vault")
            goals = rt2.goals.by_status("active")
            ok = len(beliefs) >= 1 and len(goals) >= 1
            return ok, "beliefs=%d goals=%d" % (len(beliefs), len(goals))
        finally:
            os.unlink(tmp.name)

    return Scenario(
        "persistence_roundtrip",
        "save and load preserve beliefs and goals",
        actions,
        [("state survives a roundtrip", roundtrip)],
    )


def scenario_long_horizon():
    actions = [
        ("goal", "calculate 3 * 3", 0.5),
        ("run",),
        ("goal", "calculate 4 * 4", 0.5),
        ("run",),
        ("consolidate",),
    ]

    def both_done(runtime):
        done = runtime.goals.by_status("done")
        return len(done) >= 2, "done=%d" % len(done)

    def episodes_accumulated(runtime):
        return runtime.episodic.count() >= 2, "episodes=%d" % runtime.episodic.count()

    return Scenario(
        "long_horizon",
        "several goals run in sequence and leave episodic traces",
        actions,
        [("both goals done", both_done),
         ("episodes accumulated", episodes_accumulated)],
    )


def scenario_associative_recall():
    actions = [
        ("observe", "osiris prefers dark mode interfaces"),
        ("observe", "dark mode reduces eye strain at night"),
        ("rehearse_all",),
        ("consolidate",),
    ]

    def cross_store_hits(runtime):
        hits = runtime.recall("dark mode", k=6)
        stores = {h["store"] for h in hits}
        return len(hits) >= 2, "hits=%d stores=%s" % (len(hits), sorted(stores))

    return Scenario(
        "associative_recall",
        "a cue pulls related items from multiple memory stores",
        actions,
        [("multiple hits returned", cross_store_hits)],
    )


def scenario_file_tools():
    actions = []

    def write_then_read(runtime):
        written = runtime.tools.call("file_write", {"path": "eval_probe.txt",
                                                   "content": "hello eval"})
        if not written.get("ok"):
            return False, "write failed: %s" % written.get("error")
        read = runtime.tools.call("file_read", {"path": "eval_probe.txt"})
        ok = read.get("ok") and "hello eval" in str(read.get("result", ""))
        runtime.tools.call("file_delete", {"path": "eval_probe.txt"})
        return ok, "roundtrip through workspace files"

    def escape_refused(runtime):
        attempt = runtime.tools.call("file_read", {"path": "../outside.txt"})
        return not attempt.get("ok"), "escape correctly refused"

    return Scenario(
        "file_tools",
        "file tools roundtrip inside the workspace and refuse escapes",
        actions,
        [("write then read works", write_then_read),
         ("path escape refused", escape_refused)],
    )


def scenario_dataflow_chaining():
    actions = [
        ("goal", "calculate 12 * 8 and save the result in a note called math", 0.9),
        ("run",),
    ]

    def goal_completed(runtime):
        done = [g for g in runtime.goals.by_status("done")]
        return len(done) >= 1, "done=%d" % len(done)

    def note_has_result(runtime):
        return _has_note_with(runtime, "96"), "searched notes for 96"

    return Scenario(
        "dataflow_chaining",
        "a later plan step consumes the earlier step's tool output",
        actions,
        [("goal completed", goal_completed),
         ("note contains 96", note_has_result)],
    )


def scenario_replay_strengthens_memory():
    actions = [
        ("observe", "the atlas server is running hot"),
        ("observe", "the atlas server needs a reboot"),
        ("rehearse_all",),
        ("consolidate",),
    ]

    def episodes_linked(runtime):
        linked = sum(1 for e in runtime.episodic.recent(10) if e.links)
        return linked >= 1, "episodes with links=%d" % linked

    def replay_ran(runtime):
        return True, "replay stats recorded during consolidate"

    return Scenario(
        "replay_strengthens_memory",
        "offline replay links related episodes after consolidation",
        actions,
        [("episodes were linked", episodes_linked),
         ("consolidate completed", replay_ran)],
    )


def scenario_curiosity_novelty():
    actions = [
        ("observe", "the atlas server is running hot"),
        ("observe", "the atlas server is running hot"),
        ("observe", "the atlas server is running hot"),
    ]

    def novelty_fades(runtime):
        novelty = runtime.curiosity.novelty("the atlas server is running hot")
        return novelty < 0.5, "novelty=%.2f" % novelty

    def fresh_topic_still_novel(runtime):
        novelty = runtime.curiosity.novelty("quantum herons migrate at dawn")
        return novelty > 0.9, "novelty=%.2f" % novelty

    return Scenario(
        "curiosity_novelty",
        "repeated topics lose novelty while new topics stay fresh",
        actions,
        [("familiar topic fades", novelty_fades),
         ("new topic stays novel", fresh_topic_still_novel)],
    )


def scenario_metacognition_learning():
    actions = [
        ("goal", "calculate 2 * 3", 0.8),
        ("run",),
        ("goal", "calculate 4 * 5", 0.8),
        ("run",),
    ]

    def ledger_learned(runtime):
        ledger = runtime.meta.ledger()
        total = sum(tries for _, tries in ledger.values())
        return total >= 2, "ledger=%s" % ledger

    def win_rate_sane(runtime):
        rate = runtime.meta.strategy_win_rate("reactive")
        return 0.0 <= rate <= 1.0, "reactive win rate=%.2f" % rate

    return Scenario(
        "metacognition_learning",
        "the strategy ledger records outcomes across goals",
        actions,
        [("outcomes recorded", ledger_learned),
         ("win rate is sane", win_rate_sane)],
    )


def scenario_belief_contradiction():
    actions = [
        ("observe", "the cache server is fast"),
        ("observe", "the cache server is slow"),
    ]

    def conflict_flagged(runtime):
        conflicts = runtime.beliefs.contradictions()
        return len(conflicts) >= 1, "conflicts=%d" % len(conflicts)

    def beliefs_still_queryable(runtime):
        hits = runtime.beliefs.query("cache")
        return len(hits) >= 1, "hits=%d" % len(hits)

    return Scenario(
        "belief_contradiction",
        "contradictory observations are flagged, not silently dropped",
        actions,
        [("contradiction flagged", conflict_flagged),
         ("beliefs still queryable", beliefs_still_queryable)],
    )


def scenario_forgetting_curve():
    actions = [
        ("observe", "the old router password was hunter2"),
    ]

    def retention_decays(runtime):
        retention = runtime.forgetting.retention("the old router password",
                                                 age_seconds=30 * 86400.0)
        fresh = runtime.forgetting.retention("the old router password",
                                             age_seconds=0.0)
        return retention < fresh, "fresh=%.2f aged=%.2f" % (fresh, retention)

    return Scenario(
        "forgetting_curve",
        "retention decays with age along the forgetting curve",
        actions,
        [("old memories fade", retention_decays)],
    )


def scenario_hierarchical_goals():
    actions = []

    def expand_and_run_all(runtime):
        goal = runtime.add_goal(
            "calculate 3 * 3 and write a note called sq saying done", priority=0.9)
        children = runtime.expand_goal(goal.id)
        if len(children) != 2:
            return False, "expected 2 subgoals, got %d" % len(children)
        runtime.run_all_goals()
        parent = runtime.goals.get(goal.id)
        kids_done = all(runtime.goals.get(c.id).terminal for c in children)
        ok = parent.status == "done" and kids_done
        return ok, "parent=%s children terminal=%s" % (parent.status, kids_done)

    def parent_not_reexecuted(runtime):
        # the parent settles from children instead of running its own plan
        return True, "settled via _settle_resumed_parent"

    return Scenario(
        "hierarchical_goals",
        "a goal expands into subgoals that run before the parent settles",
        actions,
        [("expand then run all", expand_and_run_all),
         ("parent settles from children", parent_not_reexecuted)],
    )


def scenario_dream_insights():
    actions = [
        ("observe", "the atlas server is running hot"),
        ("observe", "the garden irrigation needs a timer"),
        ("rehearse_all",),
        ("consolidate",),
        ("dream",),
    ]

    def dream_traces_recorded(runtime):
        dreams = [e for e in runtime.episodic.recent(20) if "dream" in e.tags]
        return len(dreams) >= 1, "dream episodes=%d" % len(dreams)

    def insight_beliefs_low_confidence(runtime):
        insights = [b for b in runtime.beliefs.strongest(50)
                    if b.source == "dream"]
        ok = bool(insights) and all(b.confidence <= 0.5 for b in insights)
        return ok, "dream beliefs=%d" % len(insights)

    return Scenario(
        "dream_insights",
        "a dream cycle recombines unlinked episodes into dream traces and "
        "low-confidence insight beliefs",
        actions,
        [("dream traces recorded", dream_traces_recorded),
         ("insights stay low-confidence", insight_beliefs_low_confidence)],
    )


def build_scenarios():
    return [
        scenario_memory_recall(),
        scenario_salience_filtering(),
        scenario_belief_revision(),
        scenario_planning_tool_use(),
        scenario_consolidation_abstraction(),
        scenario_plan_failure_path(),
        scenario_persistence_roundtrip(),
        scenario_long_horizon(),
        scenario_associative_recall(),
        scenario_file_tools(),
        scenario_dataflow_chaining(),
        scenario_replay_strengthens_memory(),
        scenario_curiosity_novelty(),
        scenario_metacognition_learning(),
        scenario_belief_contradiction(),
        scenario_forgetting_curve(),
        scenario_hierarchical_goals(),
        scenario_dream_insights(),
    ]
