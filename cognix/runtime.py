"""CognitiveRuntime: the main loop wiring memory, attention, perception, and agency.

One cognitive cycle looks like this:
  observe -> salience -> focus -> working memory -> beliefs
  goal -> plan -> execute -> reflect -> (replan | done)
  periodic: consolidate, decay, autosave
"""

import time
import os

from .agent.executor import Executor
from .agent.goals import GoalStack
from .agent.metacognition import MetaCognition
from .agent.planner import Planner, split_clauses
from .agent.reflection import reflect
from .attention.arousal import Arousal
from .attention.curiosity import CuriosityTracker
from .attention.focus import Focus
from .attention.salience import score_observation
from .config import Config
from .events import EventBus
from .introspection.inspect import query_state
from .introspection.inspect import summarize as _summarize_state
from .introspection.inspect import summary as _summary_text
from .introspection.trace import Tracer
from .memory.associative import AssociativeRecall
from .memory.consolidation import ConsolidationPolicy, consolidate
from .memory.episodic import EpisodicMemory
from .memory.forgetting import ForgettingCurve
from .memory.replay import ReplayPolicy, replay
from .memory.dream import DreamPolicy, dream
from .memory.semantic import SemanticMemory
from .memory.working import WorkingMemory
from .perception.beliefs import BeliefStore
from .perception.pipeline import observation_to_beliefs, parse_observation
from .persistence import SAVE_VERSION, load_state, save_state
from .plugins import PluginManager
from .tools.builtin import make_builtin_tools
from .tools.registry import ToolRegistry


def _restore(cls, data, *args, **kwargs):
    # works whether from_dict is a classmethod returning new or an instance method
    obj = cls(*args, **kwargs)
    if hasattr(obj, "from_dict") and data:
        result = obj.from_dict(data)
        return result if result is not None else obj
    return obj


class _Introspector:
    # live view over a runtime: summary() and dotted-path query_state()
    def __init__(self, runtime):
        self._runtime = runtime

    def summary(self):
        return _summary_text(self._runtime.state_dict())

    def summarize(self):
        return _summarize_state(self._runtime.state_dict())

    def query(self, path):
        return query_state(self._runtime.state_dict(), path)


class CognitiveRuntime:
    def __init__(self, config=None, workspace_dir=None, plugin_dir=None, now=None):
        self._now = now or time.time
        self.config = config or Config.defaults()
        self.workspace_dir = workspace_dir or self.config.get("tools.workspace_dir", ".")

        mem_cfg = "memory."
        self.working = WorkingMemory(
            capacity=self.config.get(mem_cfg + "working_capacity", 7),
            decay_rate=self.config.get(mem_cfg + "decay_rate", 0.08),
            threshold=self.config.get(mem_cfg + "activation_threshold", 0.05),
            now=self._now,
        )
        self.episodic = EpisodicMemory(now=self._now)
        self.semantic = SemanticMemory()
        self.forgetting = ForgettingCurve()
        self.assoc = AssociativeRecall(self.working, self.episodic, self.semantic)

        self.focus = Focus(
            bandwidth=self.config.get("attention.bandwidth", 3),
            threshold=self.config.get("attention.salience_threshold", 0.3),
        )
        self.arousal = Arousal(
            baseline=self.config.get("attention.arousal_baseline", 0.5),
            now=self._now,
        )
        self.curiosity = CuriosityTracker(
            bonus_scale=self.config.get("attention.curiosity_bonus_scale", 0.15),
            boredom_threshold=self.config.get("attention.curiosity_boredom_threshold", 0.8),
            now=self._now,
        )
        self.replay_policy = ReplayPolicy(
            budget=self.config.get("memory.replay_budget", 5),
            min_salience=self.config.get("memory.replay_min_salience", 0.4),
        )
        self.dream_policy = DreamPolicy(
            budget=self.config.get("memory.dream_budget", 8),
            dreams=self.config.get("memory.dreams_per_cycle", 3),
            min_salience=self.config.get("memory.dream_min_salience", 0.4),
            seed=self.config.get("memory.dream_seed"),
        )
        self.beliefs = BeliefStore()

        self.events = EventBus(
            max_log=self.config.get("runtime.event_log_size", 200),
            enabled=self.config.get("runtime.events_enabled", True),
            now=self._now,
        )
        # belief assertions announce themselves on the bus
        self.beliefs._emit = self.events.emit

        self.tools = ToolRegistry()
        for tool in make_builtin_tools(self.workspace_dir):
            self.tools.register(tool)
        if not self.config.get("tools.shell_enabled", True):
            # keep the tool listed but neutered when disabled
            pass
        if plugin_dir or self.config.get("tools.plugin_dir"):
            pdir = plugin_dir or self.config.get("tools.plugin_dir")
            if not os.path.isabs(pdir):
                pdir = os.path.join(self.workspace_dir, pdir)
            self.plugins = PluginManager(pdir)
            try:
                self.plugins.load_all(self.tools)
            except Exception:
                self.plugins = None
        else:
            self.plugins = None

        self.goals = GoalStack(now=self._now)
        self.planner = Planner(self.tools)
        self.executor = Executor(
            self.tools,
            max_retries=self.config.get("agent.max_retries", 2),
            step_timeout=self.config.get("agent.step_timeout", 30.0),
        )
        self.meta = MetaCognition()
        self.tracer = Tracer()
        self.introspect = _Introspector(self)

        self.cycles = 0
        self._since_consolidation = 0

    def now(self):
        return self._now()

    # perception

    def observe(self, raw, source="user"):
        """Run one observation through perception, attention, and memory."""
        text = raw.strip()
        report = {"text": text, "source": source, "attended": False}
        if not text:
            return report
        self.events.emit("observation.received", source=source,
                         chars=len(text))
        with self.tracer.start("observe", source=source):
            parsed = parse_observation(text)
            report["parsed"] = parsed
            salience, signals = score_observation(
                text,
                goals=self.goals.active(),
                beliefs=self.beliefs,
                episodic=self.episodic,
                now=self.now(),
            )
            report["salience"] = salience
            report["signals"] = [{"name": s.name, "score": s.score, "reason": s.reason} for s in signals]
            curiosity_bonus = self.curiosity.bonus(text)
            salience = min(1.0, salience + curiosity_bonus)
            report["salience"] = salience
            report["curiosity_bonus"] = curiosity_bonus
            self.curiosity.expose(text)
            self.arousal.update(salience)
            selected = self.focus.select([(text, salience)])
            if not selected:
                report["reason"] = "below attention threshold"
                self.events.emit("observation.ignored",
                                 source=source, salience=round(salience, 4))
                return report
            report["attended"] = True
            item = self.working.add(text, kind=parsed.get("intent", "inform"),
                                   source=source, activation=max(0.2, salience))
            report["working_id"] = item.id
            self.events.emit("observation.attended", source=source,
                             salience=round(salience, 4),
                             working_id=item.id)
            added = []
            for proposition, confidence in observation_to_beliefs(parsed):
                belief = self.beliefs.assert_belief(proposition, confidence, source)
                added.append(belief.proposition)
            report["beliefs_added"] = added
            if salience >= 0.7:
                episode = self.episodic.record(
                    summary=text[:160],
                    detail={"intent": parsed.get("intent"), "entities": parsed.get("entities", [])},
                    tags=[parsed.get("intent", "inform")],
                    salience=salience,
                    source=source,
                )
                report["episode_id"] = episode.id
                self.events.emit("episode.recorded", episode_id=episode.id,
                                 salience=round(salience, 4))
        return report

    # maintenance

    def cycle(self, dt=1.0):
        """Background tick: decay, arousal settling, periodic consolidation."""
        self.cycles += 1
        self.working.tick(dt)
        self.arousal.tick(dt)
        self._since_consolidation += 1
        if self._since_consolidation >= self.config.get("runtime.consolidate_every", 10):
            self._since_consolidation = 0
            return self.consolidate()
        return None

    def consolidate(self):
        policy = ConsolidationPolicy(
            activation_threshold=self.config.get("memory.consolidation_threshold", 0.6),
            min_rehearsals=self.config.get("memory.min_rehearsals", 2),
            semantic_support=self.config.get("memory.semantic_support", 3),
            max_age_days=self.config.get("memory.episodic_max_age_days", 30),
        )
        with self.tracer.start("consolidate"):
            stats = consolidate(self.working, self.episodic, self.semantic,
                                now=self.now(), policy=policy)
        pruned = self.episodic.forget(
            max_age_days=self.config.get("memory.episodic_max_age_days", 30))
        stats["episodes_pruned"] = pruned
        with self.tracer.start("replay"):
            replay_stats = replay(self.episodic, self.semantic, self.working,
                                  policy=self.replay_policy, now=self._now)
        stats["replay"] = replay_stats
        self.events.emit("memory.consolidated", stats=stats)
        self.events.emit("memory.replayed", stats=replay_stats)
        return stats

    def dream(self):
        """Run one dream cycle over the episodic store.

        Dreaming is separate from consolidation: it runs on demand or on a
        schedule, recombines unlinked salient episodes into dream traces,
        and proposes low-confidence insight beliefs and semantic relations.
        """
        with self.tracer.start("dream"):
            stats = dream(self.episodic, self.semantic, self.beliefs,
                          policy=self.dream_policy, now=self._now)
        self.events.emit("memory.dreamed", dreams=stats.get("dreams", 0),
                         insights=stats.get("insights", 0))
        return stats

    # agency

    def add_goal(self, description, priority=0.5, **kwargs):
        goal = self.goals.push(description, priority=priority, **kwargs)
        goal.status = "active"
        self.events.emit("goal.added", goal_id=goal.id,
                         priority=priority,
                         description=description[:120])
        return goal

    def expand_goal(self, goal_id):
        """Split a goal into subgoals via clause splitting.

        The parent suspends while children run and resumes automatically
        when they are all terminal. Returns the child goals (empty when
        the goal has fewer than two clauses).
        """
        goal = self.goals.get(goal_id)
        if goal is None:
            return []
        clauses = split_clauses(goal.description)
        if len(clauses) < 2:
            return []
        with self.tracer.start("expand_goal", goal=goal.description[:60],
                               clauses=len(clauses)):
            children = self.goals.decompose(goal.id, clauses)
        self.events.emit("goal.expanded", goal_id=goal.id,
                         children=[c.id for c in children])
        return children

    def _settle_resumed_parent(self, goal):
        """Complete a resumed parent from its children's outcomes.

        Called when a parent suspended by expand_goal becomes active
        again: if every child is terminal, the parent is marked done
        (or failed when any child failed) without re-executing it.
        Returns True when the parent was settled.
        """
        children = self.goals.children(goal.id)
        if not children or any(not c.terminal for c in children):
            return False
        failed = [c for c in children if c.status == "failed"]
        summary = "; ".join(
            "%s: %s" % (c.description[:50], c.status) for c in children
        )
        if failed:
            self.goals.fail(goal.id, reason="%d/%d subgoals failed: %s" % (
                len(failed), len(children), summary[:200]))
            self.events.emit("goal.failed", goal_id=goal.id,
                             settled_parent=True,
                             failed_children=len(failed))
        else:
            self.goals.complete(goal.id, outcome="subgoals done: %s" % summary[:200])
            self.events.emit("goal.completed", goal_id=goal.id,
                             settled_parent=True)
        return True

    def run_all_goals(self, max_goals=None):
        """Run active goals in priority order until none remain.

        Resumed parents whose children all finished are settled from
        the children's outcomes instead of being re-executed.
        """
        results = []
        while True:
            goal = self.goals.top()
            if goal is None:
                break
            if max_goals is not None and len(results) >= max_goals:
                break
            if self._settle_resumed_parent(goal):
                results.append({"ok": goal.status == "done",
                                "goal_id": goal.id, "settled_parent": True})
                continue
            results.append(self.run_goal(goal.id))
        return results

    def run_goal(self, goal_id=None, max_attempts=None):
        """Plan, execute, reflect, and replan until the goal resolves."""
        max_attempts = max_attempts or self.config.get("agent.max_attempts", 3)
        goal = self.goals.get(goal_id) if goal_id else self.goals.top()
        if goal is None:
            return {"ok": False, "error": "no active goal"}
        if goal.status == "proposed":
            goal.status = "active"
        default_strategy = self.config.get("agent.default_strategy", "auto")
        past_failures = 0
        plan = None
        with self.tracer.start("run_goal", goal=goal.description[:60]):
            for attempt in range(max_attempts):
                if plan is None:
                    strategy = self.meta.choose_strategy(goal, self.beliefs, past_failures)
                    if default_strategy != "auto":
                        strategy = default_strategy
                    with self.tracer.start("plan", strategy=strategy, attempt=attempt):
                        plan = self.planner.plan(goal, self.beliefs, strategy=strategy)
                    issues = self.planner.validate(plan)
                    if issues:
                        self.goals.fail(goal.id, reason="invalid plan: " + "; ".join(issues))
                        self.events.emit("goal.failed", goal_id=goal.id,
                                         reason="invalid plan")
                        return {"ok": False, "error": "invalid plan", "issues": issues}
                    self.events.emit("plan.started", goal_id=goal.id,
                                     strategy=strategy, attempt=attempt,
                                     steps=len(plan.steps))
                with self.tracer.start("execute", attempt=attempt):
                    trace = self.executor.run_plan(
                        plan, on_step=self._step_event)
                reflection = reflect(trace, goal)
                self._learn_from_reflection(reflection, goal, trace)
                self.meta.record_outcome(plan.strategy, reflection["success"])
                if reflection["success"]:
                    self.goals.complete(goal.id, outcome=reflection["summary"])
                    self.events.emit("goal.completed", goal_id=goal.id,
                                     attempts=attempt + 1,
                                     steps=len(trace.results))
                    return {"ok": True, "goal_id": goal.id, "attempts": attempt + 1,
                            "steps": len(trace.results), "summary": reflection["summary"]}
                past_failures += 1
                if not reflection["replan_needed"] or attempt == max_attempts - 1:
                    break
                failed = next((r for r in trace.results if not r.ok), None)
                failure = {
                    "step_id": failed.step_id if failed else None,
                    "error": failed.error if failed else "",
                    "kind": reflection["failure_kind"] or "tool_error",
                }
                with self.tracer.start("replan", attempt=attempt):
                    plan = self.planner.replan(plan, failure, self.beliefs)
                self.events.emit("plan.replanned", goal_id=goal.id,
                                 attempt=attempt,
                                 failure_kind=failure["kind"])
                if plan.status == "failed":
                    break
        self.goals.fail(goal.id, reason="exhausted attempts")
        self.events.emit("goal.failed", goal_id=goal.id,
                         reason="exhausted attempts", attempts=attempt + 1)
        return {"ok": False, "goal_id": goal.id, "attempts": attempt + 1,
                "error": "exhausted attempts"}

    def _step_event(self, result):
        # on_step callback for the executor: announce each tool step
        self.events.emit("tool.step_finished", step_id=result.step_id,
                         tool=result.tool, ok=result.ok,
                         duration=round(result.duration, 3))

    def _learn_from_reflection(self, reflection, goal, trace):
        # turn the outcome into durable memory
        summary = "goal '%s': %s" % (goal.description[:80], reflection["summary"])
        self.episodic.record(
            summary=summary,
            detail={"success": reflection["success"], "steps": len(trace.results)},
            tags=["reflection", "goal"],
            salience=0.8 if reflection["success"] else 0.6,
            source="reflection",
        )
        for lesson in reflection.get("lessons", []):
            self.beliefs.assert_belief("lesson: " + lesson.text,
                                       lesson.confidence, "reflection")
        if reflection.get("lessons"):
            self.events.emit("reflection.lessons", goal_id=goal.id,
                             success=reflection["success"],
                             lessons=[l.text[:120]
                                      for l in reflection["lessons"]])
        self.working.add(summary, kind="reflection", source="reflection",
                         activation=0.8)

    # recall and inspection

    def recall(self, query, k=5):
        return self.assoc.cue(query, k=k)

    def trace_timeline(self):
        return self.tracer.timeline()

    # persistence

    def state_dict(self):
        return {
            "version": SAVE_VERSION,
            "cycles": self.cycles,
            "working": self.working.to_dict(),
            "episodic": self.episodic.to_dict(),
            "semantic": self.semantic.to_dict(),
            "beliefs": self.beliefs.to_dict(),
            "goals": self.goals.to_dict(),
            "arousal": self.arousal.to_dict(),
            "curiosity": self.curiosity.to_dict(),
            "replay_policy": self.replay_policy.to_dict(),
            "dream_policy": self.dream_policy.to_dict(),
            "strategy_ledger": self.meta.ledger_to_dict(),
            "trace": self.tracer.to_dict(),
            "events": self.events.to_dict(),
            "config": self.config.to_dict(),
        }

    def save(self, path=None):
        path = path or self.config.get("runtime.save_path", "cognix_state.json")
        save_state(self.state_dict(), path)
        return path

    @classmethod
    def load(cls, path, workspace_dir=None, now=None):
        data = load_state(path)
        config = Config.from_dict(data.get("config", {}))
        rt = cls(config=config, workspace_dir=workspace_dir, now=now)
        rt.cycles = data.get("cycles", 0)
        rt.working = _restore(WorkingMemory, data.get("working"), now=rt._now)
        rt.episodic = _restore(EpisodicMemory, data.get("episodic"), now=rt._now)
        rt.semantic = _restore(SemanticMemory, data.get("semantic"))
        rt.beliefs = _restore(BeliefStore, data.get("beliefs"))
        rt.goals = _restore(GoalStack, data.get("goals"), now=rt._now)
        rt.arousal = _restore(Arousal, data.get("arousal"), now=rt._now)
        rt.curiosity = _restore(CuriosityTracker, data.get("curiosity"),
                               now=rt._now)
        if data.get("replay_policy"):
            rt.replay_policy = ReplayPolicy.from_dict(data["replay_policy"])
        if data.get("dream_policy"):
            rt.dream_policy = DreamPolicy.from_dict(data["dream_policy"])
        rt.meta.ledger_from_dict(data.get("strategy_ledger"))
        rt.assoc = AssociativeRecall(rt.working, rt.episodic, rt.semantic)
        if data.get("trace"):
            rt.tracer = _restore(Tracer, data["trace"])
        if data.get("events"):
            rt.events = EventBus.from_dict(data["events"], now=rt._now)
        # restored subsystems lose their emit hooks; reattach
        rt.beliefs._emit = rt.events.emit
        return rt
