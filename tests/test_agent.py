"""Tests for the cognix agent loop subsystem."""

import ast

import pytest

from cognix.agent import (
    Goal,
    GoalStack,
    Plan,
    PlanStep,
    Planner,
    ExecutionTrace,
    Executor,
    StepResult,
    Lesson,
    LessonBook,
    reflect,
    MetaCognition,
    classify_error,
)


# fake tool registry with the exact interface the agent code expects


def _safe_eval(expr):
    tree = ast.parse(expr, mode="eval")
    allowed = {ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
               ast.USub, ast.UAdd, ast.Load}
    for node in ast.walk(tree):
        if type(node) not in allowed:
            raise ValueError("bad expression")
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}})


def _calc_fn(files):
    def run(args):
        try:
            return {"ok": True, "result": _safe_eval(args["expression"]),
                    "error": ""}
        except Exception as exc:
            return {"ok": False, "result": None,
                    "error": "calc failed: %s" % exc}
    return run


def _read_fn(files):
    def run(args):
        path = args.get("path")
        if path in files:
            return {"ok": True, "result": files[path], "error": ""}
        return {"ok": False, "result": None,
                "error": "file not found: %s" % path}
    return run


def _write_fn(files):
    def run(args):
        if "path" not in args or "content" not in args:
            return {"ok": False, "result": None,
                    "error": "missing required arg 'content'"}
        files[args["path"]] = args["content"]
        return {"ok": True, "result": "wrote %d bytes" % len(args["content"]),
                "error": ""}
    return run


def _summarize_fn(files):
    def run(args):
        text = args.get("text", "")
        first = text.split(".")[0].strip()
        return {"ok": True, "result": first, "error": ""}
    return run


def _search_fn(files):
    def run(args):
        return {"ok": True,
                "result": ["result for %s" % args.get("query", "")],
                "error": ""}
    return run


SCHEMAS = [
    {"name": "calc",
     "description": "evaluate a math expression and return the number",
     "params": {"expression": {"type": "string",
                               "description": "math expression to evaluate",
                               "required": True}}},
    {"name": "file_read",
     "description": "read a file from disk and return its text contents",
     "params": {"path": {"type": "string",
                         "description": "path of the file to read",
                         "required": True}}},
    {"name": "file_write",
     "description": "write text content to a file on disk",
     "params": {"path": {"type": "string",
                         "description": "destination file path",
                         "required": True},
                "content": {"type": "string",
                            "description": "text content to write",
                            "required": True}}},
    {"name": "summarize",
     "description": "produce a short summary of the given text",
     "params": {"text": {"type": "string",
                         "description": "text to summarize",
                         "required": True},
                "max_sentences": {"type": "number",
                                  "description": "max sentences in the summary",
                                  "required": False}}},
    {"name": "search",
     "description": "search the web for information about a query",
     "params": {"query": {"type": "string",
                          "description": "search query",
                          "required": True}}},
]


class FakeRegistry:
    """Same interface as the real ToolRegistry."""

    def __init__(self):
        self.files = {"/tmp/report.txt":
                      "quarterly revenue up 12 percent. costs down 3 percent."}
        self.calls = []
        self.failures = {}
        self._tools = {
            "calc": (SCHEMAS[0], _calc_fn(self.files)),
            "file_read": (SCHEMAS[1], _read_fn(self.files)),
            "file_write": (SCHEMAS[2], _write_fn(self.files)),
            "summarize": (SCHEMAS[3], _summarize_fn(self.files)),
            "search": (SCHEMAS[4], _search_fn(self.files)),
        }

    def get(self, name):
        entry = self._tools.get(name)
        return entry[1] if entry else None

    def names(self):
        return list(self._tools)

    def schemas(self):
        return [schema for schema, _ in self._tools.values()]

    def call(self, name, args):
        self.calls.append((name, dict(args)))
        entry = self._tools.get(name)
        if entry is None:
            return {"ok": False, "result": None,
                    "error": "unknown tool %r" % name}
        remaining = self.failures.get(name, 0)
        if remaining > 0:
            self.failures[name] = remaining - 1
            return {"ok": False, "result": None,
                    "error": "temporary tool error"}
        _, fn = entry
        return fn(args)


class FakeBelief:
    def __init__(self, proposition, confidence):
        self.proposition = proposition
        self.confidence = confidence


class FakeBeliefs:
    def __init__(self, beliefs=()):
        self._beliefs = list(beliefs)

    def query(self, keyword):
        kw = keyword.lower()
        return [b for b in self._beliefs if kw in b.proposition.lower()]


@pytest.fixture
def registry():
    return FakeRegistry()


@pytest.fixture
def beliefs():
    return FakeBeliefs()


@pytest.fixture
def planner(registry):
    return Planner(registry)


@pytest.fixture
def executor(registry):
    return Executor(registry)


@pytest.fixture
def meta(registry):
    return MetaCognition(tools=registry)


# goals


def test_goal_defaults():
    goal = Goal("do the thing")
    assert goal.status == "active"
    assert goal.priority == 0.5
    assert goal.parent_id is None
    assert goal.deadline is None
    assert goal.metadata == {}
    assert goal.outcome == ""
    assert not goal.terminal
    assert goal.id.startswith("goal-")


def test_goal_priority_clamped():
    assert Goal("x", priority=5.0).priority == 1.0
    assert Goal("x", priority=-2.0).priority == 0.0


def test_goal_bad_status_raises():
    with pytest.raises(ValueError):
        Goal("x", status="bogus")


def test_stack_push_get():
    stack = GoalStack()
    goal = stack.push("write tests", priority=0.9)
    assert stack.get(goal.id) is goal
    assert stack.get("nope") is None
    assert len(stack) == 1


def test_active_sorted_by_priority_desc():
    stack = GoalStack()
    stack.push("low", priority=0.1)
    stack.push("high", priority=0.9)
    stack.push("mid", priority=0.5)
    assert [g.priority for g in stack.active()] == [0.9, 0.5, 0.1]


def test_top_returns_highest_or_none():
    stack = GoalStack()
    assert stack.top() is None
    low = stack.push("low", priority=0.2)
    high = stack.push("high", priority=0.8)
    assert stack.top() is high
    stack.suspend(high.id)
    assert stack.top() is low


def test_complete_and_fail():
    stack = GoalStack()
    goal = stack.push("task")
    assert stack.complete(goal.id, outcome="shipped") is True
    assert goal.status == "done"
    assert goal.outcome == "shipped"
    assert goal.terminal
    other = stack.push("other")
    assert stack.fail(other.id, reason="blocked") is True
    assert other.status == "failed"
    assert other.outcome == "blocked"


def test_complete_missing_or_terminal_returns_false():
    stack = GoalStack()
    assert stack.complete("missing") is False
    assert stack.fail("missing") is False
    goal = stack.push("task")
    stack.complete(goal.id)
    assert stack.complete(goal.id) is False
    assert stack.fail(goal.id) is False


def test_suspend_resume():
    stack = GoalStack()
    goal = stack.push("task")
    assert stack.suspend(goal.id) is True
    assert goal.status == "suspended"
    assert stack.suspend(goal.id) is False
    assert stack.resume(goal.id) is True
    assert goal.status == "active"
    assert stack.resume(goal.id) is False
    assert stack.resume("missing") is False


def test_decompose_links_children_and_suspends_parent():
    stack = GoalStack()
    parent = stack.push("ship feature", priority=0.9)
    children = stack.decompose(parent.id, ["write code", "write tests"])
    assert len(children) == 2
    assert all(c.parent_id == parent.id for c in children)
    assert all(c.status == "active" for c in children)
    assert parent.status == "suspended"
    assert stack.children(parent.id) == children


def test_decompose_resumes_parent_when_children_done():
    stack = GoalStack()
    parent = stack.push("ship feature")
    kids = stack.decompose(parent.id, ["a", "b"])
    stack.complete(kids[0].id)
    assert parent.status == "suspended"
    stack.complete(kids[1].id)
    assert parent.status == "active"


def test_decompose_resumes_parent_when_children_failed():
    stack = GoalStack()
    parent = stack.push("ship feature")
    kids = stack.decompose(parent.id, ["a", "b"])
    stack.fail(kids[0].id, reason="nope")
    assert parent.status == "suspended"
    stack.fail(kids[1].id, reason="nope")
    assert parent.status == "active"


def test_decompose_missing_or_terminal_parent():
    stack = GoalStack()
    assert stack.decompose("missing", ["a"]) == []
    parent = stack.push("done already")
    stack.complete(parent.id)
    assert stack.decompose(parent.id, ["a"]) == []


def test_decompose_custom_priorities():
    stack = GoalStack()
    parent = stack.push("p", priority=0.7)
    kids = stack.decompose(parent.id, ["a", "b"], priorities=[0.3, 0.9])
    assert [k.priority for k in kids] == [0.3, 0.9]


def test_overdue():
    stack = GoalStack(now=lambda: 1000.0)
    stack.push("late", deadline=900.0)
    stack.push("future", deadline=1100.0)
    done = stack.push("was late", deadline=900.0)
    stack.complete(done.id)
    overdue = stack.overdue()
    assert len(overdue) == 1
    assert overdue[0].description == "late"
    assert stack.overdue(now=800.0) == []


def test_by_status():
    stack = GoalStack()
    a = stack.push("a")
    b = stack.push("b")
    stack.suspend(b.id)
    assert stack.by_status("active") == [a]
    assert stack.by_status("suspended") == [b]


def test_goal_dict_roundtrip():
    goal = Goal("x", priority=0.7, status="suspended", parent_id="goal-1",
                deadline=123.0, metadata={"k": "v"}, outcome="wip")
    clone = Goal.from_dict(goal.to_dict())
    assert clone.to_dict() == goal.to_dict()


def test_stack_dict_roundtrip_keeps_ids_unique():
    stack = GoalStack()
    parent = stack.push("parent")
    stack.decompose(parent.id, ["kid1", "kid2"])
    clone = GoalStack.from_dict(stack.to_dict())
    assert len(clone) == 3
    assert clone.get(parent.id).status == "suspended"
    fresh = clone.push("new", priority=0.99)
    assert fresh.id not in {g.id for g in stack._goals.values()}
    assert clone.top().description == "new"


# planner


def test_split_clauses_then():
    from cognix.agent.planner import split_clauses
    assert split_clauses("read config then summarize it") == [
        "read config", "summarize it"]


def test_split_clauses_numbered_list():
    from cognix.agent.planner import split_clauses
    parts = split_clauses("1. fetch data 2. clean it 3. plot it")
    assert parts == ["fetch data", "clean it", "plot it"]


def test_split_clauses_caps_at_five():
    from cognix.agent.planner import split_clauses
    text = " then ".join("step %d" % i for i in range(8))
    assert len(split_clauses(text)) == 5


def test_decompose_strategy_maps_tools(planner, beliefs):
    goal = Goal("read config then summarize it")
    plan = planner.plan(goal, beliefs, strategy="decompose")
    assert plan.strategy == "decompose"
    assert plan.status == "ready"
    assert len(plan.steps) == 2
    assert plan.steps[0].tool == "file_read"
    assert plan.steps[1].tool == "summarize"


def test_reactive_picks_best_tool_for_math(planner, beliefs):
    goal = Goal("calculate 2+2")
    plan = planner.plan(goal, beliefs, strategy="reactive")
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.tool == "calc"
    assert step.args["expression"] == "2+2"


def test_tool_mapping_falls_back_to_think(planner, beliefs):
    goal = Goal("ponder neon jellyfish")
    plan = planner.plan(goal, beliefs, strategy="reactive")
    assert plan.steps[0].tool == "think"


def test_auto_picks_reactive_for_simple(planner, beliefs):
    plan = planner.plan(Goal("calculate 2+2"), beliefs, strategy="auto")
    assert plan.strategy == "reactive"
    assert len(plan.steps) == 1


def test_auto_picks_decompose_for_multi_clause(planner, beliefs):
    plan = planner.plan(Goal("read config then summarize it"), beliefs,
                        strategy="auto")
    assert plan.strategy == "decompose"
    assert len(plan.steps) == 2


def test_means_ends_prepends_gather_step(planner, beliefs):
    goal = Goal("summarize the report at /tmp/report.txt")
    plan = planner.plan(goal, beliefs, strategy="means_ends")
    assert plan.strategy == "means_ends"
    assert len(plan.steps) == 2
    gather, final = plan.steps
    assert gather.tool == "file_read"
    assert gather.args["path"] == "/tmp/report.txt"
    assert final.tool == "summarize"


def test_plan_unknown_strategy_raises(planner, beliefs):
    with pytest.raises(ValueError):
        planner.plan(Goal("x"), beliefs, strategy="teleport")


def test_replan_bad_args_fills_missing(planner, beliefs):
    step = PlanStep("add numbers", tool="calc", args={})
    plan = Plan("goal-1", [step], status="ready", attempts=1)
    assert planner.validate(plan)
    fixed = planner.replan(plan, {"step_id": step.id,
                                  "error": "missing required arg",
                                  "kind": "bad_args"}, beliefs)
    assert fixed.attempts == 2
    assert fixed.status == "ready"
    assert "expression" in fixed.steps[0].args
    assert planner.validate(fixed) == []


def test_replan_tool_missing_swaps_tool(planner, beliefs):
    step = PlanStep("calculate 2+2", tool="ghost_tool", args={})
    plan = Plan("goal-1", [step], attempts=1)
    fixed = planner.replan(plan, {"step_id": step.id,
                                  "error": "not registered",
                                  "kind": "tool_missing"}, beliefs)
    assert fixed.steps[0].tool == "calc"


def test_replan_gives_up_after_three_attempts(planner, beliefs):
    step = PlanStep("calculate 2+2", tool="calc",
                    args={"expression": "2+2"})
    plan = Plan("goal-1", [step], attempts=1)
    failure = {"step_id": step.id, "error": "boom", "kind": "tool_error"}
    second = planner.replan(plan, failure, beliefs)
    third = planner.replan(second, failure, beliefs)
    fourth = planner.replan(third, failure, beliefs)
    assert second.status == "ready"
    assert third.status == "ready"
    assert fourth.status == "failed"
    assert fourth.attempts == 4


def test_replan_precondition_prepends_gather(planner, beliefs):
    step = PlanStep("summarize it", tool="summarize", args={"text": "hi"})
    plan = Plan("goal-1", [step], attempts=1)
    fixed = planner.replan(
        plan,
        {"step_id": step.id, "error": "file not found: /tmp/missing.txt",
         "kind": "precondition"},
        beliefs)
    assert len(fixed.steps) == 2
    assert fixed.steps[0].tool == "file_read"
    assert fixed.steps[0].args["path"] == "/tmp/missing.txt"
    assert fixed.steps[1].id == step.id


def test_replan_tool_error_adds_think_fallback(planner, beliefs):
    step = PlanStep("calculate 2+2", tool="calc",
                    args={"expression": "2+2"})
    plan = Plan("goal-1", [step], attempts=1)
    fixed = planner.replan(plan, {"step_id": step.id, "error": "boom",
                                  "kind": "tool_error"}, beliefs)
    assert len(fixed.steps) == 2
    assert fixed.steps[1].tool == "think"
    assert fixed.steps[0].max_retries == 3


def test_validate_catches_unknown_tool(planner, beliefs):
    plan = Plan("goal-1", [PlanStep("x", tool="ghost")])
    issues = planner.validate(plan)
    assert len(issues) == 1
    assert "unknown tool" in issues[0]


def test_validate_catches_missing_required_arg(planner, beliefs):
    plan = Plan("goal-1", [PlanStep("x", tool="calc", args={})])
    issues = planner.validate(plan)
    assert any("expression" in i for i in issues)


def test_validate_empty_plan(planner):
    assert planner.validate(Plan("goal-1", [])) == ["plan has no steps"]


def test_validate_clean_plan(planner, beliefs):
    plan = planner.plan(Goal("calculate 2+2"), beliefs, strategy="reactive")
    assert planner.validate(plan) == []


def test_plan_dict_roundtrip(planner, beliefs):
    plan = planner.plan(Goal("read config then summarize it"), beliefs,
                        strategy="decompose")
    clone = Plan.from_dict(plan.to_dict())
    assert clone.to_dict() == plan.to_dict()
    assert clone.step(plan.steps[0].id).tool == "file_read"


# executor


def test_run_step_think(executor):
    step = PlanStep("consider options", tool="think")
    result = executor.run_step(step)
    assert result.ok
    assert "thought" in result.output
    assert result.attempts == 1
    assert result.error_kind is None


def test_run_step_tool_success(executor):
    step = PlanStep("add", tool="calc", args={"expression": "2+2"})
    result = executor.run_step(step)
    assert result.ok
    assert result.output == 4
    assert result.duration >= 0


def test_run_step_retries_then_succeeds(registry):
    registry.failures["calc"] = 1
    fast = Executor(registry, backoff_base=0.0)
    step = PlanStep("add", tool="calc", args={"expression": "2+2"},
                    max_retries=2)
    result = fast.run_step(step)
    assert result.ok
    assert result.output == 4
    assert result.attempts == 2


def test_run_step_gives_up_after_retries(registry):
    registry.failures["calc"] = 10
    fast = Executor(registry, backoff_base=0.0)
    step = PlanStep("add", tool="calc", args={"expression": "2+2"},
                    max_retries=2)
    result = fast.run_step(step)
    assert not result.ok
    assert result.attempts == 3
    assert result.error_kind == "tool_error"


def test_run_step_tool_missing(executor):
    step = PlanStep("x", tool="ghost", args={})
    result = executor.run_step(step)
    assert not result.ok
    assert result.error_kind == "tool_missing"
    assert "not registered" in result.error


def test_run_step_classifies_bad_args(executor):
    step = PlanStep("write", tool="file_write", args={"path": "/tmp/x"})
    result = executor.run_step(step)
    assert not result.ok
    assert result.error_kind == "bad_args"


def test_run_step_classifies_precondition(executor):
    step = PlanStep("read", tool="file_read", args={"path": "/tmp/nope.txt"})
    result = executor.run_step(step)
    assert not result.ok
    assert result.error_kind == "precondition"


def test_classify_error_kinds():
    assert classify_error("missing required arg 'x'") == "bad_args"
    assert classify_error("file not found: /tmp/x") == "precondition"
    assert classify_error("weird explosion") == "tool_error"


def test_run_plan_success(executor):
    plan = Plan("goal-1", [
        PlanStep("add", tool="calc", args={"expression": "2+2"}),
        PlanStep("think it over", tool="think"),
    ])
    trace = executor.run_plan(plan)
    assert trace.success
    assert plan.status == "done"
    assert len(trace.results) == 2
    assert all(r.ok for r in trace.results)
    assert trace.ended is not None


def test_run_plan_stops_on_tool_missing(registry):
    executor = Executor(registry)
    plan = Plan("goal-1", [
        PlanStep("add", tool="calc", args={"expression": "1+1"}),
        PlanStep("ghost step", tool="ghost"),
        PlanStep("never runs", tool="calc", args={"expression": "3+3"}),
    ])
    trace = executor.run_plan(plan)
    assert not trace.success
    assert plan.status == "failed"
    assert len(trace.results) == 2
    assert len(registry.calls) == 1


def test_run_plan_continues_past_tool_error(executor):
    plan = Plan("goal-1", [
        PlanStep("bad math", tool="calc", args={"expression": "1/0"}),
        PlanStep("good math", tool="calc", args={"expression": "1+1"}),
    ])
    trace = executor.run_plan(plan)
    assert len(trace.results) == 2
    assert not trace.results[0].ok
    assert trace.results[1].ok
    assert trace.results[1].output == 2
    assert not trace.success


def test_run_plan_on_step_callback(executor):
    seen = []
    plan = Plan("goal-1", [PlanStep("add", tool="calc",
                                   args={"expression": "2+2"})])
    executor.run_plan(plan, on_step=seen.append)
    assert len(seen) == 1
    assert seen[0].step_id == plan.steps[0].id


def test_trace_summary(executor):
    plan = Plan("goal-1", [
        PlanStep("good", tool="calc", args={"expression": "1+1"}),
        PlanStep("bad", tool="calc", args={"expression": "1/0"}),
    ])
    trace = executor.run_plan(plan)
    summary = trace.summary()
    assert "1/2 steps ok" in summary
    assert "tool_error" in summary


def test_stepresult_dict_roundtrip():
    result = StepResult("step-1", True, output=4, duration=0.5, attempts=2,
                        tool="calc")
    clone = StepResult.from_dict(result.to_dict())
    assert clone.to_dict() == result.to_dict()


def test_trace_dict_roundtrip(executor):
    plan = Plan("goal-1", [PlanStep("add", tool="calc",
                                   args={"expression": "2+2"})])
    trace = executor.run_plan(plan)
    clone = ExecutionTrace.from_dict(trace.to_dict())
    assert clone.to_dict() == trace.to_dict()
    assert clone.success


# reflection


def _good_trace(executor):
    plan = Plan("goal-9", [PlanStep("add", tool="calc",
                                   args={"expression": "2+2"})])
    return executor.run_plan(plan), plan


def test_reflect_success(executor):
    trace, _ = _good_trace(executor)
    outcome = reflect(trace, Goal("calculate 2+2", id="goal-9"))
    assert outcome["success"] is True
    assert outcome["replan_needed"] is False
    assert outcome["failure_kind"] is None
    assert "succeeded" in outcome["summary"]
    assert any(l.scope == "planning" and l.confidence == 0.5
               for l in outcome["lessons"])


def test_reflect_bad_args_lesson_and_replan(executor):
    plan = Plan("goal-1", [PlanStep("write", tool="file_write",
                                   args={"path": "/tmp/x"})])
    trace = executor.run_plan(plan)
    outcome = reflect(trace, Goal("write file"))
    assert outcome["success"] is False
    assert outcome["failure_kind"] == "bad_args"
    assert outcome["replan_needed"] is True
    lesson = outcome["lessons"][0]
    assert lesson.confidence == 0.7
    assert lesson.scope == "tool"
    assert "validate args" in lesson.text


def test_reflect_tool_missing_needs_no_replan(executor):
    plan = Plan("goal-1", [PlanStep("x", tool="ghost")])
    trace = executor.run_plan(plan)
    outcome = reflect(trace, Goal("do x"))
    assert outcome["success"] is False
    assert outcome["failure_kind"] == "tool_missing"
    assert outcome["replan_needed"] is False
    assert outcome["lessons"][0].confidence == 0.8


def test_reflect_empty_output_is_not_success():
    trace = ExecutionTrace("plan-1",
                           results=[StepResult("s1", True, output="   ")],
                           success=True)
    outcome = reflect(trace, Goal("x"))
    assert outcome["success"] is False
    assert outcome["replan_needed"] is False


def test_reflect_retry_lesson():
    trace = ExecutionTrace("plan-1",
                           results=[StepResult("s1", True, output="done",
                                               attempts=3)],
                           success=True)
    outcome = reflect(trace, Goal("x"))
    assert any(l.scope == "general" and "retries" in l.text
               for l in outcome["lessons"])


def test_lesson_dict_roundtrip():
    lesson = Lesson("keep going", confidence=0.7, scope="tool")
    clone = Lesson.from_dict(lesson.to_dict())
    assert clone.to_dict() == lesson.to_dict()


def test_lesson_bad_scope_raises():
    with pytest.raises(ValueError):
        Lesson("x", scope="bogus")


# metacognition


def test_choose_reactive_after_repeated_failures(meta, beliefs):
    goal = Goal("read config then summarize it")
    assert meta.choose_strategy(goal, beliefs, past_failures=2) == "reactive"
    assert meta.choose_strategy(goal, beliefs, past_failures=5) == "reactive"


def test_choose_reactive_for_simple_goal(meta, beliefs):
    assert meta.choose_strategy(Goal("calculate 2+2"), beliefs) == "reactive"


def test_choose_decompose_for_multi_clause(meta, beliefs):
    goal = Goal("read config then summarize it")
    assert meta.choose_strategy(goal, beliefs, past_failures=0) == "decompose"


def test_choose_means_ends_for_target_state(meta, beliefs):
    goal = Goal("ensure the nightly backup is done")
    assert meta.choose_strategy(goal, beliefs) == "means_ends"
    assert meta.choose_strategy(Goal("make sure the server is up"),
                                beliefs) == "means_ends"


def test_confidence_in_plan_full(meta, beliefs, planner):
    plan = planner.plan(Goal("calculate 2+2"), beliefs, strategy="reactive")
    assert meta.confidence_in_plan(plan, beliefs) == 1.0


def test_confidence_in_plan_drops_with_unknown_tool(meta, beliefs):
    plan = Plan("goal-1", [
        PlanStep("add", tool="calc", args={"expression": "2+2"}),
        PlanStep("mystery", tool="ghost"),
    ])
    assert meta.confidence_in_plan(plan, beliefs) == 0.5


def test_confidence_in_plan_partial_args(meta, beliefs):
    plan = Plan("goal-1", [PlanStep("x", tool="calc", args={})])
    assert meta.confidence_in_plan(plan, beliefs) == 0.0


def test_confidence_in_plan_empty(meta, beliefs):
    assert meta.confidence_in_plan(Plan("goal-1", []), beliefs) == 0.0


def test_should_ask_for_help(meta, beliefs):
    goal = Goal("hard thing")
    assert meta.should_ask_for_help(goal, failures=2) is False
    assert meta.should_ask_for_help(goal, failures=3) is True


def test_estimate_difficulty_ordering(meta):
    simple = meta.estimate_difficulty(Goal("calculate 2+2"))
    complex_goal = meta.estimate_difficulty(Goal(
        "read the config then transform the zxqk data and finally "
        "blorpt the wobble output into seven separate artifacts"))
    assert 0.0 <= simple <= 1.0
    assert 0.0 <= complex_goal <= 1.0
    assert complex_goal > simple


# added coverage: stats, pruning, explain, dry_run, lesson book, advise


def test_goal_time_left():
    goal = Goal("x", deadline=1100.0)
    assert goal.time_left(now=1000.0) == 100.0
    assert goal.time_left(now=lambda: 1050.0) == 50.0
    assert Goal("y").time_left() is None


def test_stack_stats():
    stack = GoalStack(now=lambda: 1000.0)
    stack.push("a", deadline=900.0)
    b = stack.push("b")
    stack.suspend(b.id)
    c = stack.push("c")
    stack.complete(c.id)
    stats = stack.stats()
    assert stats["active"] == 1
    assert stats["suspended"] == 1
    assert stats["done"] == 1
    assert stats["overdue"] == 1


def test_prune_terminal():
    stack = GoalStack()
    a = stack.push("a")
    b = stack.push("b")
    stack.complete(a.id)
    stack.fail(b.id)
    assert stack.prune_terminal() == 2
    assert len(stack) == 0
    assert stack.prune_terminal() == 0


def test_explain_plan(planner, beliefs):
    plan = planner.plan(Goal("calculate 2+2"), beliefs, strategy="reactive")
    text = planner.explain(plan)
    assert plan.id in text
    assert "[calc]" in text
    assert "expression='2+2'" in text
    assert "1." in text


def test_dry_run(executor):
    plan = Plan("goal-1", [
        PlanStep("add", tool="calc", args={"expression": "2+2"}),
        PlanStep("think", tool="think"),
        PlanStep("ghost", tool="ghost"),
    ])
    report = executor.dry_run(plan)
    assert len(report) == 3
    assert report[0][1] is True
    assert report[1][1] is True
    assert report[2][1] is False
    assert "not registered" in report[2][2]
    assert len(executor.tools.calls) == 0


def test_lesson_book_query_and_absorb(executor):
    book = LessonBook()
    book.add(Lesson("a", confidence=0.9, scope="tool"))
    book.add(Lesson("b", confidence=0.4, scope="planning"))
    trace, _ = _good_trace(executor)
    book.absorb(reflect(trace, Goal("calculate 2+2", id="goal-9")))
    assert len(book) == 3
    tools = book.query(scope="tool")
    assert [l.text for l in tools] == ["a"]
    best = book.query(min_confidence=0.5)
    assert best[0].confidence == 0.9
    assert best[0].text == "a"


def test_lesson_book_dict_roundtrip():
    book = LessonBook([Lesson("a", 0.7, "tool"), Lesson("b", 0.5, "general")])
    clone = LessonBook.from_dict(book.to_dict())
    assert len(clone) == 2
    assert [l.text for l in clone.query()] == ["a", "b"]


def test_next_strategy(meta, beliefs):
    assert meta.next_strategy("decompose", "tool_missing") == "reactive"
    assert meta.next_strategy("reactive", "precondition") == "means_ends"
    assert meta.next_strategy("decompose", "bad_args") == "decompose"
    assert meta.next_strategy("reactive", "tool_error") == "decompose"
    assert meta.next_strategy("decompose", "tool_error") == "reactive"


def test_advise_bundle(meta, beliefs):
    advice = meta.advise(Goal("read config then summarize it"), beliefs)
    assert advice["strategy"] == "decompose"
    assert 0.0 <= advice["difficulty"] <= 1.0
    assert advice["ask_for_help"] is False
    assert advice["past_failures"] == 0
    urgent = meta.advise(Goal("x"), beliefs, past_failures=4)
    assert urgent["strategy"] == "reactive"
    assert urgent["ask_for_help"] is True
