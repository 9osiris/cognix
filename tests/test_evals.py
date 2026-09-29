"""Tests for the eval harness, scenarios, and metrics."""

import tempfile

from cognix.evals.harness import EvalReport, Scenario, format_report, run_all, run_scenario
from cognix.evals.metrics import (
    check_pass_rate,
    mean_score,
    recall_at_k,
    success_rate,
    summarize,
    weakest_scenarios,
)
from cognix.evals.scenarios import build_scenarios
from cognix.runtime import CognitiveRuntime


def make_runtime():
    return CognitiveRuntime(workspace_dir=tempfile.mkdtemp(prefix="cognix_eval_"))


def test_scenarios_build():
    scenarios = build_scenarios()
    assert len(scenarios) >= 8
    names = [s.name for s in scenarios]
    assert len(set(names)) == len(names)


def test_harness_runs_trivial_scenario():
    scenario = Scenario(
        "trivial",
        "does the harness work",
        actions=[("observe", "hello world")],
        checks=[("always true", lambda rt: (True, "fine"))],
    )
    result = run_scenario(scenario, make_runtime)
    assert result.passed
    assert result.score == 1.0


def test_harness_catches_failing_check():
    scenario = Scenario(
        "failing",
        "a check that fails",
        actions=[],
        checks=[("nope", lambda rt: (False, "broken"))],
    )
    result = run_scenario(scenario, make_runtime)
    assert not result.passed
    assert result.score == 0.0


def test_harness_catches_bad_action():
    scenario = Scenario(
        "bad",
        "unknown action",
        actions=[("explode",)],
        checks=[],
    )
    result = run_scenario(scenario, make_runtime)
    assert result.error is not None
    assert not result.passed


def test_full_suite_runs():
    scenarios = build_scenarios()
    report = run_all(scenarios, make_runtime)
    assert report.total == len(scenarios)
    # every scenario must at least execute without harness errors
    for result in report.results:
        assert result.error is None, (result.scenario.name, result.error)


def test_full_suite_scores_well():
    report = run_all(build_scenarios(), make_runtime)
    assert report.mean_score >= 0.7, format_report(report)


def test_format_report_contents():
    report = run_all(build_scenarios()[:2], make_runtime)
    text = format_report(report)
    assert "cognix eval report" in text
    assert "PASS" in text or "FAIL" in text


def test_metrics_math():
    report = run_all(build_scenarios()[:3], make_runtime)
    assert 0.0 <= success_rate(report) <= 1.0
    assert 0.0 <= mean_score(report) <= 1.0
    assert 0.0 <= check_pass_rate(report) <= 1.0
    summary = summarize(report)
    assert summary["scenarios_total"] == 3
    assert "weakest" in summary


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0
    assert recall_at_k(["a", "b", "c"], ["z"], 2) == 0.0
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert recall_at_k([], ["a"], 5) == 0.0


def test_weakest_scenarios_sorted():
    report = run_all(build_scenarios()[:4], make_runtime)
    weakest = weakest_scenarios(report, n=2)
    assert len(weakest) == 2
    assert weakest[0][1] <= weakest[1][1]


def test_report_to_dict():
    report = run_all(build_scenarios()[:1], make_runtime)
    data = report.to_dict()
    assert data["total"] == 1
    assert len(data["scenarios"]) == 1
    assert "checks" in data["scenarios"][0]
