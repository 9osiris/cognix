"""Aggregate metrics over eval reports."""

from .harness import EvalReport


def success_rate(report):
    if not report.total:
        return 0.0
    return report.passed / report.total


def mean_score(report):
    return report.mean_score


def check_pass_rate(report):
    total = sum(len(r.checks) for r in report.results)
    if not total:
        return 0.0
    passed = sum(1 for r in report.results for c in r.checks if c.passed)
    return passed / total


def per_scenario_scores(report):
    return {r.scenario.name: r.score for r in report.results}


def weakest_scenarios(report, n=3):
    ranked = sorted(report.results, key=lambda r: r.score)
    return [(r.scenario.name, r.score) for r in ranked[:n]]


def recall_at_k(ranked_ids, relevant_ids, k):
    # generic ranking metric used by memory checks
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & set(relevant_ids)) / len(relevant_ids)


def summarize(report):
    return {
        "scenarios_total": report.total,
        "scenarios_passed": report.passed,
        "success_rate": round(success_rate(report), 3),
        "mean_score": round(mean_score(report), 3),
        "check_pass_rate": round(check_pass_rate(report), 3),
        "weakest": weakest_scenarios(report),
    }
