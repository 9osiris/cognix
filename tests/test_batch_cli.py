

"""Batch mode and CLI tests."""

import tempfile

import pytest

from cognix.runtime import CognitiveRuntime


@pytest.fixture
def rt():
    workspace = tempfile.mkdtemp(prefix="cognix_test_")
    return CognitiveRuntime(workspace_dir=workspace)


def test_batch_replay_command(rt):
    from cognix.batch import run_batch
    rt.observe("the atlas server is running hot")
    for item in rt.working.top(20):
        rt.working.rehearse(item.id)
    results = run_batch(rt, ["replay"])
    assert results[0][1] is True
    assert "replayed=" in results[0][2]


def test_batch_curiosity_commands(rt):
    from cognix.batch import run_batch
    results = run_batch(rt, ["curiosity", "curiosity: the quantum heron"])
    assert results[0][2].startswith("boredom=")
    assert results[1][2].startswith("novelty=")


def test_batch_expand_and_runall(rt):
    from cognix.batch import run_batch
    results = run_batch(rt, [
        "goal: 0.9 calculate 2 * 3 and calculate 3 * 4",
        "expand",
        "runall",
        "assert: goal-done",
    ])
    assert all(ok for _, ok, _ in results), results
    assert results[1][2] == "2 subgoals"
