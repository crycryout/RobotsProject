import math

from conftest import make_test_env_policy
from robots_project.evaluation.artifacts import summarize, write_episode_table
from robots_project.evaluation.metrics import aggregate, wilson
from robots_project.rollout.collector import collect


def test_intermediate_success_final_failure_and_reproducible_summary(tmp_path, config):
    config.termination = "evaluation"
    config.reconfiguration_freq = 1
    env, policy = make_test_env_policy(config, success_steps={0: {2}})
    manifest = collect(env, policy, config, tmp_path, 2)
    write_episode_table(tmp_path, manifest)
    summary = summarize(tmp_path, plots=False)
    task = summary["per_task"][config.env_id]
    assert task["success_once_count"] == 1
    assert task["success_at_end_count"] == 0
    assert task["completed"] == 2
    before = (tmp_path / "summary.json").read_bytes()
    summarize(tmp_path, plots=False)
    assert (tmp_path / "summary.json").read_bytes() == before
    failed = next(r for r in manifest["episodes"] if r["success_once"] == 0)
    assert failed["time_to_first_success"] is None


def test_macro_average_incomplete_denominators_and_errors():
    planned = [{"episode_id": f"a{i}", "task_id": "A"} for i in range(2)] + [
        {"episode_id": f"b{i}", "task_id": "B"} for i in range(4)]
    rows = [{"episode_id": "a0", "task_id": "A", "status": "complete", "success_once": 1,
             "success_at_end": 0, "return_env": 1, "steps": 2},
            {"episode_id": "a1", "task_id": "A", "status": "infrastructure_error"}]
    rows += [{"episode_id": f"b{i}", "task_id": "B", "status": "complete", "success_once": 0,
              "success_at_end": 0, "return_env": 0, "steps": 2} for i in range(4)]
    summary = aggregate(rows, planned)
    assert summary["macro_success_once"] == 0.5
    assert summary["macro_conservative_success_once"] == 0.25
    assert summary["per_task"]["A"]["error_rate"] == 0.5
    assert not summary["complete"]
    assert summary["completed"] == 5


def test_wilson_known_edges():
    assert wilson(0, 0) is None
    assert math.isclose(wilson(0, 20)[1], 0.161125158, abs_tol=1e-8)
    assert math.isclose(wilson(20, 20)[0], 0.838874842, abs_tol=1e-8)
