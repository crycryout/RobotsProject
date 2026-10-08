import h5py
import numpy as np

from conftest import make_test_env_policy
from robots_project.data.validation import validate_run
from robots_project.envs.observations import normalize_autoreset
from robots_project.rollout.collector import collect


def test_one_slot_ends_without_resetting_other_history(tmp_path, config):
    env, policy = make_test_env_policy(config, lengths=[2, 5])
    reset_calls = []
    original_reset = policy.reset

    def reset(slots, ids):
        reset_calls.append((slots.copy(), ids.copy()))
        original_reset(slots, ids)

    policy.reset = reset
    manifest = collect(env, policy, config, tmp_path, 4)
    assert manifest["complete"]
    assert reset_calls[1][0] == [0]
    slot_one = [(ids[1], indices[1]) for ids, indices, _ in env.executed[:5]]
    assert len({episode for episode, _ in slot_one}) == 1
    assert [index for _, index in slot_one] == list(range(5))
    assert validate_run(tmp_path)["valid"]


def test_time_limit_keeps_terminal_observation(tmp_path, config):
    config.auto_reset = True
    env, policy = make_test_env_policy(config, lengths=[2, 5])
    manifest = collect(env, policy, config, tmp_path, 2)
    for row in manifest["episodes"]:
        with h5py.File(tmp_path / row["trajectory"], "r") as f:
            assert f["transitions/truncated"][-1]
            assert not f["transitions/terminated"][:].any()
            assert f["obs/state"][-1, 0] == row["seed"] % 1000
            assert f[f"reset_events/{row['steps']}/state"][0] == -999
            assert f["obs/state"].shape[0] == row["steps"] + 1


def test_pinned_maniskill_full_batched_final_observation_layout():
    reset_obs = {"rgb": np.array([[1], [2]])}
    info = {"final_observation": {"rgb": np.array([[7], [8]])},
            "_final_observation": np.array([True, False])}
    terminal, reset = normalize_autoreset(reset_obs, info, np.array([True, False]))
    assert terminal["rgb"].tolist() == [[7], [2]]
    assert reset["rgb"].tolist() == [[1], [2]]
    assert reset_obs["rgb"].tolist() == [[1], [2]]
