import json

import h5py
import numpy as np
import pytest

from conftest import make_test_env_policy
from robots_project.policies.controls import MockWAM
from robots_project.rollout.collector import collect


def test_replan_h_smaller_than_k_and_discard_on_reset(tmp_path, config):
    env, policy = make_test_env_policy(config, lengths=[2, 5])
    manifest = collect(env, policy, config, tmp_path, 3)
    for row in manifest["episodes"]:
        with h5py.File(tmp_path / row["trajectory"], "r") as f:
            for group in f["requests"].values():
                request = json.loads(group.attrs["metadata_json"])
                assert request["executed_prefix"] <= 3
                assert group["raw_actions"].shape == (4, 2)
            assert (f["transitions/action_index"][:] < 3).all()


@pytest.mark.parametrize("failure", ["nan", "shape", "timeout"])
def test_bad_policy_is_an_infrastructure_error_without_fallback(tmp_path, config, failure):
    env, _ = make_test_env_policy(config)
    config.policy_timeout_s = 0.01 if failure == "timeout" else 1
    policy = MockWAM(env.action_spec, config.chunk_length,
                     latency_s=0.1 if failure == "timeout" else 0,
                     failure=failure)
    manifest = collect(env, policy, config, tmp_path, 2)
    assert not manifest["complete"]
    assert not env.executed
    assert len(manifest["attempts"]) == 2
    assert all(r["status"] == "infrastructure_error" for r in manifest["attempts"])
    assert not manifest["episodes"]


def test_unconfirmed_model_conversion_rejected(tmp_path, config):
    env, policy = make_test_env_policy(config)
    specs = policy.specs
    policy.specs = lambda: {**specs(), "real_wam": True, "action_conversion_confirmed": False}
    manifest = collect(env, policy, config, tmp_path, 2)
    assert not manifest["complete"]
    assert not env.executed


def test_clipping_retained_in_audit(tmp_path, config):
    env, policy = make_test_env_policy(config)
    original_act = policy.act

    def act(obs, context):
        output = original_act(obs, context)
        output.actions[:] = 2.0
        return output

    policy.act = act
    manifest = collect(env, policy, config, tmp_path, 2)
    for row in manifest["episodes"]:
        with h5py.File(tmp_path / row["trajectory"], "r") as f:
            assert np.all(f["actions/raw"][:] == 2)
            assert np.all(f["actions/executed"][:] == 1)
            assert np.all(f["actions/clip_mask"][:])
