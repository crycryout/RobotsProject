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


def test_default_retry_limit_prevents_repeated_failed_attempts(tmp_path, config):
    env, _ = make_test_env_policy(config)
    bad = MockWAM(env.action_spec, config.chunk_length, failure="nan")
    first = collect(env, bad, config, tmp_path, 2)
    env, _ = make_test_env_policy(config)
    bad = MockWAM(env.action_spec, config.chunk_length, failure="nan")
    second = collect(env, bad, config, tmp_path, 2)
    assert len(first["attempts"]) == len(second["attempts"]) == 2
    assert not env.executed


def test_reset_errors_are_recorded_as_attempts(tmp_path, config):
    env, policy = make_test_env_policy(config)

    def reset(*args):
        raise ValueError("reset failure fixture")

    env.reset = reset
    manifest = collect(env, policy, config, tmp_path, 2)
    assert len(manifest["attempts"]) == 2
    assert all(r["status"] == "infrastructure_error" for r in manifest["attempts"])


def test_model_raw_dimensions_preserved_separately_from_native_actions(tmp_path, config):
    config.obs_mode = "rgb"
    config.camera = {"width": 8, "height": 8, "shader_pack": "minimal"}
    env, policy = make_test_env_policy(config)
    original_act = policy.act
    original_specs = policy.specs
    policy.specs = lambda: {**original_specs(), "real_wam": True, "action_conversion_confirmed": True}

    def act(obs, context):
        output = original_act(obs, context)
        output.raw_model_output = np.ones((len(context["env_ids"]), config.chunk_length, 8), np.float32)
        output.conversion_metadata = {"target_action_spec_id": env.action_spec.id,
                                      "confirmation_ref": "SYNTHETIC_TRANSPORT_TEST_ONLY"}
        output.synthetic = True
        return output

    policy.act = act
    manifest = collect(env, policy, config, tmp_path, 2)
    assert manifest["complete"]
    for row in manifest["episodes"]:
        with h5py.File(tmp_path / row["trajectory"], "r") as f:
            assert f["actions/executed"].shape[-1] == 2
            assert next(iter(f["requests"].values()))["raw_model_output"].shape[-1] == 8
        assert row["synthetic"]


def test_visual_model_cannot_consume_privileged_state(tmp_path, config):
    env, policy = make_test_env_policy(config)
    specs = policy.specs
    policy.specs = lambda: {**specs(), "real_wam": True, "action_conversion_confirmed": True}
    manifest = collect(env, policy, config, tmp_path, 2)
    assert not manifest["complete"]
    assert not env.executed
    assert "sensor RGB" in manifest["attempts"][0]["error"]
