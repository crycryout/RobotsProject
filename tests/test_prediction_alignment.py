import h5py

from conftest import make_test_env_policy
from robots_project.policies.controls import MockWAM
from robots_project.rollout.collector import collect
from robots_project.integration.alignment import pairs_from_episode
from robots_project.integration.rewards import ZeroRewardHook


def test_predictions_only_valid_for_executed_prefix(tmp_path, config):
    env, _ = make_test_env_policy(config, lengths=[2, 5])
    policy = MockWAM(env.action_spec, config.chunk_length)
    manifest = collect(env, policy, config, tmp_path, 2)
    for row in manifest["episodes"]:
        with h5py.File(tmp_path / row["trajectory"], "r") as f:
            for group in f["requests"].values():
                mask = group["prediction/valid"][:]
                assert not mask[3]
                assert group["prediction/invalid_reason"].asstr()[3] == "unexecuted_or_replanned_prefix"
                if row["steps"] == 2:
                    assert mask.tolist() == [True, True, False, False]
            assert row["synthetic"]


def test_camera_time_binding_and_zero_reward_preserve_validity(tmp_path, config):
    config.obs_mode = "rgb"
    config.camera = {"width": 8, "height": 8, "shader_pack": "minimal"}
    env, _ = make_test_env_policy(config, lengths=[2, 5])
    policy = MockWAM(env.action_spec, config.chunk_length)
    manifest = collect(env, policy, config, tmp_path, 2)
    row = next(r for r in manifest["episodes"] if r["steps"] == 2)
    pairs = pairs_from_episode(tmp_path / row["trajectory"])
    assert [pair.valid for pair in pairs] == [True, True, False, False]
    assert [pair.actual_control_step for pair in pairs] == [1, 2, 3, 4]
    assert all(pair.camera_id == "base_camera" and pair.synthetic for pair in pairs)
    assert pairs[2].actual is None
    assert pairs[0].actual[0, 0, 0] == 1
    assert pairs[0].predicted[0, 0, 0] == 0
    reward = ZeroRewardHook().compute(pairs)
    assert not reward.auxiliary_reward.any()
    assert reward.validity.tolist() == [True, True, False, False]
    with h5py.File(tmp_path / row["trajectory"], "r") as f:
        assert "training_extras" not in next(iter(f["requests"].values()))


def test_clipped_prefix_invalidates_prediction(tmp_path, config):
    env, _ = make_test_env_policy(config)
    policy = MockWAM(env.action_spec, config.chunk_length)
    original = policy.act

    def act(obs, context):
        output = original(obs, context)
        output.actions[:, 0] = 2.0
        return output

    policy.act = act
    manifest = collect(env, policy, config, tmp_path, 2)
    for row in manifest["episodes"]:
        with h5py.File(tmp_path / row["trajectory"], "r") as f:
            for request in f["requests"].values():
                assert not request["prediction/valid"][:].any()
                assert request["prediction/invalid_reason"].asstr()[0] == "clipped_action_prefix"
