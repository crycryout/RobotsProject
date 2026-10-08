import h5py

from conftest import make_test_env_policy
from robots_project.policies.controls import MockWAM
from robots_project.rollout.collector import collect


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
