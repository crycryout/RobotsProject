import h5py
import pytest

from conftest import make_test_env_policy
from robots_project.data.validation import validate_episode, validate_run
from robots_project.rollout.collector import collect
from robots_project.utils import atomic_json


def test_interruption_recovery_no_duplicate_completed_episodes(tmp_path, config):
    env, policy = make_test_env_policy(config, lengths=[2, 5])
    first = collect(env, policy, config, tmp_path, 4, pause_after_episodes=1)
    saved = first["episodes"][0]
    checksum = saved["sha256"]
    assert list((tmp_path / "episodes").glob("*.tmp"))
    env, policy = make_test_env_policy(config, lengths=[2, 5])
    resumed = collect(env, policy, config, tmp_path, 4)
    assert resumed["complete"]
    assert len(resumed["episodes"]) == 4
    assert len({r["episode_id"] for r in resumed["episodes"]}) == 4
    assert next(r["sha256"] for r in resumed["episodes"] if r["episode_id"] == saved["episode_id"]) == checksum
    assert resumed["recovery"]
    assert validate_run(tmp_path)["valid"]


def test_recover_rename_before_manifest_and_sidecar(tmp_path, config):
    env, policy = make_test_env_policy(config)
    finished = collect(env, policy, config, tmp_path, 2)
    row = finished["episodes"].pop()
    finished["attempts"] = [r for r in finished["attempts"] if r["episode_id"] != row["episode_id"]]
    (tmp_path / row["trajectory"]).with_suffix(".json").unlink()
    atomic_json(tmp_path / "manifest.json", finished)
    env, policy = make_test_env_policy(config)
    restored = collect(env, policy, config, tmp_path, 2)
    assert len(restored["episodes"]) == 2
    assert len(restored["attempts"]) == 2
    assert row["episode_id"] in {r["episode_id"] for r in restored["episodes"]}


def test_checksum_and_t_plus_one_validation(tmp_path, config):
    env, policy = make_test_env_policy(config)
    manifest = collect(env, policy, config, tmp_path, 2)
    row = manifest["episodes"][0]
    path = tmp_path / row["trajectory"]
    with h5py.File(path, "r+") as f:
        f["obs/state"].resize(f["obs/state"].shape[0] - 1, axis=0)
    with pytest.raises(ValueError, match="checksum"):
        validate_episode(path, row["sha256"])
    with pytest.raises(ValueError, match="T\\+1"):
        validate_episode(path)
