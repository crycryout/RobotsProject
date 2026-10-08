"""CPU-only test fixtures are explicitly synthetic."""

import pytest

from robots_project.config import EnvConfig
from robots_project.envs.fake import FakeEnv
from robots_project.policies.controls import RandomPolicy


@pytest.fixture
def config():
    return EnvConfig(backend="fake", num_envs=2, episode_horizon=5,
                     chunk_length=4, execution_horizon=3).validate()


def make_test_env_policy(config, lengths=None, success_steps=None):
    env = FakeEnv(config, lengths=lengths, success_steps=success_steps)
    policy = RandomPolicy(env.action_spec, config.chunk_length)
    return env, policy
