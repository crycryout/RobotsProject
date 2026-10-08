"""Controllable synthetic environment for boundary tests, never robotics evidence."""

from __future__ import annotations

import numpy as np

from robots_project.envs.observations import standardize
from robots_project.types import ActionSpec, StepBatch


class FakeEnv:
    def __init__(self, config, lengths=None, success_steps=None):
        self.config = config
        self.num_envs = config.num_envs
        self.lengths = np.asarray(lengths or [config.episode_horizon] * self.num_envs)
        self.success_steps = success_steps or {}
        self.steps = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_ids = [""] * self.num_envs
        self.seeds = np.zeros(self.num_envs, dtype=np.int64)
        self.state = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.action_spec = ActionSpec("fake-native-2d-v1", 2, -np.ones(2, np.float32),
                                      np.ones(2, np.float32), {"units": "synthetic"})
        self.observation = standardize(self.state, "state")
        self.executed = []

    def specs(self):
        return {"backend": "fake", "task_id": self.config.env_id, "synthetic": True,
                "action_spec": self.action_spec.as_dict(), "dt_sim": 1 / self.config.control_freq,
                "privileged_observation": True, "num_envs": self.num_envs}

    def reset(self, env_ids, episode_ids, seeds):
        for slot, episode, seed in zip(env_ids, episode_ids, seeds, strict=True):
            self.episode_ids[slot] = episode
            self.steps[slot] = 0
            self.seeds[slot] = seed
            self.state[slot] = [seed % 1000, 0]
        self.observation = standardize(self.state, "state")
        return self.observation

    def initial_state(self, slot):
        return {"fake_state": self.state[slot:slot + 1].copy()}

    def step(self, actions):
        before = self.observation
        indices = self.steps.copy()
        self.state[:, 1] += actions[:, 0]
        self.steps += 1
        self.executed.append((self.episode_ids.copy(), indices, actions.copy()))
        success = np.asarray([int(self.steps[s]) in self.success_steps.get(s, set())
                              for s in range(self.num_envs)])
        truncated = self.steps >= self.lengths
        after = standardize(self.state, "state")
        reset = None
        if self.config.auto_reset and (success | truncated).any():
            for slot in np.flatnonzero(success | truncated):
                self.state[slot] = [-999, 0]
            reset = standardize(self.state, "state")
        self.observation = reset or after
        return StepBatch(before, actions.copy(), after, np.ones(self.num_envs, np.float32),
                         success, truncated, success, self.episode_ids.copy(),
                         np.arange(self.num_envs), indices, 1 / self.config.control_freq,
                         reset, (success | truncated) if reset else None)

    def close(self):
        pass
