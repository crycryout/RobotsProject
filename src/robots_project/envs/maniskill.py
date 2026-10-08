"""ManiSkill 3.0.1 adapter. Evaluation retains raw termination signals."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json

import numpy as np

from robots_project.config import EnvConfig
from robots_project.envs.observations import normalize_autoreset, snapshot, standardize, take
from robots_project.types import ActionSpec, StepBatch


class ManiSkillAdapter:
    def __init__(self, config: EnvConfig):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils.registration import REGISTERED_ENVS

        installed = importlib.metadata.version("mani_skill")
        if config.backend_version != installed:
            raise ValueError(f"ManiSkill version mismatch: expected {config.backend_version}, got {installed}")
        if config.env_id not in REGISTERED_ENVS:
            raise ValueError(f"Unregistered task: {config.env_id}")
        self.config = config
        self.num_envs = config.num_envs
        self.env = gym.make(
            config.env_id, robot_uids=config.robot_id, obs_mode=config.obs_mode,
            control_mode=config.control_mode, reward_mode=config.reward_mode,
            num_envs=config.num_envs, sim_backend=config.sim_backend,
            render_backend=config.render_backend, sensor_configs=config.camera,
            sim_config={"control_freq": config.control_freq, "sim_freq": config.sim_freq},
            max_episode_steps=config.episode_horizon,
            reconfiguration_freq=config.reconfiguration_freq,
        )
        self.base = self.env.unwrapped
        if config.robot_id not in self.base.SUPPORTED_ROBOTS:
            self.env.close()
            raise ValueError(f"Unsupported robot {config.robot_id} on {config.env_id}")
        if config.auto_reset:
            from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
            self.env = ManiSkillVectorEnv(self.env, auto_reset=True, ignore_terminations=False)
        space = self.base.single_action_space
        low, high = np.asarray(space.low), np.asarray(space.high)
        signature = json.dumps({"robot": config.robot_id, "control": config.control_mode,
                                "dimension": space.shape[0], "low": low.tolist(),
                                "high": high.tolist(), "backend_version": installed}, sort_keys=True)
        self.action_spec = ActionSpec(
            id="maniskill:" + hashlib.sha256(signature.encode()).hexdigest()[:20],
            dimension=space.shape[0], low=low, high=high,
            semantics={"robot_id": config.robot_id, "control_mode": config.control_mode,
                       "control_freq_hz": config.control_freq, "native_controller": True,
                       "droid_compatible": False,
                       "controller_config": {k: repr(v.config) for k, v in
                                             self.base.agent.controller.controllers.items()}},
        )
        self.episode_ids = [""] * self.num_envs
        self.steps = np.zeros(self.num_envs, dtype=np.int64)
        self.seeds = np.zeros(self.num_envs, dtype=np.int64)
        self.observation = None

    def specs(self) -> dict:
        return {"backend": "maniskill", "backend_version": self.config.backend_version,
                "task_id": self.config.env_id, "robot_id": self.config.robot_id,
                "obs_mode": self.config.obs_mode, "camera_ids": self.config.camera_ids,
                "simulation_backend": self.config.sim_backend,
                "num_envs": self.num_envs, "action_spec": self.action_spec.as_dict(),
                "dt_sim": 1 / self.config.control_freq, "synthetic": False,
                "privileged_observation": self.config.obs_mode in {"state", "state_dict"},
                "preprocessing_version": "sensor-native-rgb-u8-v1"}

    def reset(self, env_ids: list[int], episode_ids: list[str], seeds: list[int]):
        if not len(env_ids) == len(episode_ids) == len(seeds):
            raise ValueError("Reset identifiers and seeds must have matching lengths")
        for slot, episode, seed in zip(env_ids, episode_ids, seeds, strict=True):
            self.episode_ids[slot] = episode
            self.steps[slot] = 0
            self.seeds[slot] = seed
        obs, _ = self.env.reset(seed=self.seeds.tolist(), options={"env_idx": env_ids})
        self.observation = standardize(obs, self.config.obs_mode)
        return self.observation

    def initial_state(self, slot: int) -> dict:
        return take(snapshot(self.base.get_state_dict()), [slot])

    def step(self, actions: np.ndarray) -> StepBatch:
        import torch
        if self.observation is None:
            raise RuntimeError("Reset before stepping")
        if actions.shape != (self.num_envs, self.action_spec.dimension):
            raise ValueError("Executed action batch has wrong shape")
        if not np.isfinite(actions).all():
            raise ValueError("Executed actions contain non-finite values")
        if (actions < self.action_spec.low - 1e-6).any() or (actions > self.action_spec.high + 1e-6).any():
            raise ValueError("Executed actions must already be clipped and audited")
        before = self.observation
        obs, reward, terminated, truncated, info = self.env.step(
            torch.as_tensor(actions, dtype=torch.float32, device=self.base.device))
        term = np.broadcast_to(snapshot(terminated), (self.num_envs,)).astype(bool).copy()
        trunc = np.broadcast_to(snapshot(truncated), (self.num_envs,)).astype(bool).copy()
        after, reset = normalize_autoreset(obs, info, term | trunc)
        info_before = info.get("final_info", info)
        success = np.broadcast_to(snapshot(info_before["success"]), (self.num_envs,)).astype(bool).copy()
        step = StepBatch(
            obs_before=before, action_executed=actions.copy(),
            obs_after_before_reset=standardize(after, self.config.obs_mode),
            reward_env=snapshot(reward).reshape(self.num_envs), terminated=term, truncated=trunc,
            success=success, episode_id=self.episode_ids.copy(), env_slot=np.arange(self.num_envs),
            step_index=self.steps.copy(), dt_sim=1 / self.config.control_freq,
            reset_observation=standardize(reset, self.config.obs_mode) if reset is not None else None,
            reset_mask=(term | trunc) if reset is not None else None,
        )
        self.steps += 1
        self.observation = standardize(obs, self.config.obs_mode)
        return step

    def close(self) -> None:
        self.env.close()
