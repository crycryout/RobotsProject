"""Reward composition and scientific alignment metrics are owned by the trainer/reward owner."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class RewardResult:
    auxiliary_reward: np.ndarray
    validity: np.ndarray
    diagnostics: dict
    reward_version: str


class RewardHook(Protocol):
    def compute(self, batch) -> RewardResult: ...


class ZeroRewardHook:
    """Pipeline control: zero auxiliary rewards, with actual prediction validity retained."""

    def compute(self, batch):
        validity = np.asarray([pair.valid for pair in batch], dtype=bool)
        return RewardResult(np.zeros(len(batch), np.float32), validity,
                            {"purpose": "pipeline_control", "scientific_reward": False,
                             "synthetic_pairs": sum(pair.synthetic for pair in batch)}, "zero-hook-v1")
