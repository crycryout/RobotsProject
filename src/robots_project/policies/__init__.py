"""Infrastructure controls and explicitly synthetic model fixtures."""

from .controls import HoldPolicy, MockWAM, RandomPolicy, ReplayPolicy


def make_policy(name, action_spec, config, replay_path=None):
    common = dict(action_spec=action_spec, chunk_length=config.chunk_length)
    if name == "random":
        return RandomPolicy(**common, seed=config.seed)
    if name == "hold":
        return HoldPolicy(**common)
    if name == "mock":
        return MockWAM(**common, seed=config.seed)
    if name == "replay":
        if replay_path is None:
            raise ValueError("Replay requires --replay-path to a validated episode")
        return ReplayPolicy(**common, path=replay_path)
    raise ValueError(f"Unknown policy: {name}")
