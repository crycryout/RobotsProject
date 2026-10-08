"""Simulation adapters with explicit episode boundaries."""


def make_env(config):
    if config.backend == "fake":
        from .fake import FakeEnv
        return FakeEnv(config)
    from .maniskill import ManiSkillAdapter
    return ManiSkillAdapter(config)
