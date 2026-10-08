import pytest

from robots_project.config import EnvConfig, load_env_config
from robots_project.evaluation.splits import validate_splits


@pytest.mark.parametrize("value", [0, 5, 1.5, True])
def test_execution_horizon_requires_integer_prefix(value):
    with pytest.raises(ValueError, match="execution_horizon"):
        EnvConfig(execution_horizon=value).validate()


def test_unknown_configuration_keys_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("unknown_option: true\n")
    with pytest.raises(TypeError):
        load_env_config(path)


def test_disjoint_frozen_manifests():
    from pathlib import Path
    assert validate_splits(Path("manifests")) == {"train": 40, "validation": 40, "test": 100}
