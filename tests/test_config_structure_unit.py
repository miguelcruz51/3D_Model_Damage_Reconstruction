"""Unit tests for config files and clean-repo package structure.

Covers Req 2.5 (Python package initializers / directory structure) and Req 6.3
(each experiment config contains the required hyperparameter keys).
"""

import os

import pytest
import yaml

from src.training.config import REQUIRED_KEYS, VALID_MODELS, load_config, validate_config

CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_NAMES = [
    "seeddispnet",
    "pure_pcn",
    "lrpcn",
    "blind_displacement",
    "flow_displacement",
]

REQUIRED_PACKAGES = [
    "src",
    "src/data",
    "src/models",
    "src/losses",
    "src/training",
    "src/evaluation",
    "src/utils",
]

REQUIRED_DIRS = [
    "scripts",
    "configs",
    "experiments",
    "results/figures",
    "results/metrics",
    "results/qualitative",
    "docs",
    "examples",
]


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_has_required_keys(name):
    """Each YAML config contains every required hyperparameter key (Req 6.3)."""
    path = os.path.join(CLEAN_REPO_ROOT, "configs", f"{name}.yaml")
    assert os.path.isfile(path), f"missing config: {path}"
    with open(path) as handle:
        values = yaml.safe_load(handle)
    for key in REQUIRED_KEYS:
        assert key in values, f"{name}.yaml missing required key '{key}'"
    assert values["model"] in VALID_MODELS


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_loads_and_validates(name):
    """load_config parses and validates each shipped config (Req 6.3)."""
    path = os.path.join(CLEAN_REPO_ROOT, "configs", f"{name}.yaml")
    config = load_config(path)
    assert config.model in VALID_MODELS
    assert config.get("seed") is not None


def test_validate_config_rejects_missing_keys():
    """validate_config raises when a required key is absent (Req 6.3)."""
    with pytest.raises(ValueError):
        validate_config({"model": "seeddispnet"})


def test_validate_config_rejects_unknown_model():
    """validate_config raises for an unknown model identifier (Req 6.3)."""
    values = {key: 0 for key in REQUIRED_KEYS}
    values["model"] = "not_a_model"
    with pytest.raises(ValueError):
        validate_config(values)


@pytest.mark.parametrize("pkg", REQUIRED_PACKAGES)
def test_packages_have_init(pkg):
    """Each Python package directory has an __init__.py (Req 2.5)."""
    init_path = os.path.join(CLEAN_REPO_ROOT, pkg, "__init__.py")
    assert os.path.isfile(init_path), f"missing __init__.py: {init_path}"


@pytest.mark.parametrize("directory", REQUIRED_DIRS)
def test_required_directories_exist(directory):
    """Required top-level directories exist in the clean repo (Req 2.5)."""
    path = os.path.join(CLEAN_REPO_ROOT, directory)
    assert os.path.isdir(path), f"missing directory: {path}"
