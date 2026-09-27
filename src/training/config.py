"""Configuration loading and seeding for training.

Loads a flat YAML config file into a simple attribute-style ``Config`` object
that the trainer and model/loss adapters consume, validates that the required
hyperparameter keys are present, and sets the numpy/torch random seeds for
reproducibility (Req 10.2).
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Keys every training config must provide. These mirror the hyperparameters
# that the original ``train_*`` scripts exposed as command-line arguments.
REQUIRED_KEYS = (
    "model",
    "data_root",
    "num_points",
    "batch_size",
    "lr",
    "epochs",
    "lr_decay_step",
    "lr_decay_rate",
    "weight_decay",
    "seed",
    "log_dir",
)

# Model identifiers understood by the trainer/adapters.
VALID_MODELS = (
    "seeddispnet",
    "pure_pcn",
    "lrpcn",
    "blind_displacement",
    "flow_displacement",
)


class Config:
    """Attribute-style view over a flat configuration dictionary.

    Every key/value pair in the source mapping is exposed as an attribute so
    that adapters can read parameters with ``config.lr`` style access, matching
    the ``args`` objects the original training scripts used.
    """

    def __init__(self, values: Dict[str, Any]) -> None:
        self._values = dict(values)
        for key, value in self._values.items():
            setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` or ``default`` when the key is absent."""
        return self._values.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the underlying configuration mapping."""
        return dict(self._values)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self._values!r})"


def validate_config(values: Dict[str, Any]) -> None:
    """Validate that ``values`` contains every required key and a known model.

    Args:
        values: The parsed configuration mapping.

    Raises:
        ValueError: If any required key is missing or ``model`` is unknown.
    """
    missing = [key for key in REQUIRED_KEYS if key not in values]
    if missing:
        raise ValueError(
            "Missing required config keys: " + ", ".join(sorted(missing))
        )
    if values["model"] not in VALID_MODELS:
        raise ValueError(
            f"Unknown model '{values['model']}'. "
            f"Expected one of: {', '.join(VALID_MODELS)}"
        )


def set_seed(seed: int) -> None:
    """Seed the Python, numpy, and torch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Config:
    """Load, validate, and return a :class:`Config` from a YAML file.

    The random seeds are set as a side effect so that dataset construction and
    model initialization downstream are reproducible (Req 10.2).

    Args:
        path: Filesystem path to a flat YAML configuration file.

    Returns:
        A validated :class:`Config` instance.

    Raises:
        ValueError: If the file does not contain a mapping or is missing keys.
    """
    with open(path, "r") as handle:
        values = yaml.safe_load(handle)

    if not isinstance(values, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")

    validate_config(values)
    set_seed(int(values["seed"]))
    return Config(values)
