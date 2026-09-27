"""Training loop and configuration utilities.

Config utilities are imported eagerly (lightweight); the trainer is imported
lazily via ``__getattr__`` so that config-only usage does not require the
heavier training dependencies (tqdm, matplotlib).
"""

from src.training.config import Config, load_config, set_seed, validate_config

__all__ = [
    "Config",
    "load_config",
    "set_seed",
    "validate_config",
    "build_adapter",
    "train",
]


def __getattr__(name):
    if name in ("build_adapter", "train"):
        from src.training import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
