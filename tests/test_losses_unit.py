"""Unit tests for loss configuration defaults."""

from src.losses import SeedDisplacementLoss


def test_seed_displacement_loss_flag_defaults():
    """SeedDisplacementLoss defaults to L2 (use_cd=False, use_squared_cd=False).

    Validates: Requirements 4.4
    """
    loss = SeedDisplacementLoss()
    assert loss.use_cd is False
    assert loss.use_squared_cd is False
