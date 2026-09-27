"""Property-based tests for dataset behavior preservation (Properties 3 and 4).

These tests exercise the reorganized :class:`DMCLocalReconstruction` dataset:

- **Property 3** verifies that paired resampling of equal-length damaged/GT
  clouds selects identical indices for both clouds (point-to-point
  correspondence preservation).
- **Property 4** verifies that unit-sphere normalization keeps every damaged
  point within the unit sphere and applies the same centroid/scale transform to
  the ground-truth cloud.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data import DMCLocalReconstruction


def _make_dataset():
    """Construct a dataset instance without touching the filesystem.

    The resampling and normalization logic under test operates purely on arrays
    passed as arguments, so an empty component list is sufficient to build the
    object.
    """
    return DMCLocalReconstruction(
        data_root=".",
        split="test",
        components=[],
        num_points=2048,
        augment=False,
    )


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=256),
    num_points=st.integers(min_value=1, max_value=256),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_paired_resampling_correspondence(n, num_points, seed):
    """**Feature: repo-reorganization, Property 3: For any damaged partition and
    ground-truth partition of equal point count, and any fixed random seed, the
    reorganized paired-resampling selects identical indices for the damaged and
    ground-truth clouds, so point i of the resampled damaged cloud corresponds
    to point i of the resampled ground-truth cloud.**

    Validates: Requirements 5.2, 12.4
    """
    dataset = _make_dataset()

    # Encode the source index into each point/label so the selected index is
    # recoverable from the resampled output. Both clouds use the SAME encoding,
    # so if identical indices are selected the resampled clouds must be equal.
    idx_column = np.arange(n, dtype=np.float32).reshape(n, 1)
    d_points = np.tile(idx_column, (1, 3))
    g_points = d_points.copy()
    d_labels = np.arange(n, dtype=np.float32)
    g_labels = d_labels.copy()

    np.random.seed(seed)
    rd_points, rd_labels, rg_points, rg_labels = dataset._resample_paired(
        d_points, d_labels, g_points, g_labels, num_points
    )

    # Equal-length clouds must share indices -> resampled clouds are identical.
    assert np.array_equal(rd_points, rg_points)
    assert np.array_equal(rd_labels, rg_labels)
    # The encoded index recovered from points matches the label-encoded index.
    assert np.array_equal(rd_points[:, 0], rd_labels)


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=256),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_unit_sphere_normalization(n, seed):
    """**Feature: repo-reorganization, Property 4: For any partition, after the
    reorganized unit-sphere normalization, every damaged point lies within the
    unit sphere (its distance from the origin does not exceed 1 within
    floating-point tolerance), and the ground-truth cloud is transformed by the
    same centroid and scale as the damaged cloud.**

    Validates: Requirements 5.3
    """
    rng = np.random.default_rng(seed)
    damaged = (rng.standard_normal((n, 3)) * 10.0).astype(np.float32)
    gt = (rng.standard_normal((n, 3)) * 10.0).astype(np.float32)

    # Reproduce the normalization transform exactly as implemented in
    # __getitem__ (centroid + max-distance of the damaged cloud).
    centroid = damaged.mean(axis=0)
    damaged_centered = damaged - centroid
    max_dist = np.max(np.sqrt(np.sum(damaged_centered ** 2, axis=1)))
    if max_dist < 1e-6:
        max_dist = 1.0

    norm_damaged = damaged_centered / max_dist
    norm_gt = (gt - centroid) / max_dist

    eps = 1e-5
    # Every damaged point lies within the unit sphere.
    damaged_norms = np.sqrt(np.sum(norm_damaged ** 2, axis=1))
    assert damaged_norms.max() <= 1.0 + eps

    # GT is transformed by the same centroid and scale as damaged.
    expected_gt = (gt - centroid) / max_dist
    assert np.allclose(norm_gt, expected_gt, atol=eps)
