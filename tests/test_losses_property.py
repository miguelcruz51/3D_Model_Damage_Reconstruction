"""Property-based tests for loss behavior preservation (Property 2).

These tests compare the reorganized clean-repo loss functions against the
original Source_Repository implementations across random point-set pairs, for
each of the three seed-displacement branches: per-point L2 (``use_cd=False``),
L1 Chamfer distance (``use_cd=True, use_squared_cd=False``) and squared Chamfer
distance (``use_cd=True, use_squared_cd=True``).

The original ``SeedDisplacementLoss`` is loaded directly from its file path to
avoid triggering side-effect imports in the original ``models`` package.
"""

import importlib.util
import os

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Clean-repo (reorganized) loss under test.
from src.losses import SeedDisplacementLoss

_SOURCE_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _load_source_module(module_name, filename):
    """Load an original source module directly from its file path."""
    path = os.path.join(_SOURCE_REPO_ROOT, "models", filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_src_seed = _load_source_module("orig_seed_displacement_net", "seed_displacement_net.py")

TOL = 1e-5


def _make_seed_loss_inputs(batch, num_points, damaged_frac, seed):
    """Build a full corrected cloud, per-batch reconstruction lists, masks, GT.

    The reconstructed points are index-aligned with the ground-truth targets so
    that the per-point L2 branch is well defined, matching the correspondence
    used throughout the pipeline.
    """
    g = torch.Generator().manual_seed(seed)
    labels = (torch.rand(batch, num_points, generator=g) < damaged_frac).float()

    output_points = torch.randn(batch, num_points, 3, generator=g)
    gt_points = torch.randn(batch, num_points, 3, generator=g)
    gt_labels = labels.clone()

    reconstructed_list = []
    damaged_masks = []
    for b in range(batch):
        mask = labels[b] == 1
        damaged_masks.append(mask)
        k = int(mask.sum().item())
        reconstructed_list.append(torch.randn(k, 3, generator=g))
    return output_points, reconstructed_list, damaged_masks, gt_points, gt_labels


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    branch=st.sampled_from([(False, False), (True, False), (True, True)]),
    batch=st.integers(min_value=1, max_value=3),
    num_points=st.integers(min_value=8, max_value=64),
    damaged_frac=st.floats(min_value=0.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_loss_behavior_preservation(branch, batch, num_points, damaged_frac, seed):
    """**Feature: repo-reorganization, Property 2: For any pair of point sets, the
    reorganized loss functions (L1 Chamfer distance, squared Chamfer distance,
    and per-point L2) produce the same scalar value (within floating-point
    tolerance) as the corresponding Source_Repository implementations.**

    Validates: Requirements 4.2, 12.4
    """
    use_cd, use_squared_cd = branch

    inputs = _make_seed_loss_inputs(batch, num_points, damaged_frac, seed)
    output_points, reconstructed_list, damaged_masks, gt_points, gt_labels = inputs

    clean_loss = SeedDisplacementLoss(use_cd=use_cd, use_squared_cd=use_squared_cd)
    ref_loss = _src_seed.SeedDisplacementLoss(use_cd=use_cd, use_squared_cd=use_squared_cd)

    clean_value, _ = clean_loss(
        output_points, reconstructed_list, damaged_masks, gt_points, gt_labels
    )
    ref_value, _ = ref_loss(
        output_points, reconstructed_list, damaged_masks, gt_points, gt_labels
    )

    max_diff = (clean_value - ref_value).abs().item()
    assert max_diff <= TOL, f"branch={branch}: diff {max_diff} > {TOL}"
