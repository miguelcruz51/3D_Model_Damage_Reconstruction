"""Property-based tests for metric behavior preservation (Property 5).

Compares the reorganized ``src.evaluation.metrics.compute_metrics`` against the
original Source_Repository ``evaluate_all.compute_metrics`` (loaded directly from
its file path) across random ``(pred, gt, input, labels)`` tuples, asserting that
all five metrics match within floating-point tolerance.
"""

import importlib.util
import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.metrics import compute_metrics

_SOURCE_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _load_source_compute_metrics():
    """Load the original ``compute_metrics`` from ``evaluate_all.py`` by path."""
    path = os.path.join(_SOURCE_REPO_ROOT, "evaluate_all.py")
    spec = importlib.util.spec_from_file_location("orig_evaluate_all", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_metrics


_ref_compute_metrics = _load_source_compute_metrics()

TOL = 1e-6


@settings(max_examples=100, deadline=None)
@given(
    num_points=st.integers(min_value=4, max_value=64),
    damaged_frac=st.floats(min_value=0.0, max_value=1.0),
    scale_mm=st.floats(min_value=0.1, max_value=100.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_metric_behavior_preservation(num_points, damaged_frac, scale_mm, seed):
    """**Feature: repo-reorganization, Property 5: For any tuple of predicted
    points, ground-truth points, input points, and labels, the reorganized
    compute_metrics produces the same values (within floating-point tolerance)
    as the Source_Repository compute_metrics for full-cloud Chamfer distance,
    damaged-region Chamfer distance, per-point L2 error, Hausdorff distance, and
    intact-preservation error.**

    Validates: Requirements 7.4, 11.2
    """
    rng = np.random.default_rng(seed)
    n = num_points

    # Predicted and ground-truth clouds share the same length as the input, which
    # matches the displacement-method setting where intact/damaged masks apply.
    pred = rng.standard_normal((n, 3)).astype(np.float32)
    gt = rng.standard_normal((n, 3)).astype(np.float32)
    inp = rng.standard_normal((n, 3)).astype(np.float32)

    input_labels = (rng.random(n) < damaged_frac).astype(np.float32)
    # Ground-truth reconstruction targets are drawn independently.
    gt_labels = (rng.random(n) < damaged_frac).astype(np.float32)
    # pred_labels are unused by compute_metrics but kept for signature parity.
    pred_labels = input_labels

    clean = compute_metrics(pred, pred_labels, gt, gt_labels, inp, input_labels, scale_mm)
    ref = _ref_compute_metrics(pred, pred_labels, gt, gt_labels, inp, input_labels, scale_mm)

    assert set(clean.keys()) == set(ref.keys())
    for key in ref:
        assert clean[key] == pytest.approx(ref[key], abs=TOL), (
            f"metric '{key}' mismatch: clean={clean[key]} ref={ref[key]}"
        )
