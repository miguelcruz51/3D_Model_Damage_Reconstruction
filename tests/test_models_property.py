"""Property-based tests for model behavior preservation (Property 1).

These tests load a single shared ``state_dict`` into both the reorganized
clean-repo model and the original Source_Repository model, then assert that the
two produce equal outputs (within floating-point tolerance) across random
inputs, for every model in the reconstruction set.

Reference (original) model modules are loaded directly from their file paths to
avoid triggering side-effect imports in the original ``models`` package.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Clean-repo (reorganized) models under test.
from src.models import (
    BlindDisplacementNet,
    FlowDisplacementNet,
    LabelGuidedDisplacementNet,
    LocalReconstructionPCN,
    PurePCN,
    SeedDisplacementNet,
    SeedDisplacementNetNoLocal,
)

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
_src_pure = _load_source_module("orig_pure_pcn", "pure_pcn.py")
_src_lrpcn = _load_source_module("orig_local_reconstruction_pcn", "local_reconstruction_pcn.py")
_src_disp = _load_source_module("orig_displacement_net", "displacement_net.py")

TOL = 1e-5

# Each entry: (clean_repo_class, reference_class, kind)
# kind determines how the model is called and how outputs are compared.
MODEL_SET = [
    (SeedDisplacementNet, _src_seed.SeedDisplacementNet, "seed"),
    (SeedDisplacementNetNoLocal, _src_seed.SeedDisplacementNetNoLocal, "seed"),
    (PurePCN, _src_pure.PurePCN, "pcn"),
    (LocalReconstructionPCN, _src_lrpcn.LocalReconstructionPCN, "lrpcn"),
    (LabelGuidedDisplacementNet, _src_disp.LabelGuidedDisplacementNet, "labelled_disp"),
    (BlindDisplacementNet, _src_disp.BlindDisplacementNet, "blind_disp"),
    (FlowDisplacementNet, _src_disp.FlowDisplacementNet, "flow_disp"),
]


def _make_inputs(batch, num_points, damaged_frac, seed):
    """Build random points + binary labels with a controlled damaged fraction."""
    g = torch.Generator().manual_seed(seed)
    points = torch.randn(batch, num_points, 3, generator=g)
    rand = torch.rand(batch, num_points, generator=g)
    labels = (rand < damaged_frac).float()
    return points, labels


def _run(model, kind, points, labels):
    """Call a model according to its kind and return comparable tensors."""
    if kind == "seed":
        out, recon_list, _ = model(points, labels)
        return [out] + list(recon_list)
    if kind == "pcn":
        coarse, fine = model(points, training=True)
        return [coarse, fine]
    if kind == "lrpcn":
        # Inference mode returns a single merged, deterministic tensor.
        out = model(points, labels, training=False)
        return [out]
    if kind == "labelled_disp":
        out, disp = model(points, labels)
        return [out, disp]
    # blind_disp / flow_disp ignore labels
    out, disp = model(points, labels)
    return [out, disp]


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    model_idx=st.integers(min_value=0, max_value=len(MODEL_SET) - 1),
    batch=st.integers(min_value=1, max_value=3),
    num_points=st.integers(min_value=16, max_value=64),
    damaged_frac=st.floats(min_value=0.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_model_behavior_preservation(model_idx, batch, num_points, damaged_frac, seed):
    """**Feature: repo-reorganization, Property 1: For any reconstruction model in
    the clean repo (SeedDispNet, its NoLocal ablation, PurePCN,
    LocalReconstructionPCN, BlindDisplacementNet, FlowDisplacementNet) and any
    random input point cloud and labels, the reorganized model with a given set
    of weights produces output equal (within floating-point tolerance) to the
    original Source_Repository model with the same weights and inputs.**

    Validates: Requirements 3.2, 3.3, 3.4, 12.4
    """
    clean_cls, ref_cls, kind = MODEL_SET[model_idx]

    # LR-PCN's spatial std needs >1 damaged point to be meaningful; all models
    # remain valid across the full range, so no extra constraints are required.
    points, labels = _make_inputs(batch, num_points, damaged_frac, seed)

    # Build both models and load one shared state_dict into both.
    torch.manual_seed(seed)
    clean_model = clean_cls()
    ref_model = ref_cls()
    ref_model.load_state_dict(clean_model.state_dict())

    # eval() to freeze BatchNorm running stats and disable stochastic layers.
    clean_model.eval()
    ref_model.eval()

    with torch.no_grad():
        clean_out = _run(clean_model, kind, points, labels)
        ref_out = _run(ref_model, kind, points, labels)

    assert len(clean_out) == len(ref_out)
    for c, r in zip(clean_out, ref_out):
        assert c.shape == r.shape
        if c.numel() == 0:
            continue
        max_diff = (c - r).abs().max().item()
        assert max_diff <= TOL, f"{clean_cls.__name__}: max diff {max_diff} > {TOL}"
