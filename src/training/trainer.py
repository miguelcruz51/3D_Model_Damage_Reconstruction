"""Shared training loop and per-model adapters.

This module extracts the near-identical epoch loop, gradient clipping, Adam +
StepLR optimization, CSV logging, best-checkpoint saving, and curve plotting
that were duplicated across the original ``train_*`` scripts. Each model family
keeps its exact forward-pass and loss call pattern through a small
:class:`ModelAdapter`, so numerical training behavior is preserved (Req 12.4).
"""

from __future__ import annotations

import csv
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DMCLocalReconstruction
from src.losses import (
    DisplacementLoss,
    LabelAwareLoss,
    PurePCNLoss,
    SeedDisplacementLoss,
)
from src.models import (
    BlindDisplacementNet,
    FlowDisplacementNet,
    LocalReconstructionPCN,
    PurePCN,
    SeedDisplacementNet,
    SeedDisplacementNetNoLocal,
)


class ModelAdapter:
    """Bundles a model, loss, dataset options, and their call patterns.

    An adapter isolates the parts that differ between the original ``train_*``
    scripts: which model/loss to build, the exact ``forward`` and criterion
    calls, the metric keys logged per epoch, and the dataset flags used. The
    trainer itself is model-agnostic.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        step_fn: Callable[[torch.nn.Module, torch.nn.Module, tuple, torch.device], tuple],
        metric_keys: List[str],
        dataset_kwargs: Dict[str, Any],
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.step_fn = step_fn
        self.metric_keys = metric_keys
        self.dataset_kwargs = dataset_kwargs


# ---------------------------------------------------------------------------
# Per-model step functions. Each returns (loss_tensor, metrics_dict) and mirrors
# the exact forward/criterion call from the original training script.
# ---------------------------------------------------------------------------

def _seed_step(model, criterion, batch, device):
    damaged_pts, damaged_lbl, gt_pts, gt_lbl, _ = batch
    damaged_pts = damaged_pts.to(device)
    damaged_lbl = damaged_lbl.to(device)
    gt_pts = gt_pts.to(device)
    gt_lbl = gt_lbl.to(device)
    output, recon_list, masks = model(damaged_pts, damaged_lbl)
    loss, metrics = criterion(output, recon_list, masks, gt_pts, gt_lbl)
    return loss, metrics


def _pure_pcn_step(model, criterion, batch, device):
    damaged_pts, _damaged_lbl, gt_pts, _gt_lbl, _ = batch
    damaged_pts = damaged_pts.to(device)
    gt_pts = gt_pts.to(device)
    coarse, fine = model(damaged_pts, training=True)
    loss, metrics = criterion(coarse, fine, gt_pts)
    return loss, metrics


def _lrpcn_step(model, criterion, batch, device):
    damaged_pts, damaged_lbl, gt_pts, gt_lbl, _ = batch
    damaged_pts = damaged_pts.to(device)
    damaged_lbl = damaged_lbl.to(device)
    gt_pts = gt_pts.to(device)
    gt_lbl = gt_lbl.to(device)
    intact_points_list, reconstructed_points = model(damaged_pts, damaged_lbl, training=True)
    loss, metrics = criterion(intact_points_list, reconstructed_points, gt_pts, gt_lbl)
    return loss, metrics


def _displacement_step(model, criterion, batch, device):
    damaged_pts, damaged_lbl, gt_pts, gt_lbl, _ = batch
    damaged_pts = damaged_pts.to(device)
    damaged_lbl = damaged_lbl.to(device)
    gt_pts = gt_pts.to(device)
    gt_lbl = gt_lbl.to(device)
    output_pts, displacements = model(damaged_pts, damaged_lbl)
    loss, metrics = criterion(output_pts, displacements, gt_pts, gt_lbl)
    return loss, metrics


def build_adapter(config) -> ModelAdapter:
    """Construct the :class:`ModelAdapter` selected by ``config.model``.

    Model constructor parameters, loss configuration, dataset flags, and the
    exact forward/loss call pattern all match the corresponding original
    ``train_*`` script for the chosen model.

    Args:
        config: A validated configuration object exposing hyperparameters.

    Returns:
        A fully constructed :class:`ModelAdapter`.

    Raises:
        ValueError: If ``config.model`` is not a recognized identifier.
    """
    model_name = config.model
    latent_dim = int(config.get("latent_dim", 512))
    feature_dim = int(config.get("feature_dim", 256))
    num_points = int(config.num_points)

    if model_name == "seeddispnet":
        if bool(config.get("no_local", False)):
            model = SeedDisplacementNetNoLocal(latent_dim=latent_dim, feature_dim=feature_dim)
        else:
            model = SeedDisplacementNet(latent_dim=latent_dim, feature_dim=feature_dim)
        criterion = SeedDisplacementLoss(
            use_cd=bool(config.get("use_cd", False)),
            use_squared_cd=bool(config.get("use_squared_cd", False)),
        )
        return ModelAdapter(
            model=model,
            criterion=criterion,
            step_fn=_seed_step,
            metric_keys=["loss", "damaged_loss", "intact_loss"],
            dataset_kwargs={"augment": True, "only_damaged": True},
        )

    if model_name == "pure_pcn":
        model = PurePCN(num_points=num_points, latent_dim=latent_dim)
        criterion = PurePCNLoss(coarse_weight=float(config.get("coarse_weight", 0.5)))
        return ModelAdapter(
            model=model,
            criterion=criterion,
            step_fn=_pure_pcn_step,
            metric_keys=["loss", "cd_coarse", "cd_fine"],
            dataset_kwargs={"augment": True, "only_damaged": True, "erase_damaged": True},
        )

    if model_name == "lrpcn":
        model = LocalReconstructionPCN(
            num_points=num_points,
            latent_dim=latent_dim,
            predict_ratio=float(config.get("predict_ratio", 0.5)),
        )
        criterion = LabelAwareLoss(
            coverage_weight=float(config.get("coverage_weight", 0.5)),
            imbalance_aware=bool(config.get("imbalance_aware", True)),
        )
        return ModelAdapter(
            model=model,
            criterion=criterion,
            step_fn=_lrpcn_step,
            metric_keys=["loss", "cd_reconstruction", "coverage"],
            dataset_kwargs={"augment": True, "only_damaged": True, "erase_damaged": False},
        )

    if model_name in ("blind_displacement", "flow_displacement"):
        if model_name == "blind_displacement":
            model = BlindDisplacementNet(feature_dim=feature_dim)
        else:
            model = FlowDisplacementNet(feature_dim=feature_dim)
        # Both baselines penalize moving intact points with weight 0.1, matching
        # the original train_displacement.py wiring.
        criterion = DisplacementLoss(intact_weight=0.1)
        return ModelAdapter(
            model=model,
            criterion=criterion,
            step_fn=_displacement_step,
            metric_keys=["loss", "damaged_loss", "intact_loss", "mean_displacement"],
            dataset_kwargs={"augment": True, "only_damaged": True},
        )

    raise ValueError(f"Unknown model '{model_name}'")


def _run_epoch(adapter, loader, optimizer, device, epoch, train):
    """Run a single train or validation epoch and return averaged metrics."""
    model = adapter.model
    model.train() if train else model.eval()
    totals = {key: 0.0 for key in adapter.metric_keys}

    desc = f"Epoch {epoch}" if train else "Validating"
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=desc)
        for batch in pbar:
            loss, metrics = adapter.step_fn(model, adapter.criterion, batch, device)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            for key in adapter.metric_keys:
                totals[key] += metrics[key]
            pbar.set_postfix({"loss": f"{metrics['loss']:.6f}"})

    n = len(loader)
    return {key: totals[key] / n for key in adapter.metric_keys}


def _plot_curves(history, metric_keys, log_dir):
    """Save total-loss and learning-rate training curves to ``log_dir``."""
    import matplotlib

    matplotlib.use("Agg")  # headless plotting for training environments
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    best_idx = int(np.argmin(history["val_loss"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train")
    ax.plot(epochs, history["val_loss"], "r-", label="Val")
    ax.plot(epochs[best_idx], history["val_loss"][best_idx], "r*", markersize=15)
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["learning_rate"], "g-")
    ax.set_title("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], "b-", label="Train")
    plt.plot(epochs, history["val_loss"], "r-", label="Val")
    plt.plot(
        epochs[best_idx],
        history["val_loss"][best_idx],
        "r*",
        markersize=15,
        label=f"Best (Epoch {epochs[best_idx]})",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def build_dataloaders(config, adapter) -> Tuple[DataLoader, DataLoader]:
    """Build train/valid dataloaders using the adapter's dataset flags.

    Returns:
        A ``(train_loader, val_loader)`` tuple.

    Raises:
        RuntimeError: If the training split contains no samples.
    """
    train_dataset = DMCLocalReconstruction(
        data_root=config.data_root,
        split="train",
        num_points=int(config.num_points),
        **{k: v for k, v in adapter.dataset_kwargs.items() if k != "augment"},
        augment=True,
    )
    val_dataset = DMCLocalReconstruction(
        data_root=config.data_root,
        split="valid",
        num_points=int(config.num_points),
        **{k: v for k, v in adapter.dataset_kwargs.items() if k != "augment"},
        augment=False,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError("No training samples found; check data_root and split lists.")

    num_workers = int(config.get("num_workers", 4))
    batch_size = int(config.batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader


def train(config) -> str:
    """Run the full training routine for the model selected by ``config``.

    Handles device selection, dataloader construction, optimization with Adam +
    StepLR, gradient clipping, per-epoch CSV logging, best-checkpoint saving,
    and curve plotting.

    Args:
        config: A validated configuration object.

    Returns:
        The filesystem path to the best-model checkpoint.
    """
    gpu = int(config.get("gpu", 0))
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)

    adapter = build_adapter(config)
    adapter.model = adapter.model.to(device)
    params = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    print(f"Model: {config.model}, parameters: {params:,}")

    train_loader, val_loader = build_dataloaders(config, adapter)

    optimizer = optim.Adam(
        adapter.model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config.lr_decay_step),
        gamma=float(config.lr_decay_rate),
    )

    header = ["epoch"]
    header += [f"train_{k}" for k in adapter.metric_keys]
    header += [f"val_{k}" for k in adapter.metric_keys]
    header += ["learning_rate"]

    csv_path = os.path.join(log_dir, "training_log.csv")
    # "loss" is always the first metric key, so history["train_loss"] and
    # history["val_loss"] used for plotting are populated by this loop.
    history = {"epoch": [], "learning_rate": []}
    for key in adapter.metric_keys:
        history[f"train_{key}"] = []
        history[f"val_{key}"] = []

    best_loss = float("inf")
    best_epoch = 0
    checkpoint_path = os.path.join(log_dir, "best_model.pth")

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for epoch in range(1, int(config.epochs) + 1):
            train_m = _run_epoch(adapter, train_loader, optimizer, device, epoch, train=True)
            val_m = _run_epoch(adapter, val_loader, optimizer, device, epoch, train=False)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            row = [epoch]
            row += [train_m[k] for k in adapter.metric_keys]
            row += [val_m[k] for k in adapter.metric_keys]
            row += [lr]
            writer.writerow(row)
            csv_file.flush()

            history["epoch"].append(epoch)
            history["learning_rate"].append(lr)
            for key in adapter.metric_keys:
                history[f"train_{key}"].append(train_m[key])
                history[f"val_{key}"].append(val_m[key])

            print(f"Epoch {epoch}/{config.epochs}: "
                  f"train_loss={train_m['loss']:.6f}, val_loss={val_m['loss']:.6f}")

            if val_m["loss"] < best_loss:
                best_loss = val_m["loss"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": adapter.model.state_dict(),
                        "val_loss": val_m["loss"],
                        "config": config.to_dict(),
                    },
                    checkpoint_path,
                )
                print(f"  Saved best (val_loss={val_m['loss']:.6f})")

    print(f"Done. Best epoch {best_epoch}, val_loss {best_loss:.6f}")
    _plot_curves(history, adapter.metric_keys, log_dir)
    return checkpoint_path
