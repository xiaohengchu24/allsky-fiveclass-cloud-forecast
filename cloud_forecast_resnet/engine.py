from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .metrics import update_confusion
from .storage import Anchor, CompactMonth
from .tensor_builder import DenseLocalTensorBuilder


def iter_position_chunks(
    num_positions: int,
    chunk_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterable[torch.Tensor]:
    positions = (
        torch.randperm(num_positions, generator=generator)
        if shuffle
        else torch.arange(num_positions)
    )
    yield from positions.split(chunk_size)


def shuffled_anchor_indices(length: int, seed: int, epoch: int) -> list[int]:
    values = list(range(length))
    random.Random(seed + epoch).shuffle(values)
    return values


def load_anchor_frames(
    anchor: Anchor,
    stores: dict[int, CompactMonth],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = stores[anchor.month].labels
    inputs = np.asarray(labels[list(anchor.input_indices)], dtype=np.uint8).copy()
    targets = np.asarray(labels[list(anchor.target_indices)], dtype=np.uint8).copy()
    return (
        torch.from_numpy(inputs).to(device=device, non_blocking=True),
        torch.from_numpy(targets).to(device=device, non_blocking=True),
    )


def multi_horizon_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    losses = [
        criterion(logits[:, horizon], targets[:, horizon])
        for horizon in range(logits.shape[1])
    ]
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate_anchors(
    model: nn.Module,
    anchors: list[Anchor],
    stores: dict[int, CompactMonth],
    builder: DenseLocalTensorBuilder,
    chunk_size: int,
    criterion: nn.Module | None = None,
    save_predictions: bool = False,
) -> tuple[float | None, torch.Tensor, np.ndarray | None, np.ndarray | None]:
    model.eval()
    horizons = len(anchors[0].target_indices)
    confusion = torch.zeros((horizons, 5, 5), dtype=torch.int64)
    predictions_output = (
        np.empty((len(anchors), horizons, 51, 51), dtype=np.uint8)
        if save_predictions
        else None
    )
    targets_output = np.empty_like(predictions_output) if save_predictions else None
    total_loss = 0.0
    total_items = 0

    for anchor_number, anchor in enumerate(anchors):
        input_frames, target_frames = load_anchor_frames(anchor, stores, builder.device)
        for positions in iter_position_chunks(builder.num_positions, chunk_size, False):
            inputs, targets = builder.build_chunk(input_frames, target_frames, positions)
            with torch.autocast(device_type=builder.device.type, enabled=builder.device.type == "cuda"):
                logits = model(inputs)
                loss = multi_horizon_loss(logits, targets, criterion) if criterion else None
            predictions = logits.argmax(dim=2)
            update_confusion(confusion, targets, predictions)
            if loss is not None:
                total_loss += float(loss) * len(positions)
                total_items += len(positions)
            if predictions_output is not None and targets_output is not None:
                flat_positions = positions.numpy()
                for horizon in range(horizons):
                    predictions_output[anchor_number, horizon].reshape(-1)[flat_positions] = (
                        predictions[:, horizon].cpu().numpy().astype(np.uint8)
                    )
                    targets_output[anchor_number, horizon].reshape(-1)[flat_positions] = (
                        targets[:, horizon].cpu().numpy().astype(np.uint8)
                    )
    mean_loss = total_loss / total_items if total_items else None
    return mean_loss, confusion, predictions_output, targets_output
