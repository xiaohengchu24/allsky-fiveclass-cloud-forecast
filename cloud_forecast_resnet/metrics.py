from __future__ import annotations

import torch


def update_confusion(
    confusion: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int = 5,
) -> None:
    """Accumulate one confusion matrix per forecast horizon on CPU."""
    targets = targets.detach().to(device="cpu", dtype=torch.long)
    predictions = predictions.detach().to(device="cpu", dtype=torch.long)
    for horizon_index in range(targets.shape[1]):
        encoded = targets[:, horizon_index] * num_classes + predictions[:, horizon_index]
        confusion[horizon_index] += torch.bincount(
            encoded, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)


def scores_from_confusion(confusion: torch.Tensor) -> dict[str, list]:
    matrix = confusion.to(dtype=torch.float64)
    total = matrix.sum(dim=(1, 2))
    accuracy = matrix.diagonal(dim1=1, dim2=2).sum(dim=1) / total.clamp_min(1)
    tp = matrix.diagonal(dim1=1, dim2=2)
    precision = tp / matrix.sum(dim=1).clamp_min(1)
    recall = tp / matrix.sum(dim=2).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
    return {
        "accuracy": accuracy.tolist(),
        "macro_f1": f1.mean(dim=1).tolist(),
        "class_f1": f1.tolist(),
        "confusion": confusion.tolist(),
    }
