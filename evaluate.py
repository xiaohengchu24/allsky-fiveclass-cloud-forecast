from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cloud_forecast_resnet.engine import evaluate_anchors
from cloud_forecast_resnet.metrics import scores_from_confusion
from cloud_forecast_resnet.model import MultiHorizonResNet50
from cloud_forecast_resnet.storage import ExperimentIndex, load_stores
from cloud_forecast_resnet.tensor_builder import DenseLocalTensorBuilder


def confusion_from_arrays(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int = 5
) -> torch.Tensor:
    result = torch.zeros((targets.shape[1], num_classes, num_classes), dtype=torch.int64)
    for horizon in range(targets.shape[1]):
        encoded = (
            targets[:, horizon].reshape(-1).astype(np.int64) * num_classes
            + predictions[:, horizon].reshape(-1).astype(np.int64)
        )
        result[horizon] = torch.from_numpy(
            np.bincount(encoded, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes
            )
        )
    return result


def block_bootstrap_by_day(
    targets: np.ndarray,
    predictions: np.ndarray,
    metadata: np.ndarray,
    repeats: int = 1000,
    seed: int = 42,
) -> dict[str, list[list[float]]]:
    """Calendar-day block bootstrap confidence intervals for each horizon."""
    rng = np.random.default_rng(seed)
    unique_days = np.unique(metadata[:, 1])
    day_confusions = []
    for day in unique_days:
        selected = metadata[:, 1] == day
        day_confusions.append(confusion_from_arrays(targets[selected], predictions[selected]))
    samples_accuracy = []
    samples_macro_f1 = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(day_confusions), size=len(day_confusions))
        combined = sum((day_confusions[index] for index in chosen), torch.zeros_like(day_confusions[0]))
        scores = scores_from_confusion(combined)
        samples_accuracy.append(scores["accuracy"])
        samples_macro_f1.append(scores["macro_f1"])
    return {
        "accuracy_95ci": np.quantile(samples_accuracy, [0.025, 0.975], axis=0).T.tolist(),
        "macro_f1_95ci": np.quantile(samples_macro_f1, [0.025, 0.975], axis=0).T.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda")
    stores = load_stores(cfg["compact_root"], preload=True)
    test_index = ExperimentIndex(Path(cfg["index_root"]) / "test.npy", stores)
    if len(test_index) != int(cfg["expected_anchor_counts"]["test"]):
        raise RuntimeError(f"Expected 2088 test anchors, got {len(test_index)}")
    sample_store = stores[test_index.anchors[0].month]
    builder = DenseLocalTensorBuilder(sample_store.shape[1], sample_store.shape[2], device=device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = MultiHorizonResNet50(dropout=float(cfg["dropout"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, memory_format=torch.channels_last)
    _, confusion, predictions, targets = evaluate_anchors(
        model,
        test_index.anchors,
        stores,
        builder,
        int(cfg["eval_chunk_size"]),
        save_predictions=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "predictions_uint8.npy", predictions)
    np.save(args.output / "targets_uint8.npy", targets)
    metadata = np.asarray(
        [[a.month, a.day, a.timestamp_minute] for a in test_index.anchors], dtype=np.int64
    )
    np.save(args.output / "anchor_metadata.npy", metadata)
    monthly: dict[str, dict] = {}
    for month in sorted(np.unique(metadata[:, 0])):
        selected = metadata[:, 0] == month
        month_targets = targets[selected]
        month_predictions = predictions[selected]
        month_metadata = metadata[selected]
        month_confusion = confusion_from_arrays(month_targets, month_predictions)
        monthly[str(int(month))] = {
            "anchors": int(selected.sum()),
            "scores": scores_from_confusion(month_confusion),
            "calendar_day_block_bootstrap": block_bootstrap_by_day(
                month_targets,
                month_predictions,
                month_metadata,
                repeats=1000,
                seed=int(cfg["seed"]) + int(month),
            ),
        }
    result = {
        "scores": scores_from_confusion(confusion),
        "anchors": len(test_index),
        "anchors_by_month": test_index.counts_by_month(),
        "positions_per_anchor": builder.num_positions,
        "horizons_minutes": [1, 5, 10, 15],
        "monthly": monthly,
        "bootstrap_repeats": 1000,
        "bootstrap_unit": "calendar_day",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
