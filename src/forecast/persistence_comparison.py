from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HORIZONS = (5, 10, 20, 30)
CLASS_NAMES = (
    "Cloudy",
    "Obstructed",
    "Partial visibility",
    "Bright-area interference",
    "Visible",
)
TEMPORARILY_AFFECTED = (0, 2, 3)


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    result = np.zeros((len(HORIZONS), 5, 5), dtype=np.int64)
    for horizon in range(len(HORIZONS)):
        encoded = targets[:, horizon].reshape(-1).astype(np.int64) * 5
        encoded += predictions[:, horizon].reshape(-1).astype(np.int64)
        result[horizon] = np.bincount(encoded, minlength=25).reshape(5, 5)
    return result


def scores(confusion: np.ndarray) -> dict[str, list]:
    total = confusion.sum(axis=(1, 2))
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    accuracy = np.divide(diagonal.sum(axis=1), total, where=total > 0)
    support = confusion.sum(axis=2)
    predicted = confusion.sum(axis=1)
    precision = np.divide(
        diagonal,
        predicted,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=predicted > 0,
    )
    recall = np.divide(
        diagonal,
        support,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=support > 0,
    )
    class_f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": accuracy.tolist(),
        "macro_f1": class_f1.mean(axis=1).tolist(),
        "class_f1": class_f1.tolist(),
    }


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    truth = truth.astype(bool, copy=False)
    predicted = predicted.astype(bool, copy=False)
    tp = int(np.logical_and(truth, predicted).sum())
    fp = int(np.logical_and(~truth, predicted).sum())
    fn = int(np.logical_and(truth, ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "events": int(truth.sum()),
        "predicted_events": int(predicted.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def transition_metrics(
    latest: np.ndarray, targets: np.ndarray, predictions: np.ndarray
) -> dict[str, list[dict[str, float | int]]]:
    result: dict[str, list[dict[str, float | int]]] = {
        "any_state_change": [],
        "visible_to_temporarily_affected": [],
        "temporarily_affected_to_visible": [],
    }
    latest_affected = np.isin(latest, TEMPORARILY_AFFECTED)
    latest_visible = latest == 4
    for horizon in range(len(HORIZONS)):
        target = targets[:, horizon]
        predicted = predictions[:, horizon]
        result["any_state_change"].append(
            binary_metrics(target != latest, predicted != latest)
        )
        result["visible_to_temporarily_affected"].append(
            binary_metrics(
                latest_visible & np.isin(target, TEMPORARILY_AFFECTED),
                latest_visible & np.isin(predicted, TEMPORARILY_AFFECTED),
            )
        )
        result["temporarily_affected_to_visible"].append(
            binary_metrics(
                latest_affected & (target == 4),
                latest_affected & (predicted == 4),
            )
        )
    return result


def paired_day_bootstrap(
    targets: np.ndarray,
    model_predictions: np.ndarray,
    persistence_predictions: np.ndarray,
    metadata: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, list[list[float]]]:
    day_keys = metadata[:, 0].astype(np.int64) * 100 + metadata[:, 1].astype(np.int64)
    unique_days = np.unique(day_keys)
    daily_model = []
    daily_persistence = []
    for key in unique_days:
        selected = day_keys == key
        daily_model.append(confusion_matrix(targets[selected], model_predictions[selected]))
        daily_persistence.append(
            confusion_matrix(targets[selected], persistence_predictions[selected])
        )
    daily_model = np.stack(daily_model)
    daily_persistence = np.stack(daily_persistence)
    rng = np.random.default_rng(seed)
    delta_accuracy = []
    delta_macro_f1 = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(unique_days), size=len(unique_days))
        model_score = scores(daily_model[chosen].sum(axis=0))
        persistence_score = scores(daily_persistence[chosen].sum(axis=0))
        delta_accuracy.append(
            np.asarray(model_score["accuracy"])
            - np.asarray(persistence_score["accuracy"])
        )
        delta_macro_f1.append(
            np.asarray(model_score["macro_f1"])
            - np.asarray(persistence_score["macro_f1"])
        )
    return {
        "calendar_days": int(len(unique_days)),
        "delta_accuracy_95ci": np.quantile(
            np.asarray(delta_accuracy), [0.025, 0.975], axis=0
        ).T.tolist(),
        "delta_macro_f1_95ci": np.quantile(
            np.asarray(delta_macro_f1), [0.025, 0.975], axis=0
        ).T.tolist(),
    }


def comparison_summary(
    targets: np.ndarray,
    model_predictions: np.ndarray,
    persistence_predictions: np.ndarray,
) -> dict:
    model_scores = scores(confusion_matrix(targets, model_predictions))
    persistence_scores = scores(confusion_matrix(targets, persistence_predictions))
    model_accuracy = np.asarray(model_scores["accuracy"])
    persistence_accuracy = np.asarray(persistence_scores["accuracy"])
    model_macro_f1 = np.asarray(model_scores["macro_f1"])
    persistence_macro_f1 = np.asarray(persistence_scores["macro_f1"])
    persistence_error = 1.0 - persistence_accuracy
    error_reduction = np.divide(
        model_accuracy - persistence_accuracy,
        persistence_error,
        out=np.zeros_like(model_accuracy),
        where=persistence_error > 0,
    )
    return {
        "model": model_scores,
        "persistence": persistence_scores,
        "delta_accuracy": (model_accuracy - persistence_accuracy).tolist(),
        "delta_macro_f1": (model_macro_f1 - persistence_macro_f1).tolist(),
        "relative_error_reduction": error_reduction.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spatial-mask",
        type=Path,
        default=None,
        help=(
            "Optional 51x51 boolean mask. When supplied, all reported model and "
            "persistence metrics are computed only at True positions."
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=420923)
    args = parser.parse_args()

    rows = np.load(args.test_index, allow_pickle=False)
    model_predictions = np.load(
        args.test_output / "predictions_uint8.npy", mmap_mode="r"
    )
    targets = np.load(args.test_output / "targets_uint8.npy", mmap_mode="r")
    metadata = np.load(args.test_output / "anchor_metadata.npy", allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise RuntimeError(f"Unexpected test-index shape: {rows.shape}")
    expected_shape = (len(rows), len(HORIZONS), 51, 51)
    if model_predictions.shape != expected_shape:
        raise RuntimeError(f"Unexpected prediction shape: {model_predictions.shape}")
    if targets.shape != model_predictions.shape:
        raise RuntimeError("Prediction/target shape mismatch")
    expected_metadata = rows[:, [0, 6, 7]].astype(np.int64)
    if not np.array_equal(metadata, expected_metadata):
        raise RuntimeError("Saved predictions do not align with the fixed test index")

    latest = np.empty((len(rows), 51, 51), dtype=np.uint8)
    verified_targets = np.empty_like(targets)
    for month in sorted(np.unique(rows[:, 0])):
        labels = np.load(args.compact_root / f"{int(month):02d}" / "labels.npy", mmap_mode="r")
        timestamps = np.load(
            args.compact_root / f"{int(month):02d}" / "timestamps.npy", mmap_mode="r"
        )
        timestamp_to_index = {int(value): index for index, value in enumerate(timestamps)}
        selected_rows = np.flatnonzero(rows[:, 0] == month)
        for output_index in selected_rows:
            anchor_time = int(rows[output_index, 7])
            latest[output_index] = labels[timestamp_to_index[anchor_time]]
            for horizon_index, horizon in enumerate(HORIZONS):
                verified_targets[output_index, horizon_index] = labels[
                    timestamp_to_index[anchor_time + horizon]
                ]
    if not np.array_equal(targets, verified_targets):
        raise RuntimeError("Saved targets do not match compact matrices and fixed indices")
    persistence_predictions = np.repeat(latest[:, None, :, :], len(HORIZONS), axis=1)

    evaluated_positions = 2601
    mask_definition = "full 51x51 prediction grid"
    if args.spatial_mask is not None:
        spatial_mask = np.load(args.spatial_mask, allow_pickle=False).astype(bool)
        if spatial_mask.shape != (51, 51):
            raise RuntimeError(
                f"Expected a 51x51 spatial mask, got {spatial_mask.shape}"
            )
        evaluated_positions = int(spatial_mask.sum())
        if evaluated_positions == 0:
            raise RuntimeError("Spatial mask contains no valid positions")
        targets = targets[:, :, spatial_mask]
        model_predictions = model_predictions[:, :, spatial_mask]
        persistence_predictions = persistence_predictions[:, :, spatial_mask]
        latest = latest[:, spatial_mask]
        mask_definition = str(args.spatial_mask.resolve())

    overall = comparison_summary(targets, model_predictions, persistence_predictions)
    overall["paired_calendar_day_bootstrap"] = paired_day_bootstrap(
        targets,
        model_predictions,
        persistence_predictions,
        metadata,
        args.bootstrap_repeats,
        args.seed,
    )
    overall["model_transition_metrics"] = transition_metrics(
        latest, targets, model_predictions
    )
    overall["persistence_transition_metrics"] = transition_metrics(
        latest, targets, persistence_predictions
    )

    monthly = {}
    for month in sorted(np.unique(metadata[:, 0])):
        selected = metadata[:, 0] == month
        item = comparison_summary(
            targets[selected],
            model_predictions[selected],
            persistence_predictions[selected],
        )
        item["anchors"] = int(selected.sum())
        item["paired_calendar_day_bootstrap"] = paired_day_bootstrap(
            targets[selected],
            model_predictions[selected],
            persistence_predictions[selected],
            metadata[selected],
            args.bootstrap_repeats,
            args.seed + int(month),
        )
        monthly[str(int(month))] = item

    result = {
        "definition": {
            "persistence_source": "current observed state at t",
            "horizons_minutes": list(HORIZONS),
            "classes": list(CLASS_NAMES),
            "test_anchors": int(len(rows)),
            "positions_per_anchor": evaluated_positions,
            "spatial_evaluation_mask": mask_definition,
            "bootstrap_unit": "calendar day, paired between methods",
            "bootstrap_repeats": int(args.bootstrap_repeats),
        },
        "verification": {
            "metadata_matches_fixed_test_index": True,
            "saved_targets_match_compact_matrices": True,
            "spatial_sampling": "none; fixed evaluation mask applied uniformly to both methods",
        },
        "overall": overall,
        "monthly": monthly,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "persistence_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "main_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "horizon_min",
                "persistence_accuracy",
                "model_accuracy",
                "delta_accuracy",
                "delta_accuracy_ci95_low",
                "delta_accuracy_ci95_high",
                "relative_error_reduction",
                "persistence_macro_f1",
                "model_macro_f1",
                "delta_macro_f1",
                "delta_macro_f1_ci95_low",
                "delta_macro_f1_ci95_high",
            ]
        )
        boot = overall["paired_calendar_day_bootstrap"]
        for index, horizon in enumerate(HORIZONS):
            writer.writerow(
                [
                    horizon,
                    overall["persistence"]["accuracy"][index],
                    overall["model"]["accuracy"][index],
                    overall["delta_accuracy"][index],
                    boot["delta_accuracy_95ci"][index][0],
                    boot["delta_accuracy_95ci"][index][1],
                    overall["relative_error_reduction"][index],
                    overall["persistence"]["macro_f1"][index],
                    overall["model"]["macro_f1"][index],
                    overall["delta_macro_f1"][index],
                    boot["delta_macro_f1_95ci"][index][0],
                    boot["delta_macro_f1_95ci"][index][1],
                ]
            )

    np.save(args.output / "persistence_predictions_uint8.npy", persistence_predictions)
    print(json.dumps(result["definition"], ensure_ascii=False, indent=2))
    print((args.output / "main_comparison.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
