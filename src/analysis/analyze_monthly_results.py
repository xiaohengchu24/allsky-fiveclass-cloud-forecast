#!/usr/bin/env python3
"""Summarize month-specific target composition and temporal change rates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HORIZONS = (5, 10, 20, 30)
CLASSES = ("Cloudy", "Obstructed", "Partial visibility", "Bright-area interference", "Visible")


def masked(path: Path, mask: np.ndarray) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.ndim == 4:
        return values[:, :, mask]
    return values


def macro_f1(target: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for cls in range(5):
        tp = np.count_nonzero((target == cls) & (pred == cls))
        fp = np.count_nonzero((target != cls) & (pred == cls))
        fn = np.count_nonzero((target == cls) & (pred != cls))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("paper_plot_data"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    mask = np.load(root / "forecast/data/accepted_valid_sky_mask.npy").astype(bool)
    metadata = np.load(root / "forecast/results/model/test/anchor_metadata.npy")
    months = metadata[:, 0].astype(int)
    target = masked(root / "forecast/results/model/test/targets_uint8.npy", mask)
    pred = masked(root / "forecast/results/model/test/predictions_uint8.npy", mask)
    persistence = masked(
        root / "forecast/results/model/persistence_valid_sky/persistence_predictions_uint8.npy", mask
    )

    performance_rows = []
    distribution_rows = []
    for month in sorted(np.unique(months)):
        select = months == month
        for hi, horizon in enumerate(HORIZONS):
            truth = target[select, hi]
            estimate = pred[select, hi]
            base = persistence[select, hi]
            performance_rows.append(
                {
                    "month": int(month),
                    "anchors": int(select.sum()),
                    "horizon_min": horizon,
                    "model_accuracy_pct": float((estimate == truth).mean() * 100),
                    "persistence_accuracy_pct": float((base == truth).mean() * 100),
                    "state_change_rate_pct": float((base != truth).mean() * 100),
                    "model_macro_f1_pct": float(macro_f1(truth, estimate) * 100),
                }
            )
        truth_all = target[select]
        current = persistence[select, 0]
        for cls, name in enumerate(CLASSES):
            distribution_rows.append(
                {
                    "month": int(month),
                    "class_index": cls,
                    "class_name": name,
                    "future_target_fraction_pct": float((truth_all == cls).mean() * 100),
                    "current_state_fraction_pct": float((current == cls).mean() * 100),
                }
            )

    for filename, rows in (
        ("monthly_performance_diagnostics.csv", performance_rows),
        ("monthly_class_distribution.csv", distribution_rows),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    (output / "monthly_diagnostics_metadata.json").write_text(
        json.dumps(
            {
                "source": "finalan_v20261005_1 packaged predictions, targets, persistence predictions, metadata, and valid-sky mask",
                "valid_sky_positions": int(mask.sum()),
                "test_anchors": int(len(metadata)),
                "horizons_minutes": HORIZONS,
                "classes": CLASSES,
                "state_change_rate_definition": "percentage of target positions where persistence prediction differs from the future target",
                "macro_f1_definition": "unweighted mean of the five one-vs-rest class F1 scores",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
