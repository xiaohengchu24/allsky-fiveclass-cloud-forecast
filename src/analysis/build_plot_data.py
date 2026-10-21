#!/usr/bin/env python3
"""Create generic monthly accuracy tables with calendar-day bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HORIZONS = (5, 10, 20, 30)


def metadata_month_day(metadata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if metadata.dtype.names:
        names = set(metadata.dtype.names)
        month_name = next((x for x in ("month", "months") if x in names), None)
        day_name = next((x for x in ("day", "calendar_day", "date", "day_key") if x in names), None)
        timestamp_name = next(
            (x for x in ("timestamp_minute", "timestamp", "anchor_timestamp", "time") if x in names),
            None,
        )
        if month_name and day_name:
            return metadata[month_name].astype(int), metadata[day_name].astype(str)
        if timestamp_name:
            raw = metadata[timestamp_name]
        else:
            raise ValueError(f"Cannot identify time field in metadata: {metadata.dtype.names}")
    elif metadata.ndim == 2 and metadata.shape[1] >= 3:
        # The packaged forecast evaluator stores [month, calendar_day_index,
        # anchor_timestamp_minute]. Month and calendar-day block are explicit.
        explicit_month = metadata[:, 0].astype(int)
        if np.all((explicit_month >= 1) & (explicit_month <= 12)):
            return explicit_month, metadata[:, 1].astype(np.int64).astype(str)
        raw = metadata[:, 1]
    else:
        raw = metadata

    raw = np.asarray(raw)
    if np.issubdtype(raw.dtype, np.number):
        dt = raw.astype("datetime64[m]")
    else:
        dt = raw.astype("datetime64[m]")
    months = (dt.astype("datetime64[M]").astype(int) % 12 + 1).astype(int)
    days = dt.astype("datetime64[D]").astype(str)
    return months, days


def normalize_prediction_shape(
    values: np.ndarray, n_anchors: int, valid_mask: np.ndarray
) -> np.ndarray:
    """Return arrays as [anchor, horizon, position]."""
    values = np.asarray(values)
    if values.shape[0] == n_anchors and values.ndim == 3 and values.shape[1] == 4:
        return values
    if values.shape[0] == n_anchors and values.ndim == 3 and values.shape[2] == 4:
        return values.transpose(0, 2, 1)
    if values.ndim == 2 and values.shape[0] == n_anchors * 4:
        return values.reshape(n_anchors, 4, -1)
    if values.shape[:2] == (n_anchors, 4) and values.shape[2:] == valid_mask.shape:
        return values[:, :, valid_mask]
    raise ValueError(f"Unsupported prediction shape {values.shape} for {n_anchors} anchors")


def bootstrap_accuracy(
    correct: np.ndarray, days: np.ndarray, rng: np.random.Generator, repeats: int
) -> tuple[float, float, float]:
    """Bootstrap calendar days; each day's estimate is weighted by its pixel count."""
    unique_days, inverse = np.unique(days, return_inverse=True)
    day_correct = np.bincount(inverse, weights=correct.sum(axis=1).astype(float))
    day_total = np.bincount(inverse, weights=np.full(correct.shape[0], correct.shape[1], dtype=float))
    sampled = rng.integers(0, len(unique_days), size=(repeats, len(unique_days)))
    estimates = day_correct[sampled].sum(axis=1) / day_total[sampled].sum(axis=1)
    mean = float(correct.mean())
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return mean, float(lower), float(upper)


def write_wide_table(path: Path, rows: list[dict], metric: str) -> None:
    months = sorted({int(row["month"]) for row in rows})
    lookup = {(int(r["month"]), int(r["horizon_min"])): r for r in rows}
    header = ["Horizon (min)"]
    for month in months:
        header.extend([f"{month:02d} Mean", f"{month:02d} Upper limit", f"{month:02d} Lower limit"])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for horizon in HORIZONS:
            line = [horizon]
            for month in months:
                row = lookup[(month, horizon)]
                line.extend([row[f"{metric}_mean_pct"], row[f"{metric}_upper_pct"], row[f"{metric}_lower_pct"]])
            writer.writerow(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("plot_data"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    test_dir = root / "forecast/results/model/test"
    pred = np.load(test_dir / "predictions_uint8.npy", allow_pickle=False)
    target = np.load(test_dir / "targets_uint8.npy", allow_pickle=False)
    metadata = np.load(test_dir / "anchor_metadata.npy", allow_pickle=True)
    months, days = metadata_month_day(metadata)
    valid_mask = np.load(root / "forecast/data/accepted_valid_sky_mask.npy", allow_pickle=False).astype(bool)
    if valid_mask.shape != (51, 51) or int(valid_mask.sum()) != 1384:
        raise ValueError(
            f"Unexpected valid-sky mask: shape={valid_mask.shape}, valid={valid_mask.sum()}"
        )
    pred = normalize_prediction_shape(pred, len(metadata), valid_mask)
    target = normalize_prediction_shape(target, len(metadata), valid_mask)
    persistence = np.load(
        root / "forecast/results/model/persistence_valid_sky/persistence_predictions_uint8.npy",
        allow_pickle=False,
    )
    persistence = normalize_prediction_shape(persistence, len(metadata), valid_mask)

    print("metadata", metadata.shape, metadata.dtype, metadata.dtype.names, metadata[:5])
    print("prediction", pred.shape, "target", target.shape, "persistence", persistence.shape)
    print("months", {int(m): int((months == m).sum()) for m in np.unique(months)})

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for month in sorted(np.unique(months)):
        selected = months == month
        for h_index, horizon in enumerate(HORIZONS):
            row: dict[str, float | int] = {"month": int(month), "horizon_min": horizon}
            for name, values in (("model", pred), ("persistence", persistence)):
                correct = values[selected, h_index] == target[selected, h_index]
                mean, lower, upper = bootstrap_accuracy(correct, days[selected], rng, args.bootstrap)
                row[f"{name}_mean_pct"] = 100.0 * mean
                row[f"{name}_lower_pct"] = 100.0 * lower
                row[f"{name}_upper_pct"] = 100.0 * upper
            row["gain_mean_pp"] = float(row["model_mean_pct"]) - float(row["persistence_mean_pct"])
            rows.append(row)

    columns = list(rows[0])
    with (output / "monthly_accuracy_with_95ci.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_wide_table(output / "model_accuracy_by_month.csv", rows, "model")
    write_wide_table(output / "persistence_accuracy_by_month.csv", rows, "persistence")
    (output / "calculation_metadata.json").write_text(
        json.dumps(
            {
                "bootstrap_unit": "calendar day within each test month",
                "bootstrap_repeats": args.bootstrap,
                "confidence_interval": "2.5th and 97.5th percentiles",
                "seed": args.seed,
                "horizons_minutes": HORIZONS,
                "months": sorted(int(x) for x in np.unique(months)),
                "anchor_counts": {str(int(m)): int((months == m).sum()) for m in np.unique(months)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
