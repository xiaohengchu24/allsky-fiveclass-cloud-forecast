from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary = json.loads((ROOT / "results/manuscript_summary.json").read_text(encoding="utf-8"))
    manifest = ROOT / "data/classification/fixed_split_manifest.csv"
    splits: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    rows = 0
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            splits[row["split"]] += 1
            classes[row["class_name"]] += 1
    check(rows == 358434, f"classification manifest rows: {rows}")
    check(splits == {"train": 222441, "val": 76544, "test": 59449}, f"split counts: {splits}")

    expected_index_shapes = {"train": (15921, 8), "val": (3717, 8), "test": (2818, 8)}
    for name, shape in expected_index_shapes.items():
        values = np.load(ROOT / f"data/forecast_indices/{name}.npy", allow_pickle=False)
        check(values.shape == shape, f"{name} index shape: {values.shape}")

    mask = np.load(ROOT / "data/forecast_indices/valid_sky_mask.npy", allow_pickle=False)
    check(mask.shape == (51, 51), f"mask shape: {mask.shape}")
    check(int(mask.sum()) == 1384, f"valid sky positions: {mask.sum()}")

    comparison = list(
        csv.DictReader((ROOT / "results/forecast/persistence_comparison.csv").open(encoding="utf-8"))
    )
    expected = summary["forecast"]
    check([int(row["horizon_min"]) for row in comparison] == expected["horizons_minutes"], "horizons")
    for row, model, baseline, rer in zip(
        comparison,
        expected["model_accuracy"],
        expected["persistence_accuracy"],
        expected["relative_error_reduction"],
    ):
        check(abs(float(row["model_accuracy"]) - model) < 1e-12, "model accuracy")
        check(abs(float(row["persistence_accuracy"]) - baseline) < 1e-12, "persistence accuracy")
        check(abs(float(row["relative_error_reduction"]) - rer) < 1e-12, "relative error reduction")

    runtime = json.loads(
        (ROOT / "results/runtime/local_hardware_runtime.json").read_text(encoding="utf-8")
    )
    check(runtime["benchmark_date"] == "2026-10-22", "runtime benchmark date")
    check(
        runtime["raspberry_pi_4b"]["valid_sky_1384_patch_classification"]
        ["median_seconds"]["monitoring_and_forecasting"] == 131.2181,
        "Raspberry Pi valid-sky runtime",
    )
    check(
        runtime["raspberry_pi_4b"]["output_verification"]
        ["forecast_agreement_with_local_pytorch"] == 1.0,
        "Raspberry Pi forecast-output agreement",
    )

    checksum_path = ROOT / "SHA256SUMS.txt"
    if checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            expected_hash, relative = line.split("  ", 1)
            path = ROOT / relative
            check(path.is_file(), f"missing file: {relative}")
            check(sha256(path) == expected_hash, f"checksum mismatch: {relative}")

    print("Release verification: PASS")


if __name__ == "__main__":
    main()
