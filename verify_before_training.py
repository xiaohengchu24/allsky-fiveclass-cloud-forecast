from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cloud_forecast_resnet.model import MultiHorizonResNet50
from cloud_forecast_resnet.storage import ExperimentIndex, load_stores
from cloud_forecast_resnet.tensor_builder import DenseLocalTensorBuilder


EXPECTED_ANCHORS = {"train": 9753, "val": 2733, "test": 2088}
EXPECTED_MONTHS = {4, 5, 6, 10, 11, 12}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(
        Path(__file__).with_name("method_contract.json").read_text(encoding="utf-8")
    )
    expected_classifier_hash = (
        "54604ca87382755bc163164b2077475b24b5d2c8eb7348cf3e30ba31a80d20ce"
    )
    actual_hash = sha256(args.classifier_checkpoint)
    if actual_hash.lower() != expected_classifier_hash:
        raise RuntimeError(f"Classifier checkpoint SHA256 mismatch: {actual_hash}")

    stores = load_stores(args.compact_root, preload=False)
    for month, store in stores.items():
        if store.shape[1:] != (51, 51):
            raise RuntimeError(f"Month {month:02d}: unexpected matrix shape {store.shape}")
        sample = np.asarray(store.labels[: min(100, len(store.labels))])
        if sample.min() < 0 or sample.max() > 4:
            raise RuntimeError(f"Month {month:02d}: labels outside [0,4]")

    indices = {
        split: ExperimentIndex(args.index_root / f"{split}.npy", stores)
        for split in EXPECTED_ANCHORS
    }
    for split, expected in EXPECTED_ANCHORS.items():
        if len(indices[split]) != expected:
            raise RuntimeError(f"{split}: expected {expected}, got {len(indices[split])}")
    if set(stores) != EXPECTED_MONTHS:
        raise RuntimeError(f"Unexpected months: {sorted(stores)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = DenseLocalTensorBuilder(51, 51, device=device)
    anchor = indices["train"].anchors[0]
    source = stores[anchor.month].labels
    input_frames = torch.from_numpy(
        np.asarray(source[list(anchor.input_indices)], dtype=np.uint8).copy()
    ).to(device)
    target_frames = torch.from_numpy(
        np.asarray(source[list(anchor.target_indices)], dtype=np.uint8).copy()
    ).to(device)
    inputs, targets = builder.build_chunk(
        input_frames, target_frames, torch.arange(builder.num_positions)
    )
    if inputs.shape != (2601, 5, 54, 54) or targets.shape != (2601, 4):
        raise RuntimeError(f"Actual tensor mismatch: {inputs.shape}, {targets.shape}")
    if not torch.all(inputs.sum(dim=1) == 1):
        raise RuntimeError("Every input cell must be a valid five-class one-hot vector")
    if not torch.all(inputs[0, 1, :9, :] == 1):
        raise RuntimeError("Top boundary is not padded as Obstructed (class 1)")
    if not torch.all(inputs[0, 1, :, :9] == 1):
        raise RuntimeError("Left boundary is not padded as Obstructed (class 1)")
    model = MultiHorizonResNet50(dropout=0.0).to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        output = model(inputs[:2])
    if output.shape != (2, 4, 5):
        raise RuntimeError(f"Model output mismatch: {output.shape}")

    report = {
        "result": "PASS",
        "classifier_sha256": actual_hash,
        "compact_month_shapes": {str(k): list(v.shape) for k, v in stores.items()},
        "anchor_counts": {k: len(v) for k, v in indices.items()},
        "anchors_by_month": {k: v.counts_by_month() for k, v in indices.items()},
        "dense_positions_per_anchor": builder.num_positions,
        "actual_input_shape": list(inputs.shape),
        "actual_target_shape": list(targets.shape),
        "model_output_shape": list(output.shape),
        "prediction_architecture": "shared_resnet50_four_5class_heads",
        "boundary_padding_class": 1,
        "boundary_padding_name": "Obstructed",
        "method_contract": contract,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
