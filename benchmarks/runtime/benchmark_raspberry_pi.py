from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Not available on Windows; the benchmark target is Linux.
    resource = None

import cv2
import numpy as np
from PIL import Image


PATCH_SIZE = 56
MATRIX_ROI = (0, 51, 12, 63)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the released monitoring and forecasting models on a CPU."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--classification-onnx", type=Path, required=True)
    parser.add_argument("--forecast-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--classification-batch-size", type=int, default=64)
    parser.add_argument("--forecast-batch-size", type=int, default=32)
    return parser.parse_args()


def load_network(path: Path):
    started = time.perf_counter()
    network = cv2.dnn.readNetFromONNX(str(path))
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return network, time.perf_counter() - started


def run_network(network, values):
    network.setInput(values)
    return network.forward()


def prepare_image(image_path: Path, mask: np.ndarray, valid_only: bool):
    started = time.perf_counter()
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    rows, cols = padded.shape[0] // PATCH_SIZE, padded.shape[1] // PATCH_SIZE
    patch_grid = padded.reshape(rows, PATCH_SIZE, cols, PATCH_SIZE, 3).transpose(
        0, 2, 4, 1, 3
    )
    if valid_only:
        row_start, row_end, col_start, col_end = MATRIX_ROI
        patch_grid = patch_grid[row_start:row_end, col_start:col_end]
        patches = patch_grid.reshape(-1, 3, PATCH_SIZE, PATCH_SIZE)[
            np.flatnonzero(mask.reshape(-1))
        ]
    else:
        patches = patch_grid.reshape(-1, 3, PATCH_SIZE, PATCH_SIZE)
    patches = patches.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((patches - mean) / std), time.perf_counter() - started


def classify(network, image_path, mask, valid_only, batch_size):
    patches, preprocessing_seconds = prepare_image(image_path, mask, valid_only)
    started = time.perf_counter()
    for begin in range(0, len(patches), batch_size):
        run_network(network, patches[begin:begin + batch_size])
    inference_seconds = time.perf_counter() - started
    return preprocessing_seconds, inference_seconds


def build_forecast_inputs(history: np.ndarray, mask: np.ndarray, chunk_size: int):
    started = time.perf_counter()
    if history.shape != (25, 51, 51):
        raise ValueError(f"Expected history shape (25,51,51), got {history.shape}")
    if mask.shape != (51, 51):
        raise ValueError(f"Expected mask shape (51,51), got {mask.shape}")
    positions = np.flatnonzero(mask.reshape(-1))
    rows, cols = np.divmod(positions, 51)
    delta = np.arange(-9, 9, dtype=np.int64)
    rr = np.broadcast_to(rows[:, None, None] + delta[None, :, None], (len(rows), 18, 18))
    cc = np.broadcast_to(cols[:, None, None] + delta[None, None, :], (len(rows), 18, 18))
    in_bounds = (rr >= 0) & (rr < 51) & (cc >= 0) & (cc < 51)
    neighbours = np.clip(rr, 0, 50) * 51 + np.clip(cc, 0, 50)
    valid = in_bounds & mask.reshape(-1)[neighbours]
    flat_history = history.reshape(25, -1)
    output = np.empty((len(positions), 5, 90, 90), dtype=np.float32)
    eye = np.eye(5, dtype=np.float32)
    for begin in range(0, len(positions), chunk_size):
        end = min(begin + chunk_size, len(positions))
        gathered = flat_history[:, neighbours[begin:end].reshape(-1)]
        gathered = gathered.reshape(25, end - begin, 18, 18)
        gathered = np.where(valid[begin:end][None, ...], gathered, 1)
        output[begin:end] = (
            eye[gathered]
            .reshape(5, 5, end - begin, 18, 18, 5)
            .transpose(2, 5, 0, 3, 1, 4)
            .reshape(end - begin, 5, 90, 90)
        )
    return output, time.perf_counter() - started


def forecast(network, history, mask, batch_size):
    inputs, construction_seconds = build_forecast_inputs(history, mask, batch_size)
    started = time.perf_counter()
    for begin in range(0, len(inputs), batch_size):
        run_network(network, inputs[begin:begin + batch_size])
    inference_seconds = time.perf_counter() - started
    return construction_seconds, inference_seconds


def summary(values):
    result = {
        "runs_seconds": [round(float(value), 4) for value in values],
        "mean_seconds": round(float(statistics.mean(values)), 4),
        "median_seconds": round(float(statistics.median(values)), 4),
    }
    result["standard_deviation_seconds"] = (
        round(float(statistics.stdev(values)), 4) if len(values) > 1 else 0.0
    )
    return result


def main():
    args = parse_args()
    if args.repeats < 1 or args.threads < 1:
        raise ValueError("repeats and threads must be positive")
    cv2.setNumThreads(args.threads)
    mask = np.load(args.mask, allow_pickle=False).astype(bool, copy=False)
    history = np.load(args.history, allow_pickle=False).astype(np.uint8, copy=False)
    if int(mask.sum()) != 1384:
        raise ValueError(f"Expected 1384 valid positions, got {int(mask.sum())}")

    classifier, classifier_load = load_network(args.classification_onnx)
    forecaster, forecaster_load = load_network(args.forecast_onnx)
    run_network(classifier, np.zeros((1, 3, 56, 56), dtype=np.float32))
    run_network(forecaster, np.zeros((1, 5, 90, 90), dtype=np.float32))

    preprocessing, classification_inference = [], []
    construction, forecast_inference = [], []
    for _ in range(args.repeats):
        prep, infer = classify(
            classifier,
            args.image,
            mask,
            args.valid_only,
            args.classification_batch_size,
        )
        preprocessing.append(prep)
        classification_inference.append(infer)
        build, infer = forecast(
            forecaster, history, mask, args.forecast_batch_size
        )
        construction.append(build)
        forecast_inference.append(infer)

    monitoring = [a + b for a, b in zip(preprocessing, classification_inference)]
    forecasting = [a + b for a, b in zip(construction, forecast_inference)]
    combined = [a + b for a, b in zip(monitoring, forecasting)]
    result = {
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "threads": args.threads,
        },
        "runtime": f"OpenCV DNN {cv2.__version__}",
        "classification_positions": int(mask.sum()) if args.valid_only else 3774,
        "forecast_positions": int(mask.sum()),
        "model_load_seconds": {
            "classification": round(classifier_load, 4),
            "forecast": round(forecaster_load, 4),
        },
        "image_preprocessing": summary(preprocessing),
        "classification_inference": summary(classification_inference),
        "complete_monitoring": summary(monitoring),
        "forecast_tensor_construction": summary(construction),
        "forecast_inference": summary(forecast_inference),
        "complete_forecasting": summary(forecasting),
        "monitoring_and_forecasting": summary(combined),
        "peak_rss_mib": (
            round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
            if resource is not None else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
