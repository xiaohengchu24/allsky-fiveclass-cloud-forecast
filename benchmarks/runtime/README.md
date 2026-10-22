# Independent local runtime benchmark

This directory documents an independent, deployment-oriented execution-time test.
It is separate from the classification-accuracy and forecasting-accuracy evaluations.
The measurements are reported in `results/runtime/local_hardware_runtime.json`.
No cloud server was used.

## Workloads

The monitoring benchmark reads one (4144\times2822)-pixel all-sky image, divides
it into (56\times56)-pixel patches, and applies the released five-class ResNet50
checkpoint. Two Raspberry Pi variants were measured:

1. all (51\times74=3774) patches in the padded image; and
2. only the 1,384 positions selected by the released fixed valid-sky mask.

The forecasting benchmark uses 25 historical (51\times51) state matrices. It
constructs one (5\times90\times90) input for each of 1,384 valid target positions
and applies the released ResNet18 model for the 5, 10, 20, and 30 min horizons.

Model-loading time is excluded because both models remain resident during continuous
operation. Image decoding, patch extraction, normalization, forecast-input
construction, and inference are included in their corresponding stage times.

## Raspberry Pi implementation

The released PyTorch checkpoints were exported to ONNX without changing their
weights or architectures. Inference used OpenCV DNN 4.10.0 with four CPU threads on
a Raspberry Pi 4 Model B (four Arm Cortex-A72 cores, 3.7 GiB usable memory). Each
variant was repeated three times; the JSON file contains every run and the median.

The benchmark images and derived historical matrices are not redistributed. To
repeat the test, export the models and supply one original image, a `(25,51,51)`
uint8 history array, and the released valid-sky mask:

```bash
python -m pip install onnx
python benchmarks/runtime/export_models_to_onnx.py

python benchmarks/runtime/benchmark_raspberry_pi.py \
  --image /path/to/original.jpg \
  --history /path/to/historical_25x51x51.npy \
  --mask data/forecast_indices/valid_sky_mask.npy \
  --classification-onnx benchmarks/runtime/onnx/classification_resnet50.onnx \
  --forecast-onnx benchmarks/runtime/onnx/forecast_resnet18.onnx \
  --output runtime_full.json

python benchmarks/runtime/benchmark_raspberry_pi.py \
  --image /path/to/original.jpg \
  --history /path/to/historical_25x51x51.npy \
  --mask data/forecast_indices/valid_sky_mask.npy \
  --classification-onnx benchmarks/runtime/onnx/classification_resnet50.onnx \
  --forecast-onnx benchmarks/runtime/onnx/forecast_resnet18.onnx \
  --valid-only \
  --output runtime_valid_only.json
```

Generated ONNX files are intermediate deployment artifacts and are intentionally not
tracked because the authoritative PyTorch checkpoints are already included.
