# Regional all-sky cloud monitoring and prediction

This repository contains the implementation, trained weights, and evaluation summaries
for five-class regional cloud monitoring and multi-horizon regional cloud prediction from
all-sky camera images.

## Method overview

The monitoring model divides a `4144 x 2822` all-sky image into `56 x 56` image patches
and classifies each patch into one of five observing states:

1. Cloudy
2. Obstructed
3. Partial visibility
4. Bright-area interference
5. Visible

The complete classification grid has shape `51 x 74`. Prediction uses the central
`51 x 51` region of interest (columns `[12, 63)`). For every target position in this
region, the input is constructed from nine nonuniformly spaced historical matrices at
`t-20`, `t-17`, `t-14`, `t-11`, `t-8`, `t-6`, `t-4`, `t-2`, and `t-1` min.

At each historical time, an `18 x 18` neighbourhood is extracted. Positions outside the
valid region are assigned to the Obstructed class. The neighbourhoods are five-channel
one-hot encoded and arranged in a row-major `3 x 3` temporal layout, producing a
`5 x 54 x 54` input tensor. All `51 x 51 = 2601` target positions are used without
spatial subsampling.

The prediction network consists of a shared ResNet50 backbone and four independent
five-class linear heads. The four heads predict the regional observing states at
`t+1`, `t+5`, `t+10`, and `t+15` min. Per-target logits have shape `4 x 5`; the outputs
for all target positions are reassembled into a `4 x 5 x 51 x 51` tensor.

The complete machine-readable specification is provided in
[`method_contract.json`](method_contract.json).

## Repository contents

- `preparation/classifier_matrix_tools.py`: regional monitoring and compact-matrix preparation.
- `cloud_forecast_resnet/`: tensor construction, model, storage, training support, and metrics.
- `train.py`: prediction-model training.
- `evaluate.py`: independent testing and calendar-day block-bootstrap confidence intervals.
- `verify_before_training.py`: strict checks of data splits, tensor shapes, padding, and model output.
- `config.example.json`: configuration used for the reported prediction model.
- `weights/`: trained monitoring and prediction checkpoints.
- `results/`: overall and monthly evaluation summaries.
- `tests/`: unit tests for tensor construction and model output shapes.

## Environment

Python 3.10 or later is recommended. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the unit tests with:

```bash
python -m pytest -q
```

## Monitoring and matrix preparation

The monitoring checkpoint is `weights/monitoring_resnet50_best.pt`. The following
example classifies the required images for one month and generates compact matrices:

```bash
python preparation/classifier_matrix_tools.py \
  --image-dir /path/to/images/2024/04 \
  --month 4 \
  --model-path weights/monitoring_resnet50_best.pt \
  --work-root /path/to/prepared_data \
  --segment test:60:1:30
```

Image filenames must contain timestamps in the format expected by
`preparation/classifier_matrix_tools.py`. Repeat the command for the required months
and data subsets.

## Prediction training and evaluation

Edit the three paths in `config.example.json`:

- `compact_root`: directory containing monthly compact classification matrices.
- `index_root`: directory containing `train.npy`, `val.npy`, and `test.npy`.
- `output_dir`: directory for checkpoints and training history.

Verify the prepared inputs before training:

```bash
python verify_before_training.py \
  --compact-root /path/to/compact \
  --index-root /path/to/indices \
  --classifier-checkpoint weights/monitoring_resnet50_best.pt
```

Train and evaluate:

```bash
python train.py --config config.example.json

python evaluate.py \
  --config config.example.json \
  --checkpoint weights/forecast_resnet50_best.pt \
  --output results/reproduced_test
```

The expected numbers of temporal anchors are 9753 for training, 2733 for validation,
and 2088 for independent testing. The test set contains 601 April anchors, 744 October
anchors, and 743 December anchors. Each anchor contains 2601 spatial target positions.

## Reported prediction results

| Forecast horizon | Accuracy | Macro-F1 |
|---:|---:|---:|
| 1 min | 92.34% | 84.45% |
| 5 min | 90.81% | 81.34% |
| 10 min | 89.59% | 79.25% |
| 15 min | 88.74% | 77.69% |

Detailed monthly accuracies, confidence intervals, and Macro-F1 scores are available in
the `results` directory.

## Trained weights

The checkpoint files are stored with Git LFS.

| File | Purpose | SHA-256 |
|---|---|---|
| `weights/monitoring_resnet50_best.pt` | Five-class regional monitoring | `106b7bc799587a0d82d2ebf9c5fb5075929430353bd37abf29bf588baa38edfd` |
| `weights/forecast_resnet50_best.pt` | Four-horizon regional prediction | `8091f36b0a59ec15f7ea53616f5c6850140ce4e1968dfc42418faa4f459ecadb` |

After cloning, run `git lfs pull` if the checkpoint files are represented by LFS pointer
files in the working tree.
