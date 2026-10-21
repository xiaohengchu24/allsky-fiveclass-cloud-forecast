# Regional cloud monitoring and forecasting from all-sky images

This repository contains the code, fixed data splits, trained models, and numerical results associated with the manuscript **“Regional Cloud Monitoring and Cloud Distribution Forecasting from All-Sky Camera Images.”**

## Contents

- `src/classification/`: five-class regional monitoring model (ResNet50).
- `src/preprocessing/`: conversion of all-sky images into regional state matrices.
- `src/forecast/`: multi-horizon forecast model with a shared ResNet18 backbone and four classification heads.
- `src/analysis/`: monthly analysis and generic CSV table generation.
- `config/`: final model configurations.
- `data/classification/`: 2024 fixed split manifest and dataset summary, plus the
  reported dataset summary for the independent 2020 camera experiment.
- `data/forecast_indices/`: fixed training, validation, and test anchors and the valid-sky mask.
- `results/`: classification, forecast, persistence-baseline, and plotting results.
- `checkpoints/`: the independently trained 2020- and post-2024-camera
  classification checkpoints and the final forecasting checkpoint used for the
  reported results.

Raw all-sky images and the derived monthly matrices are not redistributed in this archive. The fixed manifests and indices are provided so that the reported partitions can be reconstructed when the image data are available.

## Reported results

The regional classification accuracies are 96.72% for the 2020 camera dataset and 92.07% for the 2024 camera dataset. On the independent 2024 monthly forecast tests, the model accuracies at 5, 10, 20, and 30 min are 83.13%, 81.76%, 80.46%, and 78.77%, respectively. The corresponding persistence accuracies are 82.89%, 80.38%, 77.71%, and 75.49%.

The forecast training months are May, June, August, and November. The independent test months are April, July, October, and December. Forecast evaluation uses 2,818 anchors and 1,384 valid sky positions. Confidence intervals in the released monthly tables use 1,000 calendar-day block-bootstrap resamples.

The internal classification label `unknown bright area` corresponds to “Bright-area interference” in the manuscript.

### Camera-specific classification artifacts

The released `checkpoints/classification_resnet50.pt` checkpoint and
`data/classification/fixed_split_manifest.csv` correspond to the post-2024
220-degree camera dataset used to prepare the forecasting inputs. The independent
2020 180-degree camera experiment used a separate whole-image-group split. Its
trained checkpoint is `checkpoints/classification_resnet50_2020.pt`. Its training
configuration, dataset summary, training history, classification report, and test
results are provided in `config/classification_2020.yaml`,
`data/classification/camera_2020_dataset_summary.json` and
the `results/classification/camera_2020_*` files.

The 2020 checkpoint has SHA-256
`5480875ad4b4d75a52505ff12df40da72e69d64508e4d6820cb6b41074d072bb`.
The original 2020 split manifest and raw patches are not redistributed; the released
summary records the group-wise 70%/15%/15% split and all reported class counts.

## Environment

Python 3.10 or later is recommended. Install the required packages with:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The released models were trained with PyTorch 2.2.1 and torchvision 0.17.1 using CUDA 12.1.

## Classification

For the 2020 camera experiment, place the five patch-class directories under
`data/classification_2020/original_patches/`, then run:

```bash
python src/classification/train.py --config config/classification_2020.yaml
```

For the post-2024 camera experiment, place the regional image patches at the
relative paths listed in `data/classification/fixed_split_manifest.csv`, then run:

```bash
python src/classification/train_fixed_split.py \
  --trainer src/classification/train.py \
  --config config/classification.yaml \
  --split-manifest data/classification/fixed_split_manifest.csv
```

## Forecasting

For each month, place `labels.npy` and `timestamps.npy` in `data/forecast_compact/MM/`. The labels must be an unsigned 8-bit array with shape `(time, 51, 51)`. Run training and evaluation from the repository root:

```bash
python src/forecast/train.py --config config/forecast.json
python src/forecast/evaluate.py \
  --config config/forecast.json \
  --checkpoint checkpoints/forecast_resnet18.pt \
  --output outputs/forecast_test
```

Set `PYTHONPATH=src/forecast` if the package directory is not resolved automatically by the Python installation.

## Integrity check

Run the release check before using the archive:

```bash
python verify_release.py
```

`SHA256SUMS.txt` contains hashes for every released file other than the checksum list itself.

## GitHub distribution

The checkpoints are tracked with Git LFS because the forecast checkpoint is larger
than GitHub's regular 100 MB file limit. After cloning, run `git lfs pull` to download
the checkpoint contents. The complete versioned ZIP may additionally be attached to
a GitHub Release.

## Citation

Please cite the accompanying manuscript. Machine-readable citation metadata are provided in `CITATION.cff`.

## Rights

Copyright © 2026 Hengchu Xiao et al. All rights reserved.
