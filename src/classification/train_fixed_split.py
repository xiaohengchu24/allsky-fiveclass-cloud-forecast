#!/usr/bin/env python3
"""Run the ResNet50 trainer with the packaged fixed split manifest."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path


def load_trainer(path: Path):
    spec = importlib.util.spec_from_file_location("classifier_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    args = parser.parse_args()

    module = load_trainer(Path(args.trainer))
    manifest_path = Path(args.split_manifest)

    def gather_samples(_data_root, class_names):
        samples = []
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                label = int(row["label"])
                if class_names[label] != row["class_name"]:
                    raise RuntimeError(f"Class mismatch in manifest: {row}")
                group = f"{row['split']}::{row['canonical_group']}"
                samples.append((Path(row["path"]), label, group))
        return samples

    def make_group_split(samples, _val_ratio, _test_ratio, _seed):
        buckets = {"train": [], "val": [], "test": []}
        for sample in samples:
            split = sample[2].split("::", 1)[0]
            buckets[split].append(sample)
        return buckets["train"], buckets["val"], buckets["test"]

    module.gather_samples = gather_samples
    module.make_group_split = make_group_split
    os.sys.argv = [str(Path(args.trainer)), "--config", args.config]
    module.main()


if __name__ == "__main__":
    main()
