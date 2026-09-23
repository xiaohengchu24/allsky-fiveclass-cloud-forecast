from __future__ import annotations

"""
单文件抽样标签生成器。

功能：
1. 从一个月原图中按时间段抽取可组成 9 -> (+1, +5, +10, +15) 序列的锚点；
2. 只对这些序列实际需要的原图运行五分类器；
3. 支持断点续跑，并核验所有输出；
4. 把大量单帧小文件压成训练电脑可直接读取的 uint8 月度文件。

除本文件外，只需要小块五分类模型权重和 Python 依赖。
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


IMAGE_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".fits", ".fits.bz2", ".fits.fz",
)
TIMESTAMP_PATTERN = re.compile(r"(?P<date>\d{8})[_-](?P<time>\d{6})")
REQUIRED_OFFSETS = tuple(range(-8, 1)) + (1, 5, 10, 15)


def parse_segment(value: str) -> tuple[str, int, int, int]:
    try:
        name, interval, day_start, day_end = value.split(":")
        interval, day_start, day_end = int(interval), int(day_start), int(day_end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "segment 格式应为 名称:间隔分钟:起始日:结束日，例如 train:30:1:23"
        ) from exc
    if not name or interval < 1 or not (1 <= day_start <= day_end <= 31):
        raise argparse.ArgumentTypeError(f"无效 segment: {value}")
    return name, interval, day_start, day_end


def timestamp_minute(path_or_name: Path | str) -> int:
    name = Path(path_or_name).name
    match = TIMESTAMP_PATTERN.search(name)
    if not match:
        raise ValueError(f"文件名中找不到 YYYYMMDD_HHMMSS 时间戳: {name}")
    value = datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
    if value.second != 0:
        raise ValueError(f"不是整分钟图像: {name}")
    return int(value.replace(tzinfo=timezone.utc).timestamp() // 60)


def image_stem(name: str) -> str:
    lower = name.lower()
    for suffix in IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def is_supported_image(path: Path) -> bool:
    return path.is_file() and any(path.name.lower().endswith(ext) for ext in IMAGE_SUFFIXES)


def collect_images(image_dir: Path, month: int) -> dict[int, Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(f"原图目录不存在: {image_dir}")
    by_timestamp: dict[int, Path] = {}
    ignored = 0
    for path in image_dir.iterdir():
        if not is_supported_image(path):
            continue
        try:
            timestamp = timestamp_minute(path)
        except ValueError:
            ignored += 1
            continue
        actual_month = datetime.fromtimestamp(timestamp * 60, tz=timezone.utc).month
        if actual_month != month:
            continue
        if timestamp in by_timestamp:
            raise ValueError(
                f"同一分钟出现多张图，无法确定使用哪张: {by_timestamp[timestamp].name}, {path.name}"
            )
        by_timestamp[timestamp] = path
    if ignored:
        print(f"提示：忽略 {ignored} 个文件名中没有规范时间戳的文件。")
    if not by_timestamp:
        raise FileNotFoundError(f"{image_dir} 中没有找到 {month} 月的规范时间戳图像")
    return by_timestamp


def choose_anchors(
    timestamps: set[int], interval_minutes: int, day_start: int, day_end: int, seed: int
) -> np.ndarray:
    candidates: list[int] = []
    for timestamp in sorted(timestamps):
        day = datetime.fromtimestamp(timestamp * 60, tz=timezone.utc).day
        if day_start <= day <= day_end and all(timestamp + offset in timestamps for offset in REQUIRED_OFFSETS):
            candidates.append(timestamp)
    groups: dict[int, list[int]] = defaultdict(list)
    for timestamp in candidates:
        groups[timestamp // interval_minutes].append(timestamp)
    selected: list[int] = []
    for bin_id, values in sorted(groups.items()):
        rng = np.random.default_rng(seed + bin_id)
        selected.append(values[int(rng.integers(0, len(values)))])
    return np.asarray(selected, dtype=np.int64)


def save_selection(
    prefix: Path,
    image_dir: Path,
    image_count: int,
    anchors: np.ndarray,
    required_paths: list[Path],
    interval: int,
    day_start: int,
    day_end: int,
    seed: int,
) -> dict:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    file_list = prefix.with_suffix(".files.txt")
    anchor_path = prefix.with_suffix(".anchors.npy")
    manifest_path = prefix.with_suffix(".json")
    file_list.write_text("\n".join(path.name for path in required_paths) + "\n", encoding="utf-8")
    np.save(anchor_path, anchors)
    manifest = {
        "image_dir": str(image_dir.resolve()),
        "interval_minutes": interval,
        "day_range": [day_start, day_end],
        "seed": seed,
        "anchor_count": int(len(anchors)),
        "required_image_count": len(required_paths),
        "fraction_of_month_images": len(required_paths) / max(image_count, 1),
        "file_list": str(file_list.resolve()),
        "anchors": str(anchor_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def is_fits(path: Path) -> bool:
    return any(path.name.lower().endswith(ext) for ext in (".fits", ".fits.bz2", ".fits.fz"))


def normalize_fits_to_rgb(data: np.ndarray, clip_percentiles=(2, 98)) -> np.ndarray:
    finite = data[np.isfinite(data)]
    fill = float(np.mean(finite)) if finite.size else 0.0
    data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)
    vmin, vmax = np.percentile(data, clip_percentiles)
    data = np.clip(data, vmin, vmax)
    data = np.log1p(data - vmin + 1)
    data_min, data_max = float(data.min()), float(data.max())
    if data_max == data_min:
        normalized = np.zeros_like(data, dtype=np.uint8)
    else:
        normalized = ((data - data_min) * (255.0 / (data_max - data_min))).astype(np.uint8)
    return np.stack([normalized, normalized, normalized], axis=-1)


def load_image_as_rgb(path: Path) -> np.ndarray:
    if not is_fits(path):
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("读取 FITS 文件需要安装 astropy：pip install astropy") from exc
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None or not isinstance(data, np.ndarray) or data.size == 0:
            raise ValueError("FITS 数据为空")
        if data.ndim != 2:
            raise ValueError(f"FITS 数据不是二维，shape={data.shape}")
        return normalize_fits_to_rgb(np.flipud(data.astype(np.float32)))


def pad_image(data: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = data.shape[:2]
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    return np.pad(data, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")


def load_and_prepare_patches(
    path: Path,
    patch_size: int,
    matrix_roi: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Decode, split, and normalize one image using the original float32 formula."""
    rgb = load_image_as_rgb(path)
    padded = pad_image(rgb, patch_size)
    n_rows = padded.shape[0] // patch_size
    n_cols = padded.shape[1] // patch_size
    patch_grid = padded.reshape(
        n_rows, patch_size, n_cols, patch_size, 3
    ).transpose(0, 2, 1, 3, 4)
    if matrix_roi is not None:
        row_start, row_end, col_start, col_end = map(int, matrix_roi)
        if not (0 <= row_start < row_end <= n_rows):
            raise ValueError(f"ROI row bounds {matrix_roi} exceed {n_rows} rows")
        if not (0 <= col_start < col_end <= n_cols):
            raise ValueError(f"ROI column bounds {matrix_roi} exceed {n_cols} columns")
        patch_grid = patch_grid[row_start:row_end, col_start:col_end]
        n_rows = row_end - row_start
        n_cols = col_end - col_start
    patches = patch_grid.reshape(-1, patch_size, patch_size, 3).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    patches = (patches - mean) / std
    return patches, n_rows, n_cols


class BigImagePatchDataset(Dataset):
    def __init__(self, rgb: np.ndarray, patch_size: int = 56):
        self.patch_size = patch_size
        self.padded = pad_image(rgb, patch_size)
        self.n_rows = self.padded.shape[0] // patch_size
        self.n_cols = self.padded.shape[1] // patch_size
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return self.n_rows * self.n_cols

    def __getitem__(self, index: int):
        row, col = divmod(index, self.n_cols)
        size = self.patch_size
        patch = self.padded[row * size:(row + 1) * size, col * size:(col + 1) * size]
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - self.mean) / self.std
        return torch.from_numpy(patch).permute(2, 0, 1), index


def create_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    if not model_name.startswith("resnet") or not hasattr(models, model_name):
        raise ValueError(f"当前单文件版本不支持分类器结构: {model_name}")
    model = getattr(models, model_name)(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def load_classifier(model_path: Path, device: torch.device) -> tuple[nn.Module, int]:
    if not model_path.is_file():
        raise FileNotFoundError(f"分类器权重不存在: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:  # 兼容旧版 PyTorch
        checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("权重文件应为包含 model 和 cfg 的 checkpoint 字典")
    cfg = checkpoint.get("cfg", {}) or {}
    num_classes = int(cfg.get("num_classes", 5))
    model = create_model(
        str(cfg.get("model_name", "resnet50")),
        num_classes,
        float(cfg.get("dropout", 0.5)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, memory_format=torch.channels_last).eval()
    if device.type == "cuda":
        model.half()
    return model, num_classes


def output_paths(label_dir: Path, image_name: str) -> tuple[Path, Path]:
    stem = image_stem(image_name)
    return label_dir / f"{stem}_labels.npy", label_dir / f"{stem}_mask.npy"


def outputs_complete(label_dir: Path, image_name: str, num_classes: int, overwrite: bool) -> bool:
    if overwrite:
        return False
    label_path, mask_path = output_paths(label_dir, image_name)
    if not label_path.exists() or not mask_path.exists():
        return False
    try:
        labels = np.load(label_path, mmap_mode="r")
        mask = np.load(mask_path, mmap_mode="r")
        return (
            labels.ndim == 2 and labels.shape == mask.shape and labels.size > 0
            and int(labels.min()) >= 0 and int(labels.max()) < num_classes
        )
    except Exception:
        return False


def atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def infer_required_images(
    required_paths: list[Path],
    label_dir: Path,
    model_path: Path,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    num_images_per_load: int,
    no_cuda: bool,
    overwrite: bool,
    matrix_roi: tuple[int, int, int, int] | None = None,
) -> int:
    label_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    print(f"设备: {device}；CUDA FP16: {device.type == 'cuda'}")
    model, num_classes = load_classifier(model_path, device)
    pending = [
        path for path in required_paths
        if not outputs_complete(label_dir, path.name, num_classes, overwrite)
    ]
    print(f"需要 {len(required_paths)} 张；已完成 {len(required_paths) - len(pending)} 张；待处理 {len(pending)} 张。")
    if not pending:
        return num_classes

    preload_workers = min(4, max(1, num_images_per_load))
    for start in tqdm(range(0, len(pending), num_images_per_load), desc="处理图像批次"):
        batch_paths = pending[start:start + num_images_per_load]
        with ThreadPoolExecutor(max_workers=preload_workers) as executor:
            # CPU preparation and GPU inference form a bounded pipeline: process a
            # prepared image immediately instead of retaining a full image batch.
            futures = {
                executor.submit(
                    load_and_prepare_patches, path, patch_size, matrix_roi
                ): path
                for path in batch_paths
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    patches, n_rows, n_cols = future.result()
                except Exception as exc:
                    print(f"加载或预处理失败 {path.name}: {exc}", file=sys.stderr)
                    continue
                try:
                    predictions = np.empty(len(patches), dtype=np.uint8)
                    with torch.inference_mode():
                        for patch_start in range(0, len(patches), batch_size):
                            patch_end = min(patch_start + batch_size, len(patches))
                            tensor = torch.from_numpy(
                                patches[patch_start:patch_end]
                            ).permute(0, 3, 1, 2)
                            tensor = tensor.to(
                                device=device,
                                dtype=(
                                    torch.float16
                                    if device.type == "cuda"
                                    else torch.float32
                                ),
                                non_blocking=True,
                                memory_format=torch.channels_last,
                            )
                            result = (
                                model(tensor)
                                .argmax(dim=1)
                                .cpu()
                                .numpy()
                                .astype(np.uint8)
                            )
                            predictions[patch_start:patch_end] = result
                    labels = predictions.reshape(n_rows, n_cols)
                    mask = np.ones_like(labels, dtype=bool)
                    label_path, mask_path = output_paths(label_dir, path.name)
                    atomic_save(label_path, labels)
                    atomic_save(mask_path, mask)
                except Exception as exc:
                    print(f"处理失败 {path.name}: {exc}", file=sys.stderr)
    return num_classes


def audit_required(required_paths: list[Path], label_dir: Path, num_classes: int) -> dict:
    missing: list[str] = []
    bad: list[str] = []
    for image_path in required_paths:
        label_path, mask_path = output_paths(label_dir, image_path.name)
        if not label_path.exists() or not mask_path.exists():
            missing.append(image_path.name)
            continue
        try:
            labels = np.load(label_path, mmap_mode="r")
            mask = np.load(mask_path, mmap_mode="r")
            if (
                labels.ndim != 2 or labels.shape != mask.shape
                or int(labels.min()) < 0 or int(labels.max()) >= num_classes
            ):
                bad.append(image_path.name)
        except Exception:
            bad.append(image_path.name)
    report = {
        "required": len(required_paths),
        "missing": len(missing),
        "bad": len(bad),
        "missing_examples": missing[:10],
        "bad_examples": bad[:10],
        "complete": not missing and not bad,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise RuntimeError("标签核验未通过；修复报错或直接重跑本命令后再压缩。")
    return report


def compact_labels(
    label_dir: Path,
    output_dir: Path,
    month: int,
    num_classes: int,
    rebuild: bool,
) -> dict:
    label_files = sorted(label_dir.glob("*_labels.npy"), key=timestamp_minute)
    if not label_files:
        raise FileNotFoundError(f"{label_dir} 中没有标签矩阵")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not rebuild:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(old.get("frame_count", -1)) == len(label_files):
            print(f"紧凑文件已存在且帧数一致，跳过重建: {output_dir}")
            return old
        raise FileExistsError(
            f"{manifest_path} 是旧结果且标签数量已变化；确认后添加 --rebuild-compact 重建。"
        )

    timestamps = np.asarray([timestamp_minute(path) for path in label_files], dtype=np.int64)
    if len(np.unique(timestamps)) != len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("标签时间戳重复或没有严格递增")
    first = np.load(label_files[0], mmap_mode="r")
    if first.ndim != 2:
        raise ValueError(f"标签矩阵必须是二维，实际 {first.shape}")
    height, width = map(int, first.shape)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_tmp = output_dir / "labels.tmp.npy"
    timestamps_tmp = output_dir / "timestamps.tmp.npy"
    mask_tmp = output_dir / "valid_mask.tmp.npy"
    for path in (labels_tmp, timestamps_tmp, mask_tmp):
        if path.exists():
            path.unlink()

    cube = np.lib.format.open_memmap(
        labels_tmp, mode="w+", dtype=np.uint8, shape=(len(label_files), height, width)
    )
    first_mask_path = Path(str(label_files[0]).replace("_labels.npy", "_mask.npy"))
    common_mask = np.load(first_mask_path).astype(bool)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for index, path in enumerate(tqdm(label_files, desc="压缩标签")):
        matrix = np.load(path, mmap_mode="r")
        mask_path = Path(str(path).replace("_labels.npy", "_mask.npy"))
        if matrix.shape != (height, width):
            raise ValueError(f"标签尺寸变化: {path.name}={matrix.shape}，期望 {(height, width)}")
        if int(matrix.min()) < 0 or int(matrix.max()) >= num_classes:
            raise ValueError(f"标签类别越界: {path.name}")
        if not mask_path.exists() or not np.array_equal(np.load(mask_path), common_mask):
            raise ValueError(f"mask 缺失或月份内部不一致: {mask_path.name}")
        cube[index] = matrix.astype(np.uint8, copy=False)
        class_counts += np.bincount(np.asarray(matrix)[common_mask], minlength=num_classes)
    cube.flush()
    del cube
    np.save(timestamps_tmp, timestamps)
    np.save(mask_tmp, common_mask)
    os.replace(labels_tmp, output_dir / "labels.npy")
    os.replace(timestamps_tmp, output_dir / "timestamps.npy")
    os.replace(mask_tmp, output_dir / "valid_mask.npy")
    manifest = {
        "month": month,
        "source_dir": str(label_dir.resolve()),
        "frame_count": len(label_files),
        "matrix_shape": [height, width],
        "dtype": "uint8",
        "first_timestamp_minute": int(timestamps[0]),
        "last_timestamp_minute": int(timestamps[-1]),
        "class_counts": class_counts.tolist(),
        "all_masks_verified_identical": True,
    }
    manifest_tmp = output_dir / "manifest.tmp.json"
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单文件：先抽预测序列，再生成必要的五分类矩阵并压缩"
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="当前月份原图目录")
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--model-path", type=Path, required=True, help="小块五分类 checkpoint (.pt)")
    parser.add_argument("--work-root", type=Path, required=True, help="本月全部输出的根目录")
    parser.add_argument(
        "--segment", type=parse_segment, action="append", required=True,
        help="可重复，例如 train:30:1:23 或 test:60:1:31",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-images-per-load", type=int, default=30)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--overwrite-labels", action="store_true")
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument("--rebuild-compact", action="store_true")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    if args.patch_size < 1 or args.batch_size < 1 or args.num_workers < 0 or args.num_images_per_load < 1:
        raise ValueError("patch-size、batch-size 和 num-images-per-load 必须为正数；num-workers 不能为负")

    month_name = f"{args.month:02d}"
    selection_root = args.work_root / "selections"
    label_dir = args.work_root / "labels" / month_name
    compact_dir = args.work_root / "compact" / month_name
    selection_root.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print("扫描月份原图……")
    by_timestamp = collect_images(args.image_dir, args.month)
    timestamps = set(by_timestamp)
    print(f"找到 {len(by_timestamp)} 张带有效时间戳的 {args.month} 月原图。")

    all_required_timestamps: set[int] = set()
    segment_required: dict[str, list[Path]] = {}
    seen_names: set[str] = set()
    for name, interval, day_start, day_end in args.segment:
        if name in seen_names:
            raise ValueError(f"segment 名称重复: {name}")
        seen_names.add(name)
        anchors = choose_anchors(timestamps, interval, day_start, day_end, args.seed)
        if len(anchors) == 0:
            raise ValueError(
                f"segment {name} 没有可用锚点；请检查日期范围、缺图情况和文件名时间戳。"
            )
        required_timestamps = sorted({
            int(anchor) + offset for anchor in anchors for offset in REQUIRED_OFFSETS
        })
        required_paths = [by_timestamp[timestamp] for timestamp in required_timestamps]
        segment_required[name] = required_paths
        all_required_timestamps.update(required_timestamps)
        print(f"\n抽样分段 {name}：")
        save_selection(
            selection_root / f"{month_name}_{name}",
            args.image_dir,
            len(by_timestamp),
            anchors,
            required_paths,
            interval,
            day_start,
            day_end,
            args.seed,
        )

    all_required = [by_timestamp[timestamp] for timestamp in sorted(all_required_timestamps)]
    print(f"\n所有分段合并去重后，需要分类 {len(all_required)} 张原图。")
    num_classes = infer_required_images(
        all_required,
        label_dir,
        args.model_path,
        args.patch_size,
        args.batch_size,
        args.num_workers,
        args.num_images_per_load,
        args.no_cuda,
        args.overwrite_labels,
    )

    print("\n核验各抽样分段：")
    for name, paths in segment_required.items():
        print(f"[{name}]")
        audit_required(paths, label_dir, num_classes)

    if not args.no_compact:
        print("\n生成紧凑月度文件：")
        compact_labels(
            label_dir, compact_dir, args.month, num_classes, args.rebuild_compact
        )

    print("\n全部完成。")
    print(f"抽样清单: {selection_root}")
    print(f"单帧标签: {label_dir}")
    if not args.no_compact:
        print(f"复制到训练电脑的紧凑目录: {compact_dir}")


if __name__ == "__main__":
    main()
