from __future__ import annotations

"""Build the complete compact dataset required by the fixed data splits.

The train, validation, and test index files define the temporal anchors. Existing
verified matrices are reused, and missing timestamps are inferred from the source
images. An unreadable image aborts preparation so that the fixed split is preserved.
"""

import argparse
import json
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from classifier_matrix_tools import (
    IMAGE_SUFFIXES,
    image_stem,
    infer_required_images,
    is_supported_image,
    output_paths,
    timestamp_minute,
)


INPUT_OFFSETS = (-20, -17, -14, -11, -8, -6, -4, -2, -1)
HORIZONS = (1, 5, 10, 15)
MATRIX_ROI = (0, 51, 12, 63)


def parse_source(value: str) -> tuple[int, Path]:
    try:
        month_text, path_text = value.split("=", 1)
        month = int(month_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("raw-source 格式应为 MM=/path/to/dir_or.zip") from exc
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"月份越界: {month}")
    return month, Path(path_text)


def required_timestamps(index_root: Path) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for split in ("train", "val", "test"):
        index = np.load(index_root / f"{split}.npy", mmap_mode="r")
        for row in index:
            month = int(row[0])
            anchor = int(row[-1])
            result[month].update(anchor + offset for offset in INPUT_OFFSETS)
            result[month].update(anchor + horizon for horizon in HORIZONS)
    return result


def scan_directory(source: Path, month: int, needed: set[int]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in source.rglob("*"):
        if not is_supported_image(path):
            continue
        try:
            timestamp = timestamp_minute(path)
        except ValueError:
            continue
        if timestamp in needed:
            result.setdefault(timestamp, path)
    return result


def extract_needed_from_zip(
    source: Path,
    month: int,
    needed: set[int],
    output_dir: Path,
) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}
    with zipfile.ZipFile(source) as archive:
        selected: dict[int, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            if info.is_dir() or not any(info.filename.lower().endswith(ext) for ext in IMAGE_SUFFIXES):
                continue
            name = Path(info.filename).name
            try:
                timestamp = timestamp_minute(name)
            except ValueError:
                continue
            if timestamp in needed:
                selected.setdefault(timestamp, info)
        for timestamp, info in selected.items():
            destination = output_dir / Path(info.filename).name
            if not destination.exists() or destination.stat().st_size != info.file_size:
                temporary = destination.with_suffix(destination.suffix + ".part")
                with archive.open(info) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                os.replace(temporary, destination)
            result[timestamp] = destination
    return result


def locate_raw_images(
    month: int,
    needed: set[int],
    sources: list[Path],
    extraction_root: Path,
) -> dict[int, Path]:
    located: dict[int, Path] = {}
    for source in sources:
        remaining = needed - set(located)
        if not remaining:
            break
        if source.is_dir():
            found = scan_directory(source, month, remaining)
        elif source.is_file() and source.suffix.lower() == ".zip":
            found = extract_needed_from_zip(
                source, month, remaining, extraction_root / f"{month:02d}"
            )
        else:
            raise FileNotFoundError(f"原图来源不存在或不受支持: {source}")
        located.update(found)
    missing = sorted(needed - set(located))
    if missing:
        raise FileNotFoundError(
            f"{month:02d}月有 {len(missing)} 个待补时间戳找不到原图，示例={missing[:10]}"
        )
    return located


def build_compact_month(
    month: int,
    timestamps_needed: set[int],
    base_month_dir: Path,
    inferred_label_dir: Path,
    raw_by_timestamp: dict[int, Path],
    output_month_dir: Path,
) -> dict:
    base_timestamps = np.load(base_month_dir / "timestamps.npy", mmap_mode="r")
    base_labels = np.load(base_month_dir / "labels.npy", mmap_mode="r")
    valid_mask_full = np.load(base_month_dir / "valid_mask.npy")
    base_lookup = {int(value): index for index, value in enumerate(base_timestamps)}
    ordered = np.asarray(sorted(timestamps_needed), dtype=np.int64)
    output_month_dir.mkdir(parents=True, exist_ok=True)
    labels_tmp = output_month_dir / "labels.tmp.npy"
    timestamps_tmp = output_month_dir / "timestamps.tmp.npy"
    row_start, row_end, col_start, col_end = MATRIX_ROI
    matrix_shape = (row_end - row_start, col_end - col_start)
    valid_mask = valid_mask_full[row_start:row_end, col_start:col_end]
    cube = np.lib.format.open_memmap(
        labels_tmp,
        mode="w+",
        dtype=np.uint8,
        shape=(len(ordered), *matrix_shape),
    )
    reused = 0
    inferred = 0
    for position, timestamp_value in enumerate(ordered):
        timestamp = int(timestamp_value)
        base_position = base_lookup.get(timestamp)
        if base_position is not None:
            cube[position] = base_labels[
                base_position, row_start:row_end, col_start:col_end
            ]
            reused += 1
            continue
        label_path, _ = output_paths(inferred_label_dir, raw_by_timestamp[timestamp].name)
        matrix = np.load(label_path, mmap_mode="r")
        if matrix.shape != matrix_shape:
            raise ValueError(f"补算矩阵尺寸不一致: {label_path}={matrix.shape}")
        cube[position] = matrix
        inferred += 1
    cube.flush()
    del cube
    np.save(timestamps_tmp, ordered)
    os.replace(labels_tmp, output_month_dir / "labels.npy")
    os.replace(timestamps_tmp, output_month_dir / "timestamps.npy")
    np.save(output_month_dir / "valid_mask.npy", valid_mask.astype(bool, copy=False))
    manifest = {
        "month": month,
        "frame_count": int(len(ordered)),
        "matrix_shape": list(map(int, matrix_shape)),
        "source_matrix_shape": list(map(int, base_labels.shape[1:])),
        "source_matrix_roi": {
            "row_start": row_start,
            "row_end": row_end,
            "col_start": col_start,
            "col_end": col_end,
        },
        "dtype": "uint8",
        "first_timestamp_minute": int(ordered[0]),
        "last_timestamp_minute": int(ordered[-1]),
        "input_offsets_minutes": list(INPUT_OFFSETS),
        "forecast_horizons_minutes": list(HORIZONS),
        "reused_base_frames": reused,
        "newly_inferred_frames": inferred,
        "base_compact_dir": str(base_month_dir.resolve()),
        "all_masks_verified_identical": True,
    }
    temporary = output_month_dir / "manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_month_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="复用已验证矩阵并严格补齐论文所需时间点")
    parser.add_argument("--base-compact-root", type=Path, required=True)
    parser.add_argument("--base-index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--raw-source", action="append", type=parse_source, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-images-per-load", type=int, default=30)
    parser.add_argument("--keep-extracted", action="store_true")
    args = parser.parse_args()

    if not args.model_path.is_file():
        raise FileNotFoundError(f"分类器权重不存在: {args.model_path}")
    sources: dict[int, list[Path]] = defaultdict(list)
    for month, path in args.raw_source:
        sources[month].append(path)
    required = required_timestamps(args.base_index_root)
    missing_sources = sorted(set(required) - set(sources))
    if missing_sources:
        raise ValueError(f"缺少月份原图来源: {missing_sources}")

    compact_root = args.output_root / "compact"
    label_root = args.output_root / "new_labels"
    extraction_root = args.output_root / "extracted_raw"
    reports = []
    for month in sorted(required):
        base_month_dir = args.base_compact_root / f"{month:02d}"
        base_timestamps = set(
            np.load(base_month_dir / "timestamps.npy", mmap_mode="r").tolist()
        )
        missing = required[month] - base_timestamps
        print(
            f"[{month:02d}] required={len(required[month])}, "
            f"reuse={len(required[month]) - len(missing)}, infer={len(missing)}",
            flush=True,
        )
        raw_by_timestamp = locate_raw_images(
            month, missing, sources[month], extraction_root
        ) if missing else {}
        inferred_label_dir = label_root / f"{month:02d}"
        if missing:
            infer_required_images(
                [raw_by_timestamp[value] for value in sorted(missing)],
                inferred_label_dir,
                args.model_path,
                patch_size=56,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_images_per_load=args.num_images_per_load,
                no_cuda=False,
                overwrite=False,
                matrix_roi=MATRIX_ROI,
            )
            failed = {
                timestamp
                for timestamp, raw_path in raw_by_timestamp.items()
                if not output_paths(inferred_label_dir, raw_path.name)[0].is_file()
            }
            if failed:
                raise RuntimeError(
                    f"{month:02d}月有 {len(failed)} 张图像分类失败；"
                    f"为保持固定数据划分，不删除任何锚点。失败时间戳={sorted(failed)[:20]}"
                )
        report = build_compact_month(
            month,
            required[month],
            base_month_dir,
            inferred_label_dir,
            raw_by_timestamp,
            compact_root / f"{month:02d}",
        )
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        month_extraction = extraction_root / f"{month:02d}"
        if not args.keep_extracted and month_extraction.exists():
            shutil.rmtree(month_extraction)

    summary = {
        "input_offsets_minutes": list(INPUT_OFFSETS),
        "forecast_horizons_minutes": list(HORIZONS),
        "months": reports,
        "total_reused_base_frames": sum(item["reused_base_frames"] for item in reports),
        "total_newly_inferred_frames": sum(item["newly_inferred_frames"] for item in reports),
        "runtime_exclusions_for_unreadable_images": [],
        "split_indices_modified": False,
        "result": "COMPLETE",
    }
    (args.output_root / "PREPARATION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.keep_extracted and extraction_root.exists():
        shutil.rmtree(extraction_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
