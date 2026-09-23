from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Anchor:
    month: int
    input_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    day: int
    timestamp_minute: int


class CompactMonth:
    def __init__(self, root: str | Path, month: int, preload: bool = True) -> None:
        self.month = int(month)
        path = Path(root) / f"{self.month:02d}"
        mmap = None if preload else "r"
        self.labels = np.load(path / "labels.npy", mmap_mode=mmap)
        self.timestamps = np.load(path / "timestamps.npy", mmap_mode=mmap)
        if self.labels.dtype != np.uint8:
            raise TypeError(f"{path}: labels must be uint8, got {self.labels.dtype}")
        if self.labels.ndim != 3:
            raise ValueError(f"{path}: expected labels (T,H,W), got {self.labels.shape}")
        if len(self.labels) != len(self.timestamps):
            raise ValueError(f"{path}: labels/timestamps length mismatch")
        if len(np.unique(self.timestamps)) != len(self.timestamps):
            raise ValueError(f"{path}: duplicate timestamps")
        self.timestamp_to_index = {
            int(timestamp): index for index, timestamp in enumerate(self.timestamps)
        }

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.labels.shape)


class ExperimentIndex:
    COLUMNS = (
        "month",
        "input_end",
        "target_t1",
        "target_t5",
        "target_t10",
        "target_t15",
        "day",
        "timestamp_minute",
    )

    def __init__(
        self,
        path: str | Path,
        stores: dict[int, CompactMonth],
        input_offsets: tuple[int, ...] = (-20, -17, -14, -11, -8, -6, -4, -2, -1),
        horizons: tuple[int, ...] = (1, 5, 10, 15),
    ) -> None:
        rows = np.load(path)
        if rows.ndim != 2 or rows.shape[1] != len(self.COLUMNS):
            raise ValueError(f"Unexpected index shape: {rows.shape}")
        self.anchors: list[Anchor] = []
        for row in rows:
            month = int(row[0])
            timestamp = int(row[7])
            store = stores[month]
            input_indices = tuple(
                store.timestamp_to_index[timestamp + offset] for offset in input_offsets
            )
            target_indices = tuple(
                store.timestamp_to_index[timestamp + horizon] for horizon in horizons
            )
            self.anchors.append(
                Anchor(
                    month=month,
                    input_indices=input_indices,
                    target_indices=target_indices,
                    day=int(row[6]),
                    timestamp_minute=timestamp,
                )
            )

    def __len__(self) -> int:
        return len(self.anchors)

    def counts_by_month(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for anchor in self.anchors:
            result[anchor.month] = result.get(anchor.month, 0) + 1
        return result


def load_stores(
    compact_root: str | Path,
    months: tuple[int, ...] = (4, 5, 6, 10, 11, 12),
    preload: bool = True,
) -> dict[int, CompactMonth]:
    return {
        int(month): CompactMonth(compact_root, int(month), preload=preload)
        for month in months
    }
