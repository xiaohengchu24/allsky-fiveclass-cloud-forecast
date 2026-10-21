from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class DenseLocalTensorBuilder:
    """Build dense square-layout local inputs on the GPU without disk materialization."""

    def __init__(
        self,
        height: int,
        width: int,
        row_start: int = 0,
        row_end: int = 51,
        col_start: int = 0,
        col_end: int = 51,
        window_size: int = 18,
        num_classes: int = 5,
        padding_class: int = 1,
        input_valid_mask: torch.Tensor | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        if window_size != 18:
            raise ValueError("The paper method requires an 18x18 neighbourhood")
        self.height = int(height)
        self.width = int(width)
        self.num_classes = int(num_classes)
        self.padding_class = int(padding_class)
        if not 0 <= self.padding_class < self.num_classes:
            raise ValueError("padding_class must be a valid class index")
        self.device = torch.device(device)

        rows, cols = torch.meshgrid(
            torch.arange(row_start, row_end),
            torch.arange(col_start, col_end),
            indexing="ij",
        )
        self.target_rows = rows.reshape(-1).to(self.device)
        self.target_cols = cols.reshape(-1).to(self.device)

        delta = torch.arange(-window_size // 2, window_size // 2)
        rr = rows.reshape(-1, 1, 1) + delta.reshape(1, -1, 1)
        cc = cols.reshape(-1, 1, 1) + delta.reshape(1, 1, -1)
        rr = rr.expand(-1, window_size, window_size)
        cc = cc.expand(-1, window_size, window_size)
        valid = (rr >= 0) & (rr < self.height) & (cc >= 0) & (cc < self.width)
        flat_indices = rr.clamp(0, self.height - 1) * self.width + cc.clamp(0, self.width - 1)
        if input_valid_mask is not None:
            spatial_mask = torch.as_tensor(input_valid_mask, dtype=torch.bool)
            if tuple(spatial_mask.shape) != (self.height, self.width):
                raise ValueError(
                    f"Expected input_valid_mask {(self.height, self.width)}, "
                    f"got {tuple(spatial_mask.shape)}"
                )
            valid = valid & spatial_mask.reshape(-1)[flat_indices]
        self.neighbour_indices = flat_indices.reshape(len(rows.reshape(-1)), -1).to(self.device)
        self.neighbour_valid = valid.reshape(len(rows.reshape(-1)), -1).to(self.device)

    @property
    def num_positions(self) -> int:
        return int(self.target_rows.numel())

    def build_chunk(
        self,
        input_frames: torch.Tensor,
        target_frames: torch.Tensor,
        position_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input_frames: uint8 (T,H,W), where T must be a square number
        target_frames: uint8 (4,H,W)
        position_indices: long (N,)
        returns inputs float (N,5,L*18,L*18), targets long (N,4)
        """
        position_indices = position_indices.to(self.device, non_blocking=True)
        gather_indices = self.neighbour_indices[position_indices]
        valid = self.neighbour_valid[position_indices]
        count = int(position_indices.numel())

        frame_count = int(input_frames.shape[0])
        layout_size = math.isqrt(frame_count)
        if layout_size * layout_size != frame_count:
            raise ValueError(
                f"Input frame count must form a square layout, got {frame_count}"
            )
        flat_inputs = input_frames.reshape(frame_count, -1)
        gathered = torch.gather(
            flat_inputs,
            1,
            gather_indices.reshape(1, -1).expand(frame_count, -1),
        ).reshape(frame_count, count, 18, 18)

        gathered = torch.where(
            valid.reshape(1, count, 18, 18),
            gathered,
            torch.full_like(gathered, self.padding_class),
        )
        one_hot = F.one_hot(gathered.long(), num_classes=self.num_classes)
        inputs = (
            one_hot.reshape(
                layout_size,
                layout_size,
                count,
                18,
                18,
                self.num_classes,
            )
            .permute(2, 5, 0, 3, 1, 4)
            .reshape(
                count,
                self.num_classes,
                layout_size * 18,
                layout_size * 18,
            )
            .to(dtype=torch.float32)
        )

        rows = self.target_rows[position_indices]
        cols = self.target_cols[position_indices]
        targets = target_frames[:, rows, cols].transpose(0, 1).long()
        return inputs.contiguous(memory_format=torch.channels_last), targets
