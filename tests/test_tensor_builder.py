from __future__ import annotations

import torch

from cloud_forecast_resnet.tensor_builder import DenseLocalTensorBuilder


def test_shape_order_target_and_obstructed_padding() -> None:
    builder = DenseLocalTensorBuilder(51, 51, device="cpu")
    inputs = torch.stack(
        [torch.full((51, 51), time % 5, dtype=torch.uint8) for time in range(9)]
    )
    targets = torch.stack(
        [torch.full((51, 51), horizon, dtype=torch.uint8) for horizon in range(4)]
    )
    tensor, labels = builder.build_chunk(inputs, targets, torch.tensor([0, 1300]))
    assert tensor.shape == (2, 5, 54, 54)
    assert labels.shape == (2, 4)
    assert labels[0].tolist() == [0, 1, 2, 3]
    # First target cell is row 0, col 0.  Out-of-domain cells are one-hot class 1
    # (Obstructed), never an all-channel-zero vector.
    assert torch.all(tensor[0, 1, :9, :] == 1)
    assert torch.all(tensor[0, 1, :, :9] == 1)
    assert torch.count_nonzero(tensor[0, [0, 2, 3, 4], :9, :]) == 0
    assert torch.count_nonzero(tensor[0, [0, 2, 3, 4], :, :9]) == 0
    assert torch.all(tensor[0].sum(dim=0) == 1)
    # Temporal panels are row-major: t0 top-left, t1 top-middle, ..., t8 bottom-right.
    for time in range(9):
        panel_row, panel_col = divmod(time, 3)
        pixel = tensor[1, :, panel_row * 18 + 9, panel_col * 18 + 9]
        assert int(pixel.argmax()) == time % 5
        assert float(pixel.sum()) == 1.0


def test_dense_position_count() -> None:
    builder = DenseLocalTensorBuilder(51, 51, device="cpu")
    assert builder.num_positions == 51 * 51 == 2601
