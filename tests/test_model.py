from __future__ import annotations

import torch

from cloud_forecast_resnet.model import MultiHorizonResNet50


def test_model_output_shape() -> None:
    model = MultiHorizonResNet50(dropout=0.0).eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 5, 54, 54))
    assert output.shape == (2, 4, 5)
