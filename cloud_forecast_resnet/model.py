from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet50


class MultiHorizonResNet50(nn.Module):
    """Five-channel ResNet50 with one five-class head per forecast horizon."""

    def __init__(
        self,
        num_classes: int = 5,
        horizons: tuple[int, ...] = (1, 5, 10, 15),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.horizons = tuple(int(value) for value in horizons)

        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(
            5, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(float(dropout))
        self.heads = nn.ModuleList(
            nn.Linear(features, self.num_classes) for _ in self.horizons
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.backbone(inputs))
        return torch.stack([head(features) for head in self.heads], dim=1)
