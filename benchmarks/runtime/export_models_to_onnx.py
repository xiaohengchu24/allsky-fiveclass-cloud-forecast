from pathlib import Path

import onnx
import torch
from torch import nn
from torchvision import models


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "onnx"
OUT.mkdir(parents=True, exist_ok=True)


class MultiHorizonResNet18(nn.Module):
    def __init__(self, horizons=(5, 10, 20, 30), dropout=0.3):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(float(dropout))
        self.heads = nn.ModuleList(nn.Linear(features, 5) for _ in horizons)

    def forward(self, inputs):
        features = self.dropout(self.backbone(inputs))
        return torch.stack([head(features) for head in self.heads], dim=1)


def load_classification_model():
    checkpoint = torch.load(
        ROOT / "checkpoints/classification_resnet50.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg = checkpoint.get("cfg", {}) or {}
    model = models.resnet50(weights=None)
    features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(float(cfg.get("dropout", 0.5))),
        nn.Linear(features, int(cfg.get("num_classes", 5))),
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def load_forecast_model():
    checkpoint = torch.load(
        ROOT / "checkpoints/forecast_resnet18.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg = checkpoint["config"]
    model = MultiHorizonResNet18(
        tuple(cfg["horizons_minutes"]), float(cfg["dropout"])
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def export(model, sample, destination, output_name):
    torch.onnx.export(
        model,
        sample,
        destination,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=[output_name],
        dynamic_axes={"input": {0: "batch"}, output_name: {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(destination))
    print(destination)


if __name__ == "__main__":
    with torch.inference_mode():
        export(
            load_classification_model(),
            torch.zeros(1, 3, 56, 56),
            OUT / "classification_resnet50.onnx",
            "logits",
        )
        export(
            load_forecast_model(),
            torch.zeros(1, 5, 90, 90),
            OUT / "forecast_resnet18.onnx",
            "logits",
        )
