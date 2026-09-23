from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cloud_forecast_resnet.engine import (
    evaluate_anchors,
    iter_position_chunks,
    load_anchor_frames,
    multi_horizon_loss,
    shuffled_anchor_indices,
)
from cloud_forecast_resnet.metrics import scores_from_confusion
from cloud_forecast_resnet.model import MultiHorizonResNet50
from cloud_forecast_resnet.storage import ExperimentIndex, load_stores
from cloud_forecast_resnet.tensor_builder import DenseLocalTensorBuilder


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stop-file",
        type=Path,
        help=(
            "Graceful-stop marker. Once best.pt exists, detecting this file between "
            "anchors stops training successfully and lets the calling pipeline evaluate "
            "the best completed epoch."
        ),
    )
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    contract_path = Path(__file__).with_name("method_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stores = load_stores(cfg["compact_root"], preload=True)
    index_root = Path(cfg["index_root"])
    train_index = ExperimentIndex(index_root / "train.npy", stores)
    val_index = ExperimentIndex(index_root / "val.npy", stores)
    expected = cfg["expected_anchor_counts"]
    if len(train_index) != int(expected["train"]) or len(val_index) != int(expected["val"]):
        raise RuntimeError(
            f"Anchor count mismatch: train={len(train_index)}, val={len(val_index)}"
        )

    sample_store = stores[train_index.anchors[0].month]
    builder = DenseLocalTensorBuilder(sample_store.shape[1], sample_store.shape[2], device=device)
    if builder.num_positions != 2601:
        raise RuntimeError(f"Expected 2601 dense positions, got {builder.num_positions}")

    model = MultiHorizonResNet50(dropout=float(cfg["dropout"])).to(
        device=device, memory_format=torch.channels_last
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["epochs"]), eta_min=float(cfg["minimum_learning_rate"])
    )
    weights = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    start_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macro_f1 = float(checkpoint["best_macro_f1"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

    position_generator = torch.Generator(device="cpu")
    log_path = output / "history.jsonl"
    for epoch in range(start_epoch, int(cfg["epochs"])):
        model.train()
        epoch_start = time.time()
        loss_sum = 0.0
        item_count = 0
        stop_during_epoch = False
        for anchor_number in shuffled_anchor_indices(len(train_index), seed, epoch):
            if (
                args.stop_file is not None
                and args.stop_file.is_file()
                and (output / "best.pt").is_file()
            ):
                print(
                    f"Graceful stop requested before anchor {anchor_number} of epoch {epoch}; "
                    "the current partial epoch is discarded and best.pt is retained.",
                    flush=True,
                )
                stop_during_epoch = True
                break
            anchor = train_index.anchors[anchor_number]
            input_frames, target_frames = load_anchor_frames(anchor, stores, device)
            position_generator.manual_seed(seed * 1_000_003 + epoch * len(train_index) + anchor_number)
            for positions in iter_position_chunks(
                builder.num_positions, int(cfg["train_chunk_size"]), True, position_generator
            ):
                inputs, targets = builder.build_chunk(input_frames, target_frames, positions)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(inputs)
                    loss = multi_horizon_loss(logits, targets, criterion)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(positions)
                item_count += len(positions)
        if stop_during_epoch:
            break
        scheduler.step()

        val_loss, confusion, _, _ = evaluate_anchors(
            model,
            val_index.anchors,
            stores,
            builder,
            int(cfg["eval_chunk_size"]),
            criterion,
        )
        metrics = scores_from_confusion(confusion)
        mean_macro_f1 = float(np.mean(metrics["macro_f1"]))
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / item_count,
            "val_loss": val_loss,
            "mean_val_macro_f1": mean_macro_f1,
            "metrics": metrics,
            "seconds": time.time() - epoch_start,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        improved = mean_macro_f1 > best_macro_f1
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_macro_f1": max(best_macro_f1, mean_macro_f1),
            "epochs_without_improvement": epochs_without_improvement,
            "input_offsets_minutes": [-20, -17, -14, -11, -8, -6, -4, -2, -1],
            "horizons_minutes": [1, 5, 10, 15],
            "input_shape": [5, 54, 54],
            "spatial_positions_per_anchor": 2601,
            "architecture": "shared_resnet50_four_5class_heads",
            "config": cfg,
            "method_contract": contract,
        }
        atomic_torch_save(checkpoint, output / "last.pt")
        if improved:
            best_macro_f1 = mean_macro_f1
            atomic_torch_save(checkpoint, output / "best.pt")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.stop_file is not None and args.stop_file.is_file():
            print(
                f"Graceful stop requested after completed epoch {epoch}; best.pt is retained.",
                flush=True,
            )
            break
        if epochs_without_improvement >= int(cfg["early_stopping_patience"]):
            print(
                f"Early stopping after {epochs_without_improvement} epochs without improvement.",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
