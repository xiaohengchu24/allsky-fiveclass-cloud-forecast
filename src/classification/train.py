"""
五分类云量分类训练脚本（56x56 样本，效果优先版）
文件名格式：2m4_YYYYMMDDHHMMSS_336_坐标.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report
from PIL import Image
import yaml
from tqdm import tqdm

# ---------------------- 工具函数 ----------------------
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def extract_group_id(filename: str) -> str:
    """从文件名提取大图 ID（去掉最后两个坐标字段）"""
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:-2])
    return stem

# ---------------------- Focal Loss ----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss

# ---------------------- Mixup ----------------------
def mixup_data(x, y, alpha=0.2, device='cpu'):
    """生成 mixup 数据和标签"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------------------- 数据集（类别自适应增强） ----------------------
class CloudPatchDataset(Dataset):
    def __init__(self, samples, transform=None, minority_transform=None, minority_classes=None):
        self.samples = samples
        self.transform = transform
        self.minority_transform = minority_transform
        self.minority_classes = set(minority_classes) if minority_classes else set()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if label in self.minority_classes and self.minority_transform is not None:
            image = self.minority_transform(image)
        elif self.transform is not None:
            image = self.transform(image)
        return image, label

def gather_samples(data_root: str, class_names: List[str]):
    samples = []
    root = Path(data_root)
    for label_idx, cls_name in enumerate(class_names):
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"目录不存在: {cls_dir}")
        for img_path in cls_dir.glob("*.png"):
            group = extract_group_id(img_path.name)
            samples.append((img_path, label_idx, group))
    return samples

def make_group_split(samples, val_ratio, test_ratio, seed):
    groups = np.array([s[2] for s in samples])
    unique_groups = np.unique(groups)
    np.random.seed(seed)
    np.random.shuffle(unique_groups)
    n_total = len(unique_groups)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test
    test_set = set(unique_groups[:n_test])
    val_set = set(unique_groups[n_test:n_test+n_val])
    train_set = set(unique_groups[n_test+n_val:])
    train_samples = [s for s in samples if s[2] in train_set]
    val_samples = [s for s in samples if s[2] in val_set]
    test_samples = [s for s in samples if s[2] in test_set]
    return train_samples, val_samples, test_samples

def check_group_overlap(sets):
    all_groups = [set(s[2] for s in ss) for ss in sets]
    for i in range(len(all_groups)):
        for j in range(i+1, len(all_groups)):
            if not all_groups[i].isdisjoint(all_groups[j]):
                print("WARNING: 分组重叠！")
                return False
    return True

# ---------------------- 模型构建 ----------------------
def create_model(model_name: str, num_classes: int, pretrained: bool, dropout: float, input_size=56):
    if model_name.startswith("resnet"):
        model_fn = getattr(models, model_name)
        if pretrained:
            from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
            weight_map = {
                "resnet18": ResNet18_Weights.IMAGENET1K_V1,
                "resnet34": ResNet34_Weights.IMAGENET1K_V1,
                "resnet50": ResNet50_Weights.IMAGENET1K_V1,
            }
            weights = weight_map.get(model_name, "IMAGENET1K_V1")
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    elif model_name.startswith("efficientnet"):
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    return model

# ---------------------- 训练/评估循环 ----------------------
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def run_epoch(model, loader, criterion, optimizer, scaler, device,
              train: bool, amp: bool, mixup_alpha=0.0, grad_accum=1):
    model.train(train)
    loss_meter = AverageMeter()
    all_labels, all_preds = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for idx, (imgs, labels) in enumerate(tqdm(loader, leave=False)):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train and idx % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            # Mixup
            if train and mixup_alpha > 0:
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, mixup_alpha, device)
            else:
                labels_a, labels_b, lam = labels, None, 0.0

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(imgs)
                if train and mixup_alpha > 0:
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam) / grad_accum
                else:
                    loss = criterion(logits, labels_a) / grad_accum

            if train:
                scaler.scale(loss).backward()
                if (idx + 1) % grad_accum == 0 or (idx + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels_a.cpu().numpy())
            loss_meter.update(loss.item() * grad_accum, imgs.size(0))
    return loss_meter.avg, np.concatenate(all_labels), np.concatenate(all_preds)

def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro'),
        "f1_weighted": f1_score(labels, preds, average='weighted'),
    }

def get_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

# ---------------------- 主函数 ----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_56.yaml")
    return parser.parse_args()

def load_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ["epochs","batch_size","warmup_epochs","early_stop_patience","num_workers","num_classes","image_size"]:
        cfg[key] = int(cfg[key])
    for key in ["lr","weight_decay","label_smoothing","rotation_limit","val_ratio","test_ratio","dropout","focal_gamma","mixup_alpha"]:
        cfg[key] = float(cfg[key])
    cfg["pretrained"] = bool(cfg["pretrained"])
    cfg["minority_enhance"] = bool(cfg.get("minority_enhance", True))
    if "minority_classes" not in cfg or cfg["minority_classes"] is None:
        cfg["minority_classes"] = []
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args)
    seed_everything(cfg["seed"])
    out_dir = ensure_dir(Path(cfg["output_dir"]))
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    class_names = cfg["class_names"]
    num_classes = len(class_names)
    samples = gather_samples(cfg["data_root"], class_names)
    print(f"总样本数: {len(samples)}")

    train_samples, val_samples, test_samples = make_group_split(
        samples, cfg["val_ratio"], cfg["test_ratio"], cfg["seed"]
    )
    print(f"训练集 {len(train_samples)} | 验证集 {len(val_samples)} | 测试集 {len(test_samples)}")
    if not check_group_overlap([train_samples, val_samples, test_samples]):
        raise RuntimeError("分组重叠，请检查 extract_group_id 逻辑！")

    for name, sset in [("训练集", train_samples), ("验证集", val_samples), ("测试集", test_samples)]:
        dist = Counter([x[1] for x in sset])
        str_dist = {class_names[k]: v for k, v in sorted(dist.items())}
        print(f"{name} 类别分布: {str_dist}")

    train_labels_arr = np.array([s[1] for s in train_samples])
    class_counts = np.bincount(train_labels_arr, minlength=num_classes)
    class_ratios = class_counts / len(train_samples)
    print("各类别样本比例:", {class_names[i]: f"{class_ratios[i]:.2%}" for i in range(num_classes)})

    minority_classes = cfg["minority_classes"]
    if cfg["minority_enhance"] and len(minority_classes) == 0:
        # 手动指定少数类（按经验 Partial visibility 和 unknown bright area）
        minority_classes = [2, 3]  # 索引对应 class_names 列表
        print(f"默认少数类: {[class_names[i] for i in minority_classes]}")
    else:
        print(f"手动指定少数类: {[class_names[i] for i in minority_classes]}")

    # ---------- 数据增强（56x56 专用） ----------
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 多数类增强（温和）
    major_train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),                     # 小尺寸下角度不宜过大
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 少数类增强（加强几何和颜色）
    minor_train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),                     # 更大幅度旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = CloudPatchDataset(
        train_samples,
        transform=major_train_tf,
        minority_transform=minor_train_tf,
        minority_classes=minority_classes if cfg["minority_enhance"] else []
    )
    val_ds = CloudPatchDataset(val_samples, transform=eval_tf)
    test_ds = CloudPatchDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg["model_name"], num_classes, cfg["pretrained"], cfg["dropout"]).to(device)

    class_weights = get_class_weights(train_labels_arr, num_classes).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=cfg["focal_gamma"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    total_epochs = cfg["epochs"]
    warmup = cfg["warmup_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")

    best_f1 = -1.0
    patience = 0

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","lr","train_loss","val_loss",
                                               "val_acc","val_f1_macro","val_f1_weighted"])
        writer.writeheader()

        for epoch in range(1, total_epochs+1):
            if epoch <= warmup:
                lr = cfg["lr"] * epoch / warmup
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif epoch == warmup + 1:
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg["lr"]
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-warmup)

            t_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                     train=True, amp=cfg["amp"] and device.type=="cuda",
                                     mixup_alpha=cfg["mixup_alpha"])
            v_loss, v_labels, v_preds = run_epoch(model, val_loader, criterion, optimizer, scaler, device,
                                                  train=False, amp=cfg["amp"] and device.type=="cuda",
                                                  mixup_alpha=0.0)

            if epoch > warmup:
                scheduler.step()

            metrics = compute_metrics(v_labels, v_preds)
            lr_now = optimizer.param_groups[0]["lr"]
            writer.writerow({
                "epoch": epoch, "lr": lr_now, "train_loss": t_loss, "val_loss": v_loss,
                "val_acc": metrics["accuracy"], "val_f1_macro": metrics["f1_macro"],
                "val_f1_weighted": metrics["f1_weighted"]
            })
            f.flush()
            print(f"Epoch {epoch:03d} | lr {lr_now:.2e} | t_loss {t_loss:.4f} | v_loss {v_loss:.4f} | "
                  f"acc {metrics['accuracy']:.4f} | f1_macro {metrics['f1_macro']:.4f}")

            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                patience = 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "val_metrics": metrics},
                           out_dir / "best_model56.pt")
            else:
                patience += 1
                if patience >= cfg["early_stop_patience"]:
                    print("Early stopping.")
                    break

    # 测试
    print("\n--- 测试集评估 ---")
    ckpt = torch.load(out_dir / "best_model56.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    t_loss, t_labels, t_preds = run_epoch(model, test_loader, criterion, optimizer, scaler, device,
                                          train=False, amp=cfg["amp"] and device.type=="cuda",
                                          mixup_alpha=0.0)
    test_metrics = compute_metrics(t_labels, t_preds)
    print(f"Test Loss: {t_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-macro: {test_metrics['f1_macro']:.4f}")
    print(f"F1-weighted: {test_metrics['f1_weighted']:.4f}")
    report = classification_report(t_labels, t_preds, target_names=class_names, digits=4)
    print(report)
    with open(out_dir / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open(out_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump({"test_loss": t_loss, "test_metrics": test_metrics, "best_val_f1_macro": best_f1}, f, indent=2)

if __name__ == "__main__":
    main()