# train_pipeline_v5_kfold.py
"""
Single-stream ResNet1D (with SE attention) training pipeline using 5-fold CV and class weighting.
Dataset structure:
  ROOT/
    ClassA/
      file1.csv
      file2.xlsx
    ClassB/
      ...

Reads ONLY these 11 columns (Virtual Coordinate System + derived features):
  v_pitch, v_roll, v_yaw_rate, v_linear_acc_x, v_linear_acc_y, v_linear_acc_z, v_z_highpass,
  v_jerk_x, v_jerk_y, v_jerk_z, v_acc_mag

Preprocessing:
  - Sliding windows: window_size=125 (2.5s @ 50Hz), step_size=64 (~50% overlap)
  - Zero-padding: files shorter than window_size are padded instead of discarded
  - StandardScaler (fit per-fold on TRAIN only), saved as scaler_fold{K}.pkl
  - Label encoding from folder names, saved to label_map.pkl

Model:
  - 1D ResNet with BatchNorm + Dropout + SEBlock attention and global average pooling

Artifacts:
  - best_model_fold{K}.pth
  - scaler_fold{K}.pkl
  - label_map.pkl
  - test_dataset_reserved/ (10% final holdout, copied physically)

Dependencies (install as needed):
  pip install numpy pandas scikit-learn torch openpyxl
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Configuration / Constants
# -----------------------------
FEATURE_COLS: List[str] = [
    "v_pitch",
    "v_roll",
    "v_yaw_rate",
    "v_linear_acc_x",
    "v_linear_acc_y",
    "v_linear_acc_z",
    "v_z_highpass",
    "v_jerk_x",
    "v_jerk_y",
    "v_jerk_z",
    "v_acc_mag",
]


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# -----------------------------
# Data discovery / loading
# -----------------------------
def discover_files(root_dir: Path, exts: Sequence[str] = (".csv", ".xlsx", ".xls")) -> List[Tuple[Path, str]]:
    """
    Recursively finds files under root_dir.
    Label is the *top-level folder name* under root_dir:
      root/LabelName/file.ext -> LabelName
      root/LabelName/subdir/file.ext -> LabelName
    """
    root_dir = root_dir.expanduser().resolve()
    pairs: List[Tuple[Path, str]] = []

    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        try:
            rel = p.relative_to(root_dir)
        except Exception:
            continue

        if len(rel.parts) < 2:
            continue

        label_name = rel.parts[0]
        pairs.append((p, label_name))

    return pairs


def load_timeseries(path: Path, feature_cols: Sequence[str]) -> np.ndarray:
    """
    Loads a single file (.csv or .xlsx/.xls), selects ONLY feature_cols, returns float32 ndarray [T, C].

    Robust handling:
      - coerces non-numeric to NaN -> interpolate -> ffill/bfill -> nan_to_num
      - raises if required columns missing
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.loc[:, list(feature_cols)].copy()

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df.isna().values.any():
        df = df.interpolate(method="linear", limit_direction="both", axis=0)
        df = df.ffill().bfill()

    arr = df.to_numpy(dtype=np.float32, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return arr


def validate_files(
    file_label_pairs: Sequence[Tuple[Path, str]],
    feature_cols: Sequence[str],
    min_len: int,
    logger: logging.Logger,
) -> Tuple[List[Path], List[str], Dict[Path, int]]:
    valid_paths: List[Path] = []
    valid_labels: List[str] = []
    lengths: Dict[Path, int] = {}

    skipped = 0
    for path, label in file_label_pairs:
        try:
            x = load_timeseries(path, feature_cols)
            T = int(x.shape[0])
            if T < min_len:
                raise ValueError(f"Too short: T={T} < min_len={min_len}")
            valid_paths.append(path)
            valid_labels.append(label)
            lengths[path] = T
        except Exception as e:
            skipped += 1
            logger.warning(f"Skipping file: {path} | reason: {type(e).__name__}: {e}")

    logger.info(f"Discovered {len(file_label_pairs)} files; valid={len(valid_paths)}; skipped={skipped}.")
    return valid_paths, valid_labels, lengths


def build_label_map(labels: Sequence[str]) -> Dict[str, int]:
    classes = sorted(set(labels))
    return {name: i for i, name in enumerate(classes)}


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def fit_scaler(
    train_paths: Sequence[Path],
    feature_cols: Sequence[str],
    logger: logging.Logger,
) -> StandardScaler:
    scaler = StandardScaler(with_mean=True, with_std=True)
    n_files_ok = 0
    for p in train_paths:
        try:
            x = load_timeseries(p, feature_cols)
            scaler.partial_fit(x)
            n_files_ok += 1
        except Exception as e:
            logger.warning(f"Scaler fit: skipping {p} | {type(e).__name__}: {e}")

    if n_files_ok == 0:
        raise RuntimeError("Scaler could not be fit: no valid training files were readable during fitting.")
    return scaler


# -----------------------------
# Dataset
# -----------------------------
@dataclass(frozen=True)
class WindowIndex:
    file_idx: int
    start: int


class SlidingWindowDataset(Dataset):
    """
    Each dataset item is one sliding window [C, window_size] plus its label.
    """

    def __init__(
        self,
        file_paths: Sequence[Path],
        file_labels: Sequence[int],
        file_lengths: Sequence[int],
        feature_cols: Sequence[str],
        window_size: int,
        step_size: int,
        scaler: Optional[StandardScaler],
        cache_size: int = 2,
        logger: Optional[logging.Logger] = None,
    ):
        if len(file_paths) != len(file_labels) or len(file_paths) != len(file_lengths):
            raise ValueError("file_paths, file_labels, and file_lengths must have the same length.")

        self.file_paths = list(file_paths)
        self.file_labels = list(file_labels)
        self.file_lengths = list(file_lengths)
        self.feature_cols = list(feature_cols)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.scaler = scaler
        self.cache_size = int(max(cache_size, 0))
        self.logger = logger

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive.")

        self.index: List[WindowIndex] = []
        self._build_index()

        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def _build_index(self) -> None:
        self.index.clear()

        for i, T in enumerate(self.file_lengths):
            if T <= 0:
                if self.logger:
                    self.logger.warning(f"File has no rows, skipping in index: {self.file_paths[i]}")
                continue

            if T < self.window_size:
                self.index.append(WindowIndex(file_idx=i, start=0))
                continue

            max_start = T - self.window_size
            starts = list(range(0, max_start + 1, self.step_size))
            if len(starts) == 0 or starts[-1] != max_start:
                starts.append(max_start)
            for start in starts:
                self.index.append(WindowIndex(file_idx=i, start=start))

        if len(self.index) == 0:
            raise RuntimeError("No windows were generated. Check window_size/step_size and file lengths.")

    def __len__(self) -> int:
        return len(self.index)

    def _get_series(self, file_idx: int) -> np.ndarray:
        if self.cache_size > 0 and file_idx in self._cache:
            x = self._cache.pop(file_idx)
            self._cache[file_idx] = x
            return x

        path = self.file_paths[file_idx]
        x = load_timeseries(path, self.feature_cols)

        if self.cache_size > 0:
            self._cache[file_idx] = x
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wi = self.index[idx]
        label = int(self.file_labels[wi.file_idx])

        try:
            x = self._get_series(wi.file_idx)
            window = x[wi.start : wi.start + self.window_size, :]

            if self.scaler is not None:
                window = self.scaler.transform(window)
                window = window.astype(np.float32, copy=False)

            pad_len = self.window_size - window.shape[0]
            if pad_len > 0:
                pad = np.zeros((pad_len, window.shape[1]), dtype=window.dtype)
                window = np.concatenate([window, pad], axis=0)

            window = np.transpose(window, (1, 0))
            x_tensor = torch.from_numpy(window).float()
            y_tensor = torch.tensor(label, dtype=torch.long)
            return x_tensor, y_tensor

        except Exception as e:
            if self.logger:
                path = self.file_paths[wi.file_idx]
                self.logger.warning(f"Dataset __getitem__ error for {path} @ start={wi.start} | {type(e).__name__}: {e}")

            x_tensor = torch.zeros((len(self.feature_cols), self.window_size), dtype=torch.float32)
            y_tensor = torch.tensor(label, dtype=torch.long)
            return x_tensor, y_tensor


# -----------------------------
# Model: 1D ResNet with SE attention
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels, reduction=16)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + self.shortcut(identity)
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 128, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, kernel_size=7, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks=2, kernel_size=5, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=2, kernel_size=3, stride=2, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = [
            ResidualBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout=dropout),
        ]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


# -----------------------------
# Training / Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += float(loss.item()) * yb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

        del loss

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_amp: bool,
    max_grad_norm: Optional[float] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    scaler_amp = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler_amp.scale(loss).backward()

        if max_grad_norm is not None:
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += float(loss.item()) * yb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

        del loss

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D-CNN for cat behavior classification with 5-fold CV.")

    parser.add_argument(
        "--root_dir",
        type=str,
        default="./CATS2",
        help="Root dataset directory (Root -> ClassFolder -> file.csv/.xlsx).",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Where to save model/scaler/label map.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--window_size", type=int, default=125)
    parser.add_argument("--step_size", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Enable more deterministic behavior (slower).")

    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader workers.")
    parser.add_argument("--cache_size", type=int, default=2, help="Per-worker file cache size inside Dataset.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--base_channels", type=int, default=128, help="Base channel width for the 1D ResNet.")

    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Gradient clipping (set <=0 to disable).")

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Root dir: {root_dir}")
    logger.info(f"Output dir: {output_dir}")

    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type != "cuda":
        logger.warning("CUDA not available; training will run on CPU.")

    # 1) Discover files
    pairs = discover_files(root_dir)
    if len(pairs) == 0:
        raise RuntimeError(
            f"No .csv/.xlsx files found under: {root_dir}\n"
            "Expected structure: Root/Class_Label_Folder/file.csv or file.xlsx"
        )

    # 2) Validate readable files (keep short sequences, pad later)
    valid_paths, valid_label_names, lengths_by_path = validate_files(
        pairs, FEATURE_COLS, min_len=1, logger=logger
    )
    if len(valid_paths) == 0:
        raise RuntimeError("No valid files after validation. Check required columns and file formats.")

    # 3) Build label map + encode
    label_map = build_label_map(valid_label_names)
    inv_label_map = {v: k for k, v in label_map.items()}
    logger.info(f"Classes ({len(label_map)}): {label_map}")
    save_pickle(
        {"label_to_index": label_map, "index_to_label": inv_label_map},
        output_dir / "label_map.pkl",
    )

    y_all = np.array([label_map[name] for name in valid_label_names], dtype=np.int64)

    # 4) Hold out 10% as final test set (physical copy)
    indices = np.arange(len(valid_paths))
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=0.1,
        random_state=int(args.seed),
        shuffle=True,
        stratify=y_all,
    )

    test_paths = [valid_paths[i] for i in test_idx]
    test_label_names = [valid_label_names[i] for i in test_idx]
    trainval_paths = [valid_paths[i] for i in trainval_idx]
    trainval_labels = [int(y_all[i]) for i in trainval_idx]
    trainval_lengths = [int(lengths_by_path[p]) for p in trainval_paths]

    logger.info(f"Reserved test files: {len(test_paths)} | Remaining for CV: {len(trainval_paths)}")

    # Physical copy of test set
    test_export_dir = output_dir / "test_dataset_reserved"
    test_export_dir.mkdir(parents=True, exist_ok=True)
    copied_test_files = 0
    for src_path, label_name in zip(test_paths, test_label_names):
        dest_dir = test_export_dir / label_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name
        try:
            shutil.copy2(src_path, dest_path)
            copied_test_files += 1
        except Exception as e:
            logger.warning(f"Failed to copy test file {src_path} -> {dest_path} | {type(e).__name__}: {e}")
    logger.info(f"Test set exported to: {test_export_dir} | copied files: {copied_test_files}")

    # Common DataLoader settings tuned for throughput
    pin_memory = True
    loader_common_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": pin_memory,
        "drop_last": False,
        "persistent_workers": True if int(args.num_workers) > 0 else False,
    }
    if int(args.num_workers) > 0:
        loader_common_kwargs["prefetch_factor"] = 2

    max_grad_norm = args.max_grad_norm if args.max_grad_norm and args.max_grad_norm > 0 else None
    num_classes = len(label_map)
    base_channels = int(args.base_channels)

    # 5) 5-Fold CV on remaining data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(args.seed))

    for fold, (train_indices, val_indices) in enumerate(skf.split(trainval_paths, trainval_labels)):
        logger.info(f"===== Fold {fold + 1}/5 =====")

        fold_train_paths = [trainval_paths[i] for i in train_indices]
        fold_val_paths = [trainval_paths[i] for i in val_indices]

        fold_train_labels = [trainval_labels[i] for i in train_indices]
        fold_val_labels = [trainval_labels[i] for i in val_indices]

        fold_train_lengths = [trainval_lengths[i] for i in train_indices]
        fold_val_lengths = [trainval_lengths[i] for i in val_indices]

        # Class weights for this fold
        y_train_fold = np.array(fold_train_labels, dtype=np.int64)
        classes_array = np.arange(num_classes, dtype=np.int64)
        
        # 1. 计算原始平衡权重
        raw_weights = compute_class_weight(class_weight="balanced", classes=classes_array, y=y_train_fold)
        
        # 2. 【关键修改】应用“平方根平滑” (Square Root Smoothing)
        # 作用：将极端权重（如 1:5）温和化为（1:2.2），既照顾小样本，又不至于让模型对大样本（如基础位移）过敏
        soft_weights = np.power(raw_weights, 0.5)
        
        # 转换 Tensor
        class_weights_tensor = torch.tensor(soft_weights, dtype=torch.float32, device=device)
        
        logger.info(f"[Fold {fold + 1}] Raw weights: {raw_weights}")
        logger.info(f"[Fold {fold + 1}] Soft weights (used): {soft_weights}")

        # Fit scaler on train split only
        scaler = fit_scaler(fold_train_paths, FEATURE_COLS, logger=logger)
        save_pickle(scaler, output_dir / f"scaler_fold{fold}.pkl")

        # Datasets + loaders
        train_dataset = SlidingWindowDataset(
            file_paths=fold_train_paths,
            file_labels=fold_train_labels,
            file_lengths=fold_train_lengths,
            feature_cols=FEATURE_COLS,
            window_size=args.window_size,
            step_size=args.step_size,
            scaler=scaler,
            cache_size=args.cache_size,
            logger=logger,
        )

        val_dataset = SlidingWindowDataset(
            file_paths=fold_val_paths,
            file_labels=fold_val_labels,
            file_lengths=fold_val_lengths,
            feature_cols=FEATURE_COLS,
            window_size=args.window_size,
            step_size=args.step_size,
            scaler=scaler,
            cache_size=args.cache_size,
            logger=logger,
        )

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_common_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_common_kwargs,
        )

        logger.info(f"[Fold {fold + 1}] Train windows: {len(train_dataset)} | Val windows: {len(val_dataset)}")

        # Model / optimizer / loss
        model = ResNet1D(
            in_channels=len(FEATURE_COLS),
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=float(args.dropout),
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        best_metric = -1.0
        best_path = output_dir / f"best_model_fold{fold}.pth"

        for epoch in range(1, int(args.epochs) + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                use_amp=bool(args.amp),
                max_grad_norm=max_grad_norm,
            )

            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            logger.info(
                f"[Fold {fold + 1}/5] Epoch {epoch:03d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            metric = val_acc
            if metric > best_metric:
                best_metric = metric
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "num_classes": num_classes,
                        "feature_cols": FEATURE_COLS,
                        "window_size": int(args.window_size),
                        "step_size": int(args.step_size),
                        "label_map": label_map,
                        "model_arch": "ResNet1D_SE",
                        "base_channels": base_channels,
                        "dropout": float(args.dropout),
                        "fold": fold,
                    },
                    best_path,
                )
                logger.info(f"[Fold {fold + 1}] Saved new best model to {best_path} (metric={best_metric:.4f})")

        logger.info(f"[Fold {fold + 1}] Training complete. Best Val Acc: {best_metric:.4f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("All folds completed.")
    logger.info(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    # Required for Windows when using DataLoader(num_workers>0)
    main()
