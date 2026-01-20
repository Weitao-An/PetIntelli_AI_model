import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")

# ===========================================================
# 1. 配置区域 (10个基础特征)
# ===========================================================
READ_COLS = [
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
]
FEATURE_COLS = READ_COLS + ["v_acc_mag"]

# ===========================================================
# 2. 模型定义
# ===========================================================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        ident = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        return self.relu(out + ident)

class ResNet1D(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=128, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, kernel_size=7, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks=2, kernel_size=5, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=2, kernel_size=3, stride=2, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, blocks, kernel_size, stride, dropout):
        layers = [
            ResidualBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout=dropout),
        ]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

# ===========================================================
# 3. 数据预处理
# ===========================================================
def preprocess_excel(file_path, scaler):
    try:
        if str(file_path).lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception:
        return None

    missing = [c for c in READ_COLS if c not in df.columns]
    if missing:
        return None

    df_subset = df[READ_COLS].copy()
    for c in READ_COLS:
        df_subset[c] = pd.to_numeric(df_subset[c], errors="coerce")
    
    if df_subset.isna().any().any():
        df_subset = df_subset.interpolate(method='linear').ffill().bfill()
        df_subset = df_subset.fillna(0)

    acc_x = df_subset["v_linear_acc_x"].to_numpy(dtype=np.float32)
    acc_y = df_subset["v_linear_acc_y"].to_numpy(dtype=np.float32)
    acc_z = df_subset["v_linear_acc_z"].to_numpy(dtype=np.float32)
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2).reshape(-1, 1)

    data = df_subset.to_numpy(dtype=np.float32)
    data = np.concatenate([data, acc_mag], axis=1)

    try:
        data_scaled = scaler.transform(data)
    except Exception:
        return None
    
    return data_scaled

# ===========================================================
# 4. 主程序
# ===========================================================
def main():
    parser = argparse.ArgumentParser(description="猫行为识别 - 纯净版")
    parser.add_argument("--input", type=str, required=True, help="文件路径")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="模型目录")
    args = parser.parse_args()

    # 这里的 print 删除了
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- A. 加载配置和模型 ---
    artifacts = Path(args.artifacts_dir)
    
    label_map_path = artifacts / "label_map.pkl"
    if not label_map_path.exists():
        print("❌ Error: 缺少 label_map.pkl")
        return
    with open(label_map_path, "rb") as f:
        data = pickle.load(f)
        idx_to_label = data["index_to_label"]
        num_classes = len(idx_to_label)

    scaler_path = artifacts / "scaler_fold0.pkl"
    if not scaler_path.exists():
        print("❌ Error: 缺少 scaler_fold0.pkl")
        return
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model0_path = artifacts / "best_model_fold0.pth"
    if not model0_path.exists():
         print("❌ Error: 缺少模型文件")
         return
         
    ckpt = torch.load(model0_path, map_location=device)
    base_channels = ckpt.get("base_channels", 128)
    window_size = ckpt["window_size"]
    
    # 这里的 "加载模型中..." print 删除了
    models = []
    for fold in range(5):
        p = artifacts / f"best_model_fold{fold}.pth"
        if p.exists():
            m = ResNet1D(in_channels=11, num_classes=num_classes, base_channels=base_channels, dropout=0.0).to(device)
            state_dict = torch.load(p, map_location=device)["model_state_dict"]
            m.load_state_dict(state_dict)
            m.eval()
            models.append(m)
    
    if not models:
        print("❌ Error: 未加载到模型")
        return

    # --- B. 处理数据 ---
    input_path = Path(args.input)
    # 这里的 "分析文件..." print 删除了
    
    data = preprocess_excel(input_path, scaler)
    if data is None: 
        print("❌ Error: 数据处理失败")
        return

    # --- C. 切片 ---
    T = data.shape[0]
    step = window_size // 2  
    windows = []

    if T < window_size:
        pad = np.zeros((window_size - T, 11), dtype=np.float32)
        w = np.concatenate([data, pad], axis=0)
        windows.append(w)
    else:
        for start in range(0, T - window_size + 1, step):
            w = data[start : start + window_size]
            windows.append(w)
        if T > window_size and (T - window_size) % step != 0:
            windows.append(data[-window_size:])

    batch_input = np.array(windows) 
    batch_input = np.transpose(batch_input, (0, 2, 1)) 
    tensor_input = torch.tensor(batch_input).float().to(device)

    # --- D. 推理 ---
    # 这里的 "计算中..." print 删除了
    with torch.no_grad():
        ensemble_logits = torch.zeros((tensor_input.size(0), num_classes)).to(device)
        for m in models:
            logits = m(tensor_input)
            probs = torch.softmax(logits, dim=1)
            ensemble_logits += probs
        
        avg_probs = ensemble_logits / len(models)
        file_final_prob = avg_probs.mean(dim=0)
        pred_idx = torch.argmax(file_final_prob).item()
        confidence = file_final_prob[pred_idx].item()

    # --- E. 输出结果 ---
    pred_label = idx_to_label[pred_idx]
    display_label = pred_label.replace("大类_", "")

    print("\n" + "="*40)
    print(f"🐱 识别结果: 【 {display_label} 】")
    print(f"🎯 置信度:   {confidence:.2%}")
    print("="*40 + "\n")

    print("--- 可能性排名 ---")
    top3_vals, top3_idxs = torch.topk(file_final_prob, k=min(3, num_classes))
    for i in range(len(top3_idxs)):
        idx = top3_idxs[i].item()
        score = top3_vals[i].item()
        name = idx_to_label[idx].replace("大类_", "")
        print(f"{i+1}. {name}: {score:.2%}")

if __name__ == "__main__":
    main()