import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
import sys
import os

# 添加ai_service模块路径
sys.path.insert(0, str(Path(__file__).parent))

# 从ai_service导入必要的函数
from ai_service import (
    convert_csv_to_virtual_features,
    perform_inference_from_csvs,
    kelong_model,
    kelong_scaler,
    kelong_label_map,
    processed_model,
    processed_scaler,
    processed_feature_cols
)

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
    parser = argparse.ArgumentParser(description="猫行为识别 - 双模型版本")
    parser.add_argument("--input", type=str, required=True, help="原始IMU数据CSV文件路径（格式：sequence, timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（可选，用于保存生成的特征文件）")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: 文件不存在: {input_path}")
        return
    
    # 检查模型是否已加载
    if kelong_model is None and processed_model is None:
        print("❌ Error: 模型未加载，请检查模型文件路径")
        return
    
    print(f"📂 读取文件: {input_path}")
    
    # --- A. 从CSV文件读取并转换为虚拟特征 ---
    try:
        timestamp_df, window_df = convert_csv_to_virtual_features(input_path)
        print(f"✅ 成功生成特征:")
        print(f"   - per_timestamp特征: {len(timestamp_df)} 行")
        print(f"   - per_window特征: {len(window_df)} 行")
    except Exception as e:
        print(f"❌ Error: 特征转换失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- B. 保存特征文件（可选）---
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = input_path.stem
        timestamp_csv = output_dir / f"{base_name}_virtual_per_timestamp_feature.csv"
        window_csv = output_dir / f"{base_name}_virtual_per_window_feature.csv"
        
        timestamp_df.to_csv(timestamp_csv, index=False)
        window_df.to_csv(window_csv, index=False)
        print(f"💾 特征文件已保存:")
        print(f"   - {timestamp_csv}")
        print(f"   - {window_csv}")
    else:
        # 使用临时文件
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        base_name = input_path.stem
        timestamp_csv = temp_dir / f"{base_name}_virtual_per_timestamp_feature.csv"
        window_csv = temp_dir / f"{base_name}_virtual_per_window_feature.csv"
        
        timestamp_df.to_csv(timestamp_csv, index=False)
        window_df.to_csv(window_csv, index=False)
    
    # --- C. 执行推理 ---
    print("\n🔍 执行推理...")
    try:
        inference_result = perform_inference_from_csvs(timestamp_csv, window_csv)
        
        if inference_result.get("status") == "error":
            print(f"❌ Error: {inference_result.get('error', 'Unknown error')}")
            return
        
        action = inference_result.get("action", "unknown")
        confidence = inference_result.get("confidence", 0.0)
        model_used = inference_result.get("model_used", "Unknown")
        
        # --- D. 输出结果 ---
        print("\n" + "="*50)
        print(f"🐱 识别结果: 【 {action} 】")
        print(f"🎯 置信度:   {confidence:.2%}")
        print(f"🤖 使用模型: {model_used}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error: 推理失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件（如果不是用户指定的输出目录）
        if not args.output_dir:
            try:
                if timestamp_csv.exists():
                    timestamp_csv.unlink()
                if window_csv.exists():
                    window_csv.unlink()
            except:
                pass

if __name__ == "__main__":
    main()