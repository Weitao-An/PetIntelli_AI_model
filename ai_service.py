import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import logging
import pickle
import time
from typing import List, Any, Optional, Dict
import redis
from datetime import datetime
from pathlib import Path
import os
import tempfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 1. 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
TEMP_DIR.mkdir(exist_ok=True)

# 特征列定义
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

# --- 环境配置（参考 constants.py 规范）---
ENV = os.getenv("ENV", "dev")  # 环境标识：dev/test/prod

# Redis 配置（与 web 端保持一致）
REDIS_HOST = os.getenv("REDIS_HOST", "47.97.19.58")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "petpypkj2025")  # 从环境变量读取，生产环境必须设置
REDIS_DECODE = os.getenv("REDIS_DECODE", "True").lower() == "true"  # 是否自动 decode
REDIS_RECONNECT_DELAY = int(os.getenv("REDIS_RECONNECT_DELAY", 5))  # 重连延迟（秒）

# Redis Key 前缀（按规范：pi:{env}）
REDIS_KEY_PREFIX = f"pi:{ENV}"

# Redis Stream 配置
# 可以通过环境变量 REDIS_STREAM_TELEMETRY 自定义，否则使用默认格式
REDIS_STREAM_TELEMETRY = os.getenv(
    "REDIS_STREAM_TELEMETRY", 
    f"pi:{ENV}:stream:telemetry_ai"  # 默认格式：pi:{env}:stream:telemetry_ai
)  # AI遥测数据流
REDIS_STREAM_CONSUMER_GROUP = "ai_service"  # 消费者组名称
REDIS_STREAM_CONSUMER_NAME = f"ai_worker_{os.getpid()}"  # 消费者名称（使用进程ID）

# Redis Key 模板
REDIS_KEY_DEV_LATEST = f"{REDIS_KEY_PREFIX}:dev:{{nfc_uid}}:latest"  # Hash Key
REDIS_PUB_CHANNEL = f"{REDIS_KEY_PREFIX}:pub:{{nfc_uid}}"  # Pub/Sub 频道


# --- 2. 模型定义 ---
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

# --- 3. 加载模型和配置 ---
models = []
label_map = {}
scaler = None
window_size = 125
base_channels = 128

try:
    # 加载标签映射
    label_map_path = ARTIFACTS_DIR / "label_map.pkl"
    if label_map_path.exists():
        with open(label_map_path, "rb") as f:
            data = pickle.load(f)
            label_map = data.get("index_to_label", {})
            num_classes = len(label_map)
    else:
        logger.error("缺少 label_map.pkl")
        exit(1)
    
    # 加载scaler
    scaler_path = ARTIFACTS_DIR / "scaler_fold0.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        logger.error("缺少 scaler_fold0.pkl")
        exit(1)
    
    # 加载模型配置
    model0_path = ARTIFACTS_DIR / "best_model_fold0.pth"
    if model0_path.exists():
        ckpt = torch.load(model0_path, map_location=DEVICE)
        base_channels = ckpt.get("base_channels", 128)
        window_size = ckpt.get("window_size", 125)
    else:
        logger.error("缺少模型文件")
        exit(1)
    
    # 加载所有fold的模型
    for fold in range(5):
        p = ARTIFACTS_DIR / f"best_model_fold{fold}.pth"
        if p.exists():
            m = ResNet1D(in_channels=11, num_classes=num_classes, base_channels=base_channels, dropout=0.0).to(DEVICE)
            state_dict = torch.load(p, map_location=DEVICE)["model_state_dict"]
            m.load_state_dict(state_dict)
            m.eval()
            models.append(m)
    
    if not models:
        logger.error("未加载到模型")
        exit(1)
    
    logger.info(f"成功加载 {len(models)} 个模型到: {DEVICE}")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    exit(1)

# --- 4. IMU数据转换函数 ---
def convert_imu_to_virtual_features(imu_data: List[Dict], base_timestamp_ms: int = 0) -> pd.DataFrame:
    """
    将原始IMU数据转换为虚拟坐标系特征
    
    支持两种格式：
    1. 新格式（EMQX）：samples 包含 ax, ay, az, gx, gy, gz, dt_ms（相对时间）
    2. 旧格式：包含 acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, timestamp（绝对时间）
    
    Args:
        imu_data: IMU数据列表
        base_timestamp_ms: 基准时间戳（毫秒），用于将 dt_ms 转换为绝对时间戳
        
    Returns:
        包含虚拟坐标系特征的DataFrame
    """
    if not imu_data:
        raise ValueError("IMU数据为空")
    
    # 转换为DataFrame
    df = pd.DataFrame(imu_data)
    
    # 检查是新格式还是旧格式
    has_new_format = 'ax' in df.columns and 'dt_ms' in df.columns
    has_old_format = 'acc_x' in df.columns and 'timestamp' in df.columns
    
    if has_new_format:
        # 新格式：ax, ay, az, gx, gy, gz, dt_ms
        # 需要转换为旧格式的字段名，并计算绝对时间戳
        required_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'dt_ms']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列（新格式）: {missing_cols}")
        
        # 转换为数值类型
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 将 dt_ms 转换为绝对时间戳（秒）
        # dt_ms 是相对时间，需要累积计算
        dt_ms_values = df['dt_ms'].fillna(40).values  # 默认40ms
        timestamps = []
        current_time_ms = base_timestamp_ms
        for dt_ms in dt_ms_values:
            timestamps.append(current_time_ms / 1000.0)  # 转换为秒
            current_time_ms += dt_ms
        
        df['timestamp'] = timestamps
        
        # 重命名字段以匹配旧格式
        # 注意：保持原始单位，不做转换（模型训练时使用的单位可能就是这个）
        # 如果后续发现需要单位转换，可以在这里调整
        df['acc_x'] = df['ax'].values  # 直接使用原始值
        df['acc_y'] = df['ay'].values
        df['acc_z'] = df['az'].values
        df['gyro_x'] = df['gx'].values  # 直接使用原始值
        df['gyro_y'] = df['gy'].values
        df['gyro_z'] = df['gz'].values
        
    elif has_old_format:
        # 旧格式：acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, timestamp
        required_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列（旧格式）: {missing_cols}")
        
        # 转换为数值类型
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        raise ValueError("无法识别IMU数据格式，需要包含 (ax,ay,az,gx,gy,gz,dt_ms) 或 (acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,timestamp)")
    
    # 按timestamp排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 计算时间间隔（用于导数计算，单位：秒）
    dt = df['timestamp'].diff().fillna(0.04)  # 默认40ms = 0.04秒
    dt = dt.replace(0, 0.04)  # 避免除零
    dt_values = dt.values  # 转换为numpy数组以便后续使用
    
    # 计算pitch和roll（从加速度计）
    acc_x = df['acc_x'].values
    acc_y = df['acc_y'].values
    acc_z = df['acc_z'].values
    
    # 计算加速度幅值
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # 计算pitch和roll（使用atan2）
    # pitch = atan2(acc_x, sqrt(acc_y^2 + acc_z^2))
    # roll = atan2(acc_y, sqrt(acc_x^2 + acc_z^2))
    pitch = np.arctan2(acc_x, np.sqrt(acc_y**2 + acc_z**2))
    roll = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
    
    # yaw_rate直接使用gyro_z
    yaw_rate = df['gyro_z'].values
    
    # 线性加速度（假设已经做了重力补偿，直接使用原始值）
    # 如果需要重力补偿，可以减去重力分量
    linear_acc_x = acc_x
    linear_acc_y = acc_y
    # z轴可能需要减去重力（假设重力约为1g，但这里先保持原值）
    linear_acc_z = acc_z - 1.0  # 减去重力加速度（约1g）
    
    # 计算jerk（加速度的导数）
    # 第一行的jerk应该为0
    jerk_x = np.zeros_like(linear_acc_x)
    jerk_y = np.zeros_like(linear_acc_y)
    jerk_z = np.zeros_like(linear_acc_z)
    
    # 计算导数（跳过第一行）
    if len(linear_acc_x) > 1:
        for i in range(1, len(linear_acc_x)):
            if dt_values[i] > 0:
                jerk_x[i] = (linear_acc_x[i] - linear_acc_x[i-1]) / dt_values[i]
                jerk_y[i] = (linear_acc_y[i] - linear_acc_y[i-1]) / dt_values[i]
                jerk_z[i] = (linear_acc_z[i] - linear_acc_z[i-1]) / dt_values[i]
    
    # 计算z轴高通滤波（一阶高通滤波器）
    # 使用更标准的RC高通滤波器实现
    alpha = 0.1  # 滤波系数
    z_highpass = np.zeros_like(linear_acc_z)
    z_highpass[0] = 0  # 第一行为0
    
    for i in range(1, len(linear_acc_z)):
        if dt_values[i] > 0:
            # RC高通滤波器: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
            z_highpass[i] = alpha * (z_highpass[i-1] + linear_acc_z[i] - linear_acc_z[i-1])
    
    # 构建结果DataFrame
    result_df = pd.DataFrame({
        'timestamp': df['timestamp'].values,
        'v_pitch': pitch,
        'v_roll': roll,
        'v_yaw_rate': yaw_rate,
        'v_linear_acc_x': linear_acc_x,
        'v_linear_acc_y': linear_acc_y,
        'v_linear_acc_z': linear_acc_z,
        'v_jerk_x': jerk_x,
        'v_jerk_y': jerk_y,
        'v_jerk_z': jerk_z,
        'v_z_highpass': z_highpass
    })
    
    return result_df

# --- 5. 推理函数 ---
def perform_inference_from_csv(csv_path: Path) -> Dict[str, Any]:
    """
    从CSV文件执行推理
    
    Args:
        csv_path: 特征CSV文件路径
        
    Returns:
        包含推理结果的字典
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 检查必要的列
        missing = [c for c in READ_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV文件缺少必要的列: {missing}")
        
        # 提取特征列
        df_subset = df[READ_COLS].copy()
        for c in READ_COLS:
            df_subset[c] = pd.to_numeric(df_subset[c], errors="coerce")
        
        # 处理缺失值
        if df_subset.isna().any().any():
            df_subset = df_subset.interpolate(method='linear').ffill().bfill()
            df_subset = df_subset.fillna(0)
        
        # 计算加速度幅值
        acc_x = df_subset["v_linear_acc_x"].to_numpy(dtype=np.float32)
        acc_y = df_subset["v_linear_acc_y"].to_numpy(dtype=np.float32)
        acc_z = df_subset["v_linear_acc_z"].to_numpy(dtype=np.float32)
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2).reshape(-1, 1)
        
        # 合并特征
        data = df_subset.to_numpy(dtype=np.float32)
        data = np.concatenate([data, acc_mag], axis=1)
        
        # 标准化
        data_scaled = scaler.transform(data)
        
        # 滑动窗口切片
        T = data_scaled.shape[0]
        step = window_size // 2
        windows = []
        
        if T < window_size:
            pad = np.zeros((window_size - T, 11), dtype=np.float32)
            w = np.concatenate([data_scaled, pad], axis=0)
            windows.append(w)
        else:
            for start in range(0, T - window_size + 1, step):
                w = data_scaled[start : start + window_size]
                windows.append(w)
            if T > window_size and (T - window_size) % step != 0:
                windows.append(data_scaled[-window_size:])
        
        batch_input = np.array(windows)
        batch_input = np.transpose(batch_input, (0, 2, 1))
        tensor_input = torch.tensor(batch_input).float().to(DEVICE)
        
        # 推理
        with torch.no_grad():
            ensemble_logits = torch.zeros((tensor_input.size(0), len(label_map))).to(DEVICE)
            for m in models:
                logits = m(tensor_input)
                probs = torch.softmax(logits, dim=1)
                ensemble_logits += probs
            
            avg_probs = ensemble_logits / len(models)
            file_final_prob = avg_probs.mean(dim=0)
            pred_idx = torch.argmax(file_final_prob).item()
            confidence = file_final_prob[pred_idx].item()
        
        # 获取预测标签
        pred_label = label_map.get(pred_idx, "unknown")
        display_label = pred_label.replace("大类_", "")
        
        result = {
            "action": display_label,
            "confidence": float(confidence),
            "prediction_index": int(pred_idx),
            "status": "success"
        }
        
        logger.info(f"推理成功 - Action: {display_label}, Confidence: {confidence:.2%}")
        return result
        
    except Exception as e:
        error_msg = f"推理出错: {str(e)}"
        logger.error(f"推理失败: {error_msg}")
        return {
            "action": "unknown",
            "confidence": 0.0,
            "status": "error",
            "error": error_msg
        }

# --- 6. Emotion 处理函数 ---
def process_emotion(sound_osslink: str) -> dict:
    """
    处理 emotion 数据（通过 sound_osslink 下载和处理音频）
    
    Args:
        sound_osslink: 音频 OSS 链接路径，例如 "audio/DEVICE_001/2024/12/24/clip_12345.opus"
        
    Returns:
        包含 emotion 结果的字典
    """
    try:
        if not sound_osslink:
            logger.warning("sound_osslink 为空，使用默认 emotion 值")
            return {
                "emotion": "unknown",
                "score": 0
            }
        
        logger.info(f"处理 emotion - OSS Link: {sound_osslink}")
        
        # TODO: 实现音频下载和处理逻辑
        # 1. 从 OSS 下载音频文件
        # 2. 调用 emotion 模型处理音频
        # 3. 返回 emotion 结果
        
        # 暂时使用 Mock 返回
        logger.debug(f"Emotion 处理（Mock）- OSS Link: {sound_osslink}")
        return {
            "emotion": "calm",
            "score": 80
        }
        
    except Exception as e:
        logger.error(f"Emotion 处理失败 - OSS Link: {sound_osslink}, Error: {e}")
        return {
            "emotion": "error",
            "score": 0
        }

# --- 7. Redis 客户端管理 ---
def create_redis_client():
    """
    创建 Redis 客户端
    
    Returns:
        Redis 客户端对象，如果连接失败则返回 None
    """
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=REDIS_DECODE,  # 根据配置决定是否自动 decode（Stream 操作通常需要 False）
            socket_connect_timeout=5,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
        # 测试连接
        client.ping()
        logger.info(f"成功连接到 Redis: {REDIS_HOST}:{REDIS_PORT}")
        return client
    except Exception as e:
        logger.error(f"Redis 连接失败: {e}")
        return None

def ensure_consumer_group(client: redis.Redis, stream_name: str, group_name: str):
    """
    确保 Consumer Group 存在，如果不存在则创建
    新创建的 Consumer Group 将从最新消息开始读取（使用 $）
    
    Args:
        client: Redis 客户端
        stream_name: Stream 名称
        group_name: Consumer Group 名称
        
    Returns:
        bool: 是否新创建了 Consumer Group（True=新创建，False=已存在）
    """
    try:
        # 尝试创建 Consumer Group（从最新消息开始，使用 $）
        client.xgroup_create(
            name=stream_name,
            groupname=group_name,
            id="$",  # $ 表示从最新消息开始读取（只读取创建后的新消息）
            mkstream=True  # 如果 Stream 不存在则创建
        )
        logger.info(f"创建 Consumer Group: {group_name} for Stream: {stream_name} (从最新消息开始)")
        return True  # 新创建
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            # Consumer Group 已存在，这是正常的
            logger.debug(f"Consumer Group 已存在: {group_name}，将只读取新消息")
            return False  # 已存在
        else:
            logger.error(f"创建 Consumer Group 失败: {e}")
            raise

# --- 8. 更新 Redis Hash 单个字段 ---
def update_latest_field(client: redis.Redis, nfc_uid: str, field: str, value: str):
    """
    更新 Redis Hash Key pi:dev:dev:{nfc_uid}:latest 的单个字段
    
    Args:
        client: Redis 客户端
        nfc_uid: NFC UID
        field: 字段名
        value: 字段值
        
    Returns:
        是否成功
    """
    if client is None:
        logger.warning(f"Redis未连接，跳过更新字段 - NFC UID: {nfc_uid}, Field: {field}")
        return False
    
    try:
        # 构建 Redis Hash Key: pi:dev:dev:{nfc_uid}:latest
        redis_key = f"pi:dev:dev:{nfc_uid}:latest"
        
        # 更新单个字段
        client.hset(redis_key, field, str(value))
        
        # 设置过期时间（24小时）
        client.expire(redis_key, 24 * 60 * 60)
        
        logger.debug(f"更新 Redis Hash 字段 - Key: {redis_key}, Field: {field}")
        return True
        
    except Exception as e:
        logger.error(f"更新 Redis Hash 字段失败 - NFC UID: {nfc_uid}, Field: {field}, Error: {e}")
        return False

# --- 8.1 批量更新 Redis Hash 字段 ---
def update_redis_hash_latest(client: redis.Redis, nfc_uid: str, ai_result: dict):
    """
    批量更新 Redis Hash Key pi:dev:dev:{nfc_uid}:latest 的多个字段
    
    Args:
        client: Redis 客户端
        nfc_uid: NFC UID
        ai_result: AI 处理结果字典（包含 emotion_state, emotion_score 等字段）
        
    Returns:
        是否成功
    """
    if client is None:
        logger.warning(f"Redis未连接，跳过写入 Hash - NFC UID: {nfc_uid}")
        return False
    
    try:
        # 构建 Redis Hash Key: pi:dev:dev:{nfc_uid}:latest
        redis_key = f"pi:dev:dev:{nfc_uid}:latest"
        
        # 将 AI 结果字段写入 Hash
        # 只更新 AI 相关的字段，保留其他字段
        hash_data = {}
        if "emotion_state" in ai_result:
            hash_data["emotion_state"] = str(ai_result["emotion_state"])
        if "emotion_score" in ai_result:
            hash_data["emotion_score"] = str(ai_result["emotion_score"])
        if "emotion_message" in ai_result:
            hash_data["emotion_message"] = str(ai_result["emotion_message"])
        if "action" in ai_result:
            hash_data["action"] = str(ai_result["action"])
        if "action_confidence" in ai_result:
            hash_data["action_confidence"] = str(ai_result["action_confidence"])
        if "inference_timestamp" in ai_result:
            hash_data["inference_timestamp"] = str(ai_result["inference_timestamp"])
        
        if hash_data:
            client.hset(redis_key, mapping=hash_data)
            # 设置过期时间（24小时，参考 constants.py 的 TTL_LATEST）
            client.expire(redis_key, 24 * 60 * 60)
            logger.info(f"更新 Redis Hash - Key: {redis_key}, Fields: {list(hash_data.keys())}")
            return True
        else:
            logger.warning(f"没有 AI 结果字段需要更新 - NFC UID: {nfc_uid}")
            return False
        
    except Exception as e:
        logger.error(f"更新 Redis Hash 失败 - NFC UID: {nfc_uid}, Error: {e}")
        return False

# --- 9. 发布结果到 Pub/Sub ---
def publish_ai_result(client: redis.Redis, nfc_uid: str, ai_result: dict):
    """
    通过 Pub/Sub 频道发布 AI 结果
    
    Args:
        client: Redis 客户端
        nfc_uid: NFC UID
        ai_result: AI 处理结果字典
        
    Returns:
        是否成功
    """
    if client is None:
        logger.warning(f"Redis未连接，跳过发布 - NFC UID: {nfc_uid}")
        return False
    
    try:
        # 构建 Pub/Sub 频道: pi:{env}:pub:{nfc_uid}
        channel = REDIS_PUB_CHANNEL.format(nfc_uid=nfc_uid)
        
        # 将结果序列化为JSON字符串
        message_json = json.dumps(ai_result, ensure_ascii=False)
        
        # 发布消息
        client.publish(channel, message_json)
        
        logger.info(f"发布 AI 结果到 Pub/Sub - Channel: {channel}")
        return True
        
    except Exception as e:
        logger.error(f"发布 Pub/Sub 消息失败 - NFC UID: {nfc_uid}, Error: {e}")
        return False

# --- 10. 处理 Stream 消息数据 ---
def process_stream_message(message_data: dict) -> dict:
    """
    处理从 Redis Stream 获取的消息数据
    
    消息格式（EMQX 直接发送的格式）：
    - nfc_uid: 设备ID
    - dev_ts_ms: 设备时间戳（毫秒）
    - battery: 电池电量
    - items: 数组，包含不同类型的数据项
      - items 中 type="imu" 的项包含 samples（字段：ax, ay, az, gx, gy, gz, dt_ms）
      - items 中 type="audio_meta" 的项包含 sound_osslink
    
    Args:
        message_data: Stream 消息数据字典
        
    Returns:
        AI 处理结果字典（包含 emotion_state, emotion_score, action 等字段）
    """
    try:
        # 解析消息顶层字段
        nfc_uid = message_data.get("nfc_uid", "unknown")
        dev_ts_ms = message_data.get("dev_ts_ms", 0)
        
        logger.info(f"开始处理 Stream 消息 - NFC UID: {nfc_uid}, Dev TS: {dev_ts_ms}")
        
        # 如果 message_data 是字符串，先解析 JSON
        if isinstance(message_data, str):
            try:
                message_data = json.loads(message_data)
            except json.JSONDecodeError as e:
                logger.error(f"消息 JSON 解析失败 - NFC UID: {nfc_uid}, Error: {e}")
                return {
                    "nfc_uid": nfc_uid,
                    "action": "error",
                    "action_confidence": 0.0,
                    "emotion_state": "error",
                    "emotion_score": 0,
                    "emotion_message": f"数据解析失败: {str(e)}",
                    "inference_timestamp": datetime.now().isoformat(),
                    "inference_error": f"JSON解析失败: {str(e)}"
                }
        
        # 如果 message_data 是字典但只有一个字段（可能是 "payload" 或 "data"），尝试解析该字段
        if isinstance(message_data, dict) and len(message_data) == 1:
            single_key = list(message_data.keys())[0]
            single_value = message_data[single_key]
            if isinstance(single_value, str):
                try:
                    parsed_json = json.loads(single_value)
                    if isinstance(parsed_json, dict):
                        message_data = parsed_json
                        logger.debug(f"从字段 {single_key} 解析 JSON 消息")
                except (json.JSONDecodeError, TypeError):
                    pass  # 不是 JSON，使用原始数据
        
        # 更新 nfc_uid 和 dev_ts_ms（可能在解析后更新）
        nfc_uid = message_data.get("nfc_uid", nfc_uid)
        dev_ts_ms = message_data.get("dev_ts_ms", dev_ts_ms)
        
        # 从顶层获取 items 数组（新格式直接在顶层有 items）
        items = message_data.get("items", [])
        imu_item = None
        audio_meta_item = None
        
        for item in items:
            item_type = item.get("type", "")
            if item_type == "imu":
                imu_item = item
            elif item_type == "audio_meta":
                audio_meta_item = item
        
        # 初始化结果字典
        ai_result = {
            "nfc_uid": nfc_uid,
            "inference_timestamp": datetime.now().isoformat()
        }
        
        # 1. 处理 Emotion（从 audio_meta 中提取 sound_osslink）
        sound_osslink = ""
        if audio_meta_item:
            audio_v = audio_meta_item.get("v", {})
            sound_osslink = audio_v.get("sound_osslink", "")
        
        try:
            emotion_result = process_emotion(sound_osslink)
            ai_result["emotion_state"] = emotion_result.get("emotion", "calm")
            ai_result["emotion_score"] = emotion_result.get("score", 80)
            ai_result["emotion_message"] = f"宠物状态{emotion_result.get('emotion', 'calm')}，情绪评分{emotion_result.get('score', 80)}。"
        except Exception as e:
            logger.error(f"Emotion 处理失败 - NFC UID: {nfc_uid}, Error: {e}")
            ai_result["emotion_state"] = "error"
            ai_result["emotion_score"] = 0
            ai_result["emotion_message"] = f"Emotion 处理失败: {str(e)}"
        
        # 2. 处理 IMU 数据（Action 推理）
        if not imu_item:
            logger.warning(f"未找到 IMU 数据 - NFC UID: {nfc_uid}")
            ai_result["action"] = "unknown"
            ai_result["action_confidence"] = 0.0
            return ai_result
        
        # 提取 IMU samples
        imu_v = imu_item.get("v", {})
        imu_samples = imu_v.get("samples", [])
        imu_timestamp = imu_v.get("timestamp", dev_ts_ms)  # 使用 IMU 项的 timestamp 或设备时间戳
        
        if not imu_samples:
            logger.warning(f"IMU samples 为空 - NFC UID: {nfc_uid}")
            ai_result["action"] = "unknown"
            ai_result["action_confidence"] = 0.0
            return ai_result
        
        logger.info(f"提取到 IMU 数据 - NFC UID: {nfc_uid}, Samples 数量: {len(imu_samples)}")
        
        # 2.1 将IMU数据转换为虚拟坐标系特征
        try:
            # 传递基准时间戳（毫秒）用于新格式的时间戳计算
            features_df = convert_imu_to_virtual_features(imu_samples, base_timestamp_ms=imu_timestamp)
        except Exception as e:
            logger.error(f"IMU数据转换失败 - NFC UID: {nfc_uid}, Error: {e}")
            ai_result["action"] = "error"
            ai_result["action_confidence"] = 0.0
            ai_result["inference_error"] = str(e)
            return ai_result
        
        # 2.2 保存为临时CSV文件
        temp_csv_path = TEMP_DIR / f"{nfc_uid}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        try:
            features_df.to_csv(temp_csv_path, index=False)
            logger.debug(f"特征CSV已保存: {temp_csv_path}")
        except Exception as e:
            logger.error(f"保存CSV文件失败 - NFC UID: {nfc_uid}, Error: {e}")
            ai_result["action"] = "error"
            ai_result["action_confidence"] = 0.0
            ai_result["inference_error"] = f"CSV保存失败: {str(e)}"
            return ai_result
        
        # 2.3 执行推理
        try:
            inference_result = perform_inference_from_csv(temp_csv_path)
            ai_result["action"] = inference_result.get("action", "unknown")
            ai_result["action_confidence"] = inference_result.get("confidence", 0.0)
            
            if inference_result.get("status") == "error":
                ai_result["inference_error"] = inference_result.get("error", "Unknown error")
        except Exception as e:
            logger.error(f"推理失败 - NFC UID: {nfc_uid}, Error: {e}")
            ai_result["action"] = "error"
            ai_result["action_confidence"] = 0.0
            ai_result["inference_error"] = str(e)
        finally:
            # 清理临时文件
            try:
                if temp_csv_path.exists():
                    temp_csv_path.unlink()
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")
        
        logger.info(f"处理完成 - NFC UID: {nfc_uid}, Action: {ai_result.get('action')}, Confidence: {ai_result.get('action_confidence', 0):.2%}, Emotion: {ai_result.get('emotion_state')}")
        return ai_result
        
    except Exception as e:
        logger.error(f"处理 Stream 消息失败: {e}", exc_info=True)
        return {
            "nfc_uid": message_data.get("nfc_uid", "unknown"),
            "action": "error",
            "action_confidence": 0.0,
            "emotion_state": "error",
            "emotion_score": 0,
            "emotion_message": f"处理失败: {str(e)}",
            "inference_timestamp": datetime.now().isoformat(),
            "inference_error": str(e)
        }

# --- 11. Redis Stream 消费者主循环 ---
def redis_stream_consumer_loop():
    """
    Redis Stream 消费者主循环，使用 xreadgroup 阻塞式读取消息
    具有断线重连机制和消息确认机制
    """
    logger.info("启动 Redis Stream 消费者...")
    logger.info(f"监听 Stream: {REDIS_STREAM_TELEMETRY}")
    logger.info(f"Consumer Group: {REDIS_STREAM_CONSUMER_GROUP}")
    logger.info(f"Consumer Name: {REDIS_STREAM_CONSUMER_NAME}")
    
    redis_client = None
    
    while True:
        try:
            # 创建或重新创建 Redis 客户端
            if redis_client is None:
                redis_client = create_redis_client()
                if redis_client is None:
                    logger.warning(f"Redis 连接失败，{REDIS_RECONNECT_DELAY} 秒后重试...")
                    time.sleep(REDIS_RECONNECT_DELAY)
                    continue
                
                # 确保 Consumer Group 存在
                try:
                    group_created = ensure_consumer_group(redis_client, REDIS_STREAM_TELEMETRY, REDIS_STREAM_CONSUMER_GROUP)
                    
                    # 如果 Consumer Group 是新创建的，使用 $ 已经定位到最新位置
                    # 后续使用 > 将只读取创建后的新消息
                    if group_created:
                        logger.info("Consumer Group 为新创建，已定位到最新消息位置，将只读取新消息")
                    
                except Exception as e:
                    logger.error(f"创建 Consumer Group 失败: {e}，{REDIS_RECONNECT_DELAY} 秒后重试...")
                    redis_client = None
                    time.sleep(REDIS_RECONNECT_DELAY)
                    continue
            
            # 使用 xreadgroup 阻塞式读取消息
            # > 表示只读取未确认的新消息（创建 Consumer Group 后的新消息）
            logger.debug(f"等待 Stream 消息: {REDIS_STREAM_TELEMETRY}")
            messages = redis_client.xreadgroup(
                groupname=REDIS_STREAM_CONSUMER_GROUP,
                consumername=REDIS_STREAM_CONSUMER_NAME,
                streams={REDIS_STREAM_TELEMETRY: ">"},  # > 表示只读取新消息
                count=1,  # 每次读取1条消息
                block=0  # 阻塞等待，0表示无限阻塞
            )
            
            if not messages:
                # 没有消息，继续循环
                continue
            
            # xreadgroup 返回格式: [(stream_name, [(message_id, {field: value, ...}), ...]), ...]
            for stream_name, stream_messages in messages:
                for message_id, message_data in stream_messages:
                    logger.info(f"收到 Stream 消息 - Stream: {stream_name}, Message ID: {message_id}")
                    
                    # 解析消息数据（根据 REDIS_DECODE 配置，可能是 bytes 或字符串）
                    parsed_data = {}
                    for key, value in message_data.items():
                        if isinstance(key, bytes):
                            key = key.decode('utf-8')
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        parsed_data[key] = value
                    
                    # 处理消息
                    # 如果消息只有一个字段（通常是 "payload" 或 "data"），尝试解析为 JSON
                    try:
                        # 检查是否是单个 JSON 字符串字段
                        if len(parsed_data) == 1:
                            single_key = list(parsed_data.keys())[0]
                            single_value = parsed_data[single_key]
                            # 尝试解析为 JSON
                            try:
                                parsed_json = json.loads(single_value) if isinstance(single_value, str) else single_value
                                if isinstance(parsed_json, dict):
                                    parsed_data = parsed_json
                                    logger.debug(f"从单个字段解析 JSON 消息: {single_key}")
                            except (json.JSONDecodeError, TypeError):
                                pass  # 不是 JSON，使用原始数据
                        
                        ai_result = process_stream_message(parsed_data)
                        nfc_uid = ai_result.get("nfc_uid", "unknown")
                        
                        # 使用 update_latest_field 逐个更新 Redis Hash 字段
                        update_success = True
                        if "emotion_state" in ai_result:
                            if not update_latest_field(redis_client, nfc_uid, "emotion_state", ai_result["emotion_state"]):
                                update_success = False
                        if "emotion_score" in ai_result:
                            if not update_latest_field(redis_client, nfc_uid, "emotion_score", str(ai_result["emotion_score"])):
                                update_success = False
                        if "emotion_message" in ai_result:
                            if not update_latest_field(redis_client, nfc_uid, "emotion_message", ai_result["emotion_message"]):
                                update_success = False
                        if "action" in ai_result:
                            if not update_latest_field(redis_client, nfc_uid, "action", ai_result["action"]):
                                update_success = False
                        if "action_confidence" in ai_result:
                            if not update_latest_field(redis_client, nfc_uid, "action_confidence", str(ai_result["action_confidence"])):
                                update_success = False
                        if "inference_timestamp" in ai_result:
                            if not update_latest_field(redis_client, nfc_uid, "inference_timestamp", ai_result["inference_timestamp"]):
                                update_success = False
                        
                        # 发布到 Pub/Sub（通知机制）
                        publish_success = publish_ai_result(redis_client, nfc_uid, ai_result)
                        
                        # 确认消息（XACK）- 只要字段更新成功就确认（Pub/Sub 失败不影响）
                        if update_success:
                            redis_client.xack(
                                REDIS_STREAM_TELEMETRY,
                                REDIS_STREAM_CONSUMER_GROUP,
                                message_id
                            )
                            logger.info(f"消息已确认 - Message ID: {message_id}, NFC UID: {nfc_uid}, 字段更新: {update_success}, Pub/Sub: {publish_success}")
                        else:
                            logger.warning(f"字段更新失败，消息未确认 - Message ID: {message_id}, NFC UID: {nfc_uid}")
                            if not publish_success:
                                logger.warning(f"Pub/Sub 发布也失败 - NFC UID: {nfc_uid}")
                            
                    except Exception as e:
                        logger.error(f"处理 Stream 消息时发生异常: {e}", exc_info=True)
                        # 处理失败时不确认消息，让消息保留在 Pending 列表中
                        # 可以根据需要实现重试机制或死信队列
                        nfc_uid = parsed_data.get("nfc_uid", "unknown")
                        logger.error(f"消息处理失败，未确认 - Message ID: {message_id}, NFC UID: {nfc_uid}, Error: {e}")
        
        except redis.ConnectionError as e:
            logger.error(f"Redis 连接错误: {e}，尝试重连...")
            redis_client = None
            time.sleep(REDIS_RECONNECT_DELAY)
        
        except redis.TimeoutError as e:
            logger.error(f"Redis 超时错误: {e}，尝试重连...")
            redis_client = None
            time.sleep(REDIS_RECONNECT_DELAY)
        
        except Exception as e:
            logger.error(f"Redis Stream 消费者发生未知错误: {e}", exc_info=True)
            redis_client = None
            time.sleep(REDIS_RECONNECT_DELAY)

# --- 12. 主程序入口 ---
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AI 模型服务 - Redis Stream 消费者模式")
    logger.info("=" * 60)
    logger.info(f"环境: {ENV}")
    logger.info(f"模型设备: {DEVICE}")
    logger.info(f"已加载模型数量: {len(models)}")
    logger.info(f"Redis 服务器: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Redis Stream: {REDIS_STREAM_TELEMETRY}")
    logger.info(f"Consumer Group: {REDIS_STREAM_CONSUMER_GROUP}")
    logger.info(f"Consumer Name: {REDIS_STREAM_CONSUMER_NAME}")
    logger.info(f"Redis Key 前缀: {REDIS_KEY_PREFIX}")
    logger.info("=" * 60)
    
    # 启动 Stream 消费者循环
    try:
        redis_stream_consumer_loop()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"服务异常退出: {e}", exc_info=True)
    finally:
        logger.info("AI 模型服务已关闭")