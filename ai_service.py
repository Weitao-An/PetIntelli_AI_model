import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import asyncio
import logging
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional, Dict
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import redis
from datetime import datetime
from pathlib import Path
import os
import tempfile
import subprocess

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="IMU 动态形状推理服务")

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

# Kafka 配置
# 注意：如果 Kafka 在 Docker 容器中运行，且配置了 EXTERNAL 监听器（9094），应使用 9094 端口
# 如果使用 INTERNAL 监听器（9092），需要确保 advertised address 正确配置
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "imu_data_topic")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "ai_model_service_group")

# Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

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
def convert_imu_to_virtual_features(imu_data: List[Dict]) -> pd.DataFrame:
    """
    将原始IMU数据转换为虚拟坐标系特征
    
    Args:
        imu_data: IMU数据列表，每个元素包含 sequence, timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        
    Returns:
        包含虚拟坐标系特征的DataFrame
    """
    if not imu_data:
        raise ValueError("IMU数据为空")
    
    # 转换为DataFrame
    df = pd.DataFrame(imu_data)
    
    # 确保必要的列存在
    required_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 转换为数值类型
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 按timestamp排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 计算时间间隔（用于导数计算）
    dt = df['timestamp'].diff().fillna(df['timestamp'].iloc[0] if len(df) > 0 else 0.04)
    dt = dt.replace(0, 0.04)  # 默认50Hz采样率
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

# --- 6. 初始化 Redis 客户端 ---
redis_client = None
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=5
    )
    # 测试连接
    redis_client.ping()
    logger.info(f"成功连接到 Redis: {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.warning(f"Redis 连接失败: {e}，将无法写入Redis，但服务将继续运行")
    redis_client = None

# --- 7. 将结果写入 Redis ---
def write_to_redis(key: str, data: dict):
    """
    将数据写入 Redis
    
    Args:
        key: Redis键
        data: 要写入的数据字典
    """
    if redis_client is None:
        logger.warning(f"Redis未连接，跳过写入 - Key: {key}")
        return
    
    try:
        # 将结果序列化为JSON字符串
        result_json = json.dumps(data, ensure_ascii=False)
        
        # 写入Redis
        redis_client.set(key, result_json)
        
        # 设置过期时间（例如1小时）
        redis_client.expire(key, 3600)
        
        logger.info(f"结果已写入 Redis - Key: {key}")
        
    except Exception as e:
        logger.error(f"写入 Redis 失败 - Key: {key}, Error: {e}")

# --- 8. Kafka 消费者处理函数 ---
def process_kafka_message(message):
    """
    处理从Kafka接收到的消息
    
    期望的JSON格式（从Kafka）：
    {
        "ts_last": "1735123456789",
        "ts_rx": "1735123456789",
        "battery": 85,
        "online": 1,
        "gps_lat_e7": "350000123",
        "gps_lon_e7": "1390000456",
        "temp_x100": "2530",
        "heartRate": 95,
        "respiratoryRate": 22,
        "steps": 156,
        "calorie": 78,
        "imu_data": [
            {"sequence": 0, "timestamp": 0.0, "acc_x": ..., "acc_y": ..., "acc_z": ..., "gyro_x": ..., "gyro_y": ..., "gyro_z": ...},
            ...
        ],
        "emotion_sound_osslink": "xxxxxxxxxxxx",  // 暂时忽略
        "isOutdoor": 1,
        "endurance": 1224,
        "lastCharge": "2024/12/20 14:30",
        "uploadMode": 1,
        "isBuzzing": 0,
        "isVibrating": 0,
        "isFlashing": 0,
        "last_event_code": 5,
        "last_event_ts_ms": "1735123456788",
        ...其他字段...
    }
    
    处理后发送到Redis的JSON格式：
    {
        // 保留所有原始字段（除了imu_data和emotion_sound_osslink）
        "ts_last": "1735123456789",
        "ts_rx": "1735123456789",
        "battery": 85,
        ...所有其他原始字段...
        "action": "缓慢走动",  // 新增：从imu_data处理得到
        "action_confidence": 0.6935,  // 新增：置信度
        "inference_timestamp": "2024-01-01T00:00:00"  // 新增：推理时间戳
    }
    
    Args:
        message: Kafka消息对象
    """
    try:
        # 解析JSON消息
        message_value = message.value.decode('utf-8') if isinstance(message.value, bytes) else message.value
        original_data = json.loads(message_value)
        
        # 提取设备ID（可能从不同字段获取）
        device_id = original_data.get("device_id") or original_data.get("deviceId") or "unknown"
        imu_data = original_data.get("imu_data", [])
        
        if not imu_data:
            logger.warning(f"收到空IMU数据 - Device ID: {device_id}")
            # 即使没有IMU数据，也保留其他字段并发送到Redis（action设为unknown）
            result_data = original_data.copy()
            result_data.pop("imu_data", None)  # 移除imu_data
            result_data.pop("emotion_sound_osslink", None)  # 移除emotion_sound_osslink
            result_data["action"] = "unknown"
            result_data["action_confidence"] = 0.0
            result_data["inference_timestamp"] = datetime.now().isoformat()
            
            # 写入Redis
            redis_key = f"device_data:{device_id}" if device_id != "unknown" else f"device_data:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            write_to_redis(redis_key, result_data)
            return
        
        logger.info(f"收到Kafka消息 - Device ID: {device_id}, IMU数据点数: {len(imu_data)}")
        
        # 1. 将IMU数据转换为虚拟坐标系特征
        try:
            features_df = convert_imu_to_virtual_features(imu_data)
        except Exception as e:
            logger.error(f"IMU数据转换失败 - Device ID: {device_id}, Error: {e}")
            # 转换失败时，仍然保留原始数据并发送到Redis（action设为error）
            result_data = original_data.copy()
            result_data.pop("imu_data", None)
            result_data.pop("emotion_sound_osslink", None)
            result_data["action"] = "error"
            result_data["action_confidence"] = 0.0
            result_data["inference_timestamp"] = datetime.now().isoformat()
            result_data["inference_error"] = str(e)
            redis_key = f"device_data:{device_id}" if device_id != "unknown" else f"device_data:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            write_to_redis(redis_key, result_data)
            return
        
        # 2. 保存为临时CSV文件
        temp_csv_path = TEMP_DIR / f"{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        try:
            features_df.to_csv(temp_csv_path, index=False)
            logger.info(f"特征CSV已保存: {temp_csv_path}")
        except Exception as e:
            logger.error(f"保存CSV文件失败 - Device ID: {device_id}, Error: {e}")
            # 保存失败时，仍然保留原始数据并发送到Redis
            result_data = original_data.copy()
            result_data.pop("imu_data", None)
            result_data.pop("emotion_sound_osslink", None)
            result_data["action"] = "error"
            result_data["action_confidence"] = 0.0
            result_data["inference_timestamp"] = datetime.now().isoformat()
            result_data["inference_error"] = f"CSV保存失败: {str(e)}"
            redis_key = f"device_data:{device_id}" if device_id != "unknown" else f"device_data:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            write_to_redis(redis_key, result_data)
            return
        
        # 3. 执行推理
        try:
            inference_result = perform_inference_from_csv(temp_csv_path)
        except Exception as e:
            logger.error(f"推理失败 - Device ID: {device_id}, Error: {e}")
            # 清理临时文件
            if temp_csv_path.exists():
                temp_csv_path.unlink()
            # 推理失败时，仍然保留原始数据并发送到Redis
            result_data = original_data.copy()
            result_data.pop("imu_data", None)
            result_data.pop("emotion_sound_osslink", None)
            result_data["action"] = "error"
            result_data["action_confidence"] = 0.0
            result_data["inference_timestamp"] = datetime.now().isoformat()
            result_data["inference_error"] = str(e)
            redis_key = f"device_data:{device_id}" if device_id != "unknown" else f"device_data:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            write_to_redis(redis_key, result_data)
            return
        
        # 4. 构建结果数据：保留所有原始字段，移除imu_data和emotion_sound_osslink，添加action字段
        result_data = original_data.copy()
        
        # 移除不需要的字段
        result_data.pop("imu_data", None)  # 移除原始IMU数据
        result_data.pop("emotion_sound_osslink", None)  # 移除emotion音频链接（暂时不考虑emotion模型）
        
        # 添加推理结果
        result_data["action"] = inference_result.get("action", "unknown")
        result_data["action_confidence"] = inference_result.get("confidence", 0.0)
        result_data["inference_timestamp"] = datetime.now().isoformat()
        
        if inference_result.get("status") == "error":
            result_data["inference_error"] = inference_result.get("error", "Unknown error")
        
        # 5. 写入Redis
        # 使用device_id作为key的一部分，如果没有device_id则使用时间戳
        redis_key = f"device_data:{device_id}" if device_id != "unknown" else f"device_data:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        write_to_redis(redis_key, result_data)
        
        # 6. 清理临时文件
        try:
            if temp_csv_path.exists():
                temp_csv_path.unlink()
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
        
        logger.info(f"处理完成 - Device ID: {device_id}, Action: {result_data.get('action')}, Confidence: {result_data.get('action_confidence', 0):.2%}")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}, Message: {message.value}")
    except Exception as e:
        logger.error(f"处理Kafka消息失败: {e}", exc_info=True)

# --- 9. Kafka 消费者任务 ---
async def kafka_consumer_task():
    """
    异步Kafka消费者任务
    """
    consumer = None
    try:
        logger.info(f"启动Kafka消费者 - Topic: {KAFKA_TOPIC}, Servers: {KAFKA_BOOTSTRAP_SERVERS}")
        
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
            group_id=KAFKA_GROUP_ID,
            auto_offset_reset='latest',  # 从最新消息开始消费
            enable_auto_commit=True,
            value_deserializer=lambda m: m.decode('utf-8') if m else None,
            consumer_timeout_ms=1000  # 1秒超时，用于定期检查
        )
        
        while True:
            try:
                # 轮询消息
                message_pack = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        process_kafka_message(message)
                
                # 短暂休眠，避免CPU占用过高
                await asyncio.sleep(0.1)
                
            except KafkaError as e:
                logger.error(f"Kafka消费错误: {e}")
                await asyncio.sleep(5)  # 出错后等待5秒再重试
                
    except Exception as e:
        logger.error(f"Kafka消费者启动失败: {e}")
    finally:
        if consumer:
            consumer.close()
            logger.info("Kafka消费者已关闭")

# --- 10. 启动后台任务 ---
@app.on_event("startup")
async def startup_event():
    """应用启动时启动Kafka消费者"""
    asyncio.create_task(kafka_consumer_task())
    logger.info("AI模型服务已启动，Kafka消费者正在运行...")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    logger.info("正在关闭AI模型服务...")
    if redis_client is not None:
        redis_client.close()

# --- 11. 保留HTTP接口用于测试和监控 ---
@app.post("/predict")
async def predict(request: dict):
    """
    HTTP接口，用于测试和监控
    
    接受完整的JSON数据（与Kafka消息格式相同），处理imu_data并返回结果
    """
    try:
        # 提取设备ID和IMU数据
        device_id = request.get("device_id") or request.get("deviceId") or "unknown"
        imu_data = request.get("imu_data", [])
        
        if not imu_data:
            raise HTTPException(status_code=400, detail="缺少imu_data字段")
        
        # 1. 转换IMU数据
        features_df = convert_imu_to_virtual_features(imu_data)
        
        # 2. 保存为临时CSV
        temp_csv_path = TEMP_DIR / f"{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        features_df.to_csv(temp_csv_path, index=False)
        
        # 3. 执行推理
        inference_result = perform_inference_from_csv(temp_csv_path)
        
        # 4. 构建结果：保留所有原始字段，移除imu_data和emotion_sound_osslink，添加action
        result = request.copy()
        result.pop("imu_data", None)
        result.pop("emotion_sound_osslink", None)
        result["action"] = inference_result.get("action", "unknown")
        result["action_confidence"] = inference_result.get("confidence", 0.0)
        result["inference_timestamp"] = datetime.now().isoformat()
        
        if inference_result.get("status") == "error":
            result["inference_error"] = inference_result.get("error", "Unknown error")
            # 清理临时文件
            if temp_csv_path.exists():
                temp_csv_path.unlink()
            raise HTTPException(status_code=500, detail=result["inference_error"])
        
        # 5. 写入Redis
        redis_key = f"device_data:{device_id}" if device_id != "unknown" else f"device_data:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        write_to_redis(redis_key, result)
        
        # 6. 清理临时文件
        if temp_csv_path.exists():
            temp_csv_path.unlink()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HTTP接口处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口"""
    redis_connected = False
    if redis_client is not None:
        try:
            redis_client.ping()
            redis_connected = True
        except:
            pass
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "redis_connected": redis_connected,
        "device": str(DEVICE)
    }

@app.get("/redis/{device_id}")
async def get_redis_result(device_id: str):
    """从Redis获取指定设备的最新结果"""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis未连接")
    
    try:
        redis_key = f"device_data:{device_id}"
        result_json = redis_client.get(redis_key)
        
        if result_json:
            return json.loads(result_json)
        else:
            raise HTTPException(status_code=404, detail=f"未找到设备 {device_id} 的结果")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取Redis数据失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("启动FastAPI服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)