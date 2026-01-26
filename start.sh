#!/bin/bash
# AI 模型服务启动脚本
# 使用方法: ./start.sh

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 设置环境变量（如果 .env 文件存在则加载）
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 设置默认环境变量
export ENV=${ENV:-dev}
export REDIS_HOST=${REDIS_HOST:-47.97.19.58}
export REDIS_PORT=${REDIS_PORT:-6379}
export REDIS_DB=${REDIS_DB:-0}
export REDIS_PASSWORD=${REDIS_PASSWORD:-petpypkj2025}
export REDIS_DECODE=${REDIS_DECODE:-False}
export TEMP_DIR=${TEMP_DIR:-$SCRIPT_DIR/temp}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# 模型目录配置（双模型架构）
# Kelong模型目录（用于判断快速奔跑/缓慢走动）
export KELONG_ARTIFACTS_DIR=${KELONG_ARTIFACTS_DIR:-$SCRIPT_DIR/kelong_artifacts}
# Processed模型目录（用于判断安静休息/吃喝护理）
export PROCESSED_MODELS_DIR=${PROCESSED_MODELS_DIR:-$SCRIPT_DIR/processed_models}

# 创建必要的目录
mkdir -p "$TEMP_DIR"
mkdir -p "$KELONG_ARTIFACTS_DIR"
mkdir -p "$PROCESSED_MODELS_DIR"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查模型文件
KELONG_MODEL_FOUND=false
PROCESSED_MODEL_FOUND=false

# 检查Kelong模型文件
if [ -d "$KELONG_ARTIFACTS_DIR" ]; then
    if [ -f "$KELONG_ARTIFACTS_DIR/label_map.pkl" ] && \
       [ -f "$KELONG_ARTIFACTS_DIR/scaler_fold0.pkl" ] && \
       [ -f "$KELONG_ARTIFACTS_DIR/best_model_fold0.pth" ]; then
        KELONG_MODEL_FOUND=true
        echo "✓ Kelong模型文件已找到: $KELONG_ARTIFACTS_DIR"
    else
        echo "警告: Kelong模型文件不完整，请确保以下文件存在："
        echo "  - $KELONG_ARTIFACTS_DIR/label_map.pkl"
        echo "  - $KELONG_ARTIFACTS_DIR/scaler_fold0.pkl"
        echo "  - $KELONG_ARTIFACTS_DIR/best_model_fold0.pth"
    fi
else
    echo "警告: Kelong模型目录不存在: $KELONG_ARTIFACTS_DIR"
fi

# 检查Processed模型文件
if [ -d "$PROCESSED_MODELS_DIR" ]; then
    if [ -f "$PROCESSED_MODELS_DIR/processed_2class_model.pkl" ] && \
       [ -f "$PROCESSED_MODELS_DIR/processed_2class_scaler.pkl" ] && \
       [ -f "$PROCESSED_MODELS_DIR/processed_2class_selector.pkl" ] && \
       [ -f "$PROCESSED_MODELS_DIR/processed_2class_le.pkl" ] && \
       [ -f "$PROCESSED_MODELS_DIR/processed_2class_cols.pkl" ]; then
        PROCESSED_MODEL_FOUND=true
        echo "✓ Processed模型文件已找到: $PROCESSED_MODELS_DIR"
    else
        echo "警告: Processed模型文件不完整，请确保以下文件存在："
        echo "  - $PROCESSED_MODELS_DIR/processed_2class_model.pkl"
        echo "  - $PROCESSED_MODELS_DIR/processed_2class_scaler.pkl"
        echo "  - $PROCESSED_MODELS_DIR/processed_2class_selector.pkl"
        echo "  - $PROCESSED_MODELS_DIR/processed_2class_le.pkl"
        echo "  - $PROCESSED_MODELS_DIR/processed_2class_cols.pkl"
    fi
else
    echo "警告: Processed模型目录不存在: $PROCESSED_MODELS_DIR"
fi

# 检查至少有一个模型可用
if [ "$KELONG_MODEL_FOUND" = false ] && [ "$PROCESSED_MODEL_FOUND" = false ]; then
    echo "错误: 未找到任何模型文件，请检查模型目录配置"
    exit 1
fi

# 启动服务
echo "=========================================="
echo "启动 AI 模型服务（双模型架构）..."
echo "环境: $ENV"
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo "Kelong模型目录: $KELONG_ARTIFACTS_DIR"
echo "Processed模型目录: $PROCESSED_MODELS_DIR"
echo "=========================================="
python3 ai_service.py

