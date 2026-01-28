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

# 模型目录配置（四分类RandomForest模型）
# 优先级：1. 环境变量 2. 服务器路径 3. 父目录下的deployment文件夹 4. 当前目录下的deployment文件夹
if [ -n "$DEPLOYMENT_MODEL_DIR" ]; then
    export DEPLOYMENT_MODEL_DIR="$DEPLOYMENT_MODEL_DIR"
elif [ -d "/home/Drame/Analysis/20260128" ]; then
    export DEPLOYMENT_MODEL_DIR="/home/Drame/Analysis/20260128"
elif [ -d "$SCRIPT_DIR/../deployment" ]; then
    export DEPLOYMENT_MODEL_DIR="$(cd "$SCRIPT_DIR/../deployment" && pwd)"
elif [ -d "$SCRIPT_DIR/deployment" ]; then
    export DEPLOYMENT_MODEL_DIR="$SCRIPT_DIR/deployment"
else
    export DEPLOYMENT_MODEL_DIR="/home/Drame/Analysis/20260128"
fi

# 创建必要的目录
mkdir -p "$TEMP_DIR"
mkdir -p "$DEPLOYMENT_MODEL_DIR"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查模型文件
MODEL_FOUND=false

# 检查四分类模型文件
if [ -d "$DEPLOYMENT_MODEL_DIR" ]; then
    if [ -f "$DEPLOYMENT_MODEL_DIR/rf_model.pkl" ] && \
       [ -f "$DEPLOYMENT_MODEL_DIR/scaler.pkl" ] && \
       [ -f "$DEPLOYMENT_MODEL_DIR/model_metadata.json" ]; then
        MODEL_FOUND=true
        echo "✓ 四分类模型文件已找到: $DEPLOYMENT_MODEL_DIR"
    else
        echo "错误: 模型文件不完整，请确保以下文件存在："
        echo "  - $DEPLOYMENT_MODEL_DIR/rf_model.pkl"
        echo "  - $DEPLOYMENT_MODEL_DIR/scaler.pkl"
        echo "  - $DEPLOYMENT_MODEL_DIR/model_metadata.json"
        exit 1
    fi
else
    echo "错误: 模型目录不存在: $DEPLOYMENT_MODEL_DIR"
    exit 1
fi

# 启动服务
echo "=========================================="
echo "启动 AI 模型服务（四分类RandomForest模型）..."
echo "环境: $ENV"
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo "模型目录: $DEPLOYMENT_MODEL_DIR"
echo "=========================================="
python3 ai_service.py
