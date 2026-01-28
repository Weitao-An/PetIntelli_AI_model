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

# 模型目录配置（嵌套模型：四分类 + 二分类）
# 第一层：四分类模型路径
if [ -n "$FOUR_CLASS_MODEL_DIR" ]; then
    export FOUR_CLASS_MODEL_DIR="$FOUR_CLASS_MODEL_DIR"
elif [ -d "/home/Drame/Analysis/deployment" ]; then
    export FOUR_CLASS_MODEL_DIR="/home/Drame/Analysis/deployment"
elif [ -d "$SCRIPT_DIR/../deployment" ]; then
    export FOUR_CLASS_MODEL_DIR="$(cd "$SCRIPT_DIR/../deployment" && pwd)"
elif [ -d "$SCRIPT_DIR/deployment" ]; then
    export FOUR_CLASS_MODEL_DIR="$SCRIPT_DIR/deployment"
else
    export FOUR_CLASS_MODEL_DIR="/home/Drame/Analysis/deployment"
fi

# 第二层：二分类模型路径
if [ -n "$BINARY_MODEL_DIR" ]; then
    export BINARY_MODEL_DIR="$BINARY_MODEL_DIR"
elif [ -d "/home/yan/二分类" ]; then
    export BINARY_MODEL_DIR="/home/yan/二分类"
elif [ -d "$SCRIPT_DIR/../deployment" ]; then
    export BINARY_MODEL_DIR="$(cd "$SCRIPT_DIR/../deployment" && pwd)"
elif [ -d "$SCRIPT_DIR/deployment" ]; then
    export BINARY_MODEL_DIR="$SCRIPT_DIR/deployment"
else
    export BINARY_MODEL_DIR="/home/yan/二分类"
fi

# 创建必要的目录
mkdir -p "$TEMP_DIR"
mkdir -p "$FOUR_CLASS_MODEL_DIR"
mkdir -p "$BINARY_MODEL_DIR"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查模型文件
# 检查四分类模型文件
if [ -d "$FOUR_CLASS_MODEL_DIR" ]; then
    if [ -f "$FOUR_CLASS_MODEL_DIR/rf_model.pkl" ] && \
       [ -f "$FOUR_CLASS_MODEL_DIR/scaler.pkl" ] && \
       [ -f "$FOUR_CLASS_MODEL_DIR/model_metadata.json" ]; then
        echo "✓ 四分类模型文件已找到: $FOUR_CLASS_MODEL_DIR"
    else
        echo "错误: 四分类模型文件不完整，请确保以下文件存在："
        echo "  - $FOUR_CLASS_MODEL_DIR/rf_model.pkl"
        echo "  - $FOUR_CLASS_MODEL_DIR/scaler.pkl"
        echo "  - $FOUR_CLASS_MODEL_DIR/model_metadata.json"
        exit 1
    fi
else
    echo "错误: 四分类模型目录不存在: $FOUR_CLASS_MODEL_DIR"
    exit 1
fi

# 检查二分类模型文件（.joblib 或 .pkl）
if [ -d "$BINARY_MODEL_DIR" ]; then
    BINARY_MODEL_FILE=$(find "$BINARY_MODEL_DIR" -maxdepth 1 \( -name "*.joblib" -o -name "*.pkl" \) | head -n 1)
    if [ -n "$BINARY_MODEL_FILE" ] && [ -f "$BINARY_MODEL_FILE" ]; then
        echo "✓ 二分类模型文件已找到: $BINARY_MODEL_FILE"
    else
        echo "错误: 二分类模型文件不存在，请确保目录中存在 .joblib 或 .pkl 文件："
        echo "  - $BINARY_MODEL_DIR"
        exit 1
    fi
else
    echo "错误: 二分类模型目录不存在: $BINARY_MODEL_DIR"
    exit 1
fi

# 启动服务
echo "=========================================="
echo "启动 AI 模型服务（嵌套模型：四分类 + 二分类）..."
echo "环境: $ENV"
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo "第一层模型目录（四分类）: $FOUR_CLASS_MODEL_DIR"
echo "第二层模型目录（二分类）: $BINARY_MODEL_DIR"
echo "=========================================="
python3 ai_service.py
