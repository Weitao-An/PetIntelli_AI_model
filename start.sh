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
export ENV=${ENV:-prod}
export REDIS_HOST=${REDIS_HOST:-47.97.19.58}
export REDIS_PORT=${REDIS_PORT:-6379}
export REDIS_DB=${REDIS_DB:-0}
export REDIS_PASSWORD=${REDIS_PASSWORD:-petpypkj2025}
export REDIS_DECODE=${REDIS_DECODE:-False}
export ARTIFACTS_DIR=${ARTIFACTS_DIR:-$SCRIPT_DIR/artifacts}
export TEMP_DIR=${TEMP_DIR:-$SCRIPT_DIR/temp}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# 创建必要的目录
mkdir -p "$TEMP_DIR"
mkdir -p "$ARTIFACTS_DIR"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查模型文件
if [ ! -d "$ARTIFACTS_DIR" ] || [ -z "$(ls -A $ARTIFACTS_DIR/*.pth 2>/dev/null)" ]; then
    echo "警告: 未找到模型文件，请确保 artifacts 目录包含模型文件"
fi

# 启动服务
echo "=========================================="
echo "启动 AI 模型服务..."
echo "环境: $ENV"
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo "=========================================="
python3 ai_service.py

