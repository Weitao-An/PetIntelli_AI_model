#!/bin/bash
# 部署脚本 - 用于快速部署 AI 模型服务
# 使用方法: ./deploy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "AI 模型服务部署脚本"
echo "=========================================="

# 1. 检查 Python
echo "[1/5] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi
echo "✓ Python 版本: $(python3 --version)"

# 2. 创建虚拟环境
echo "[2/5] 创建/更新虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ 虚拟环境已创建"
else
    echo "✓ 虚拟环境已存在"
fi

# 3. 安装依赖
echo "[3/5] 安装 Python 依赖..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
echo "✓ 依赖安装完成"

# 4. 创建必要目录
echo "[4/5] 创建必要目录..."
mkdir -p artifacts
mkdir -p temp
mkdir -p logs
echo "✓ 目录创建完成"

# 5. 检查模型文件
echo "[5/5] 检查模型文件..."
if [ -z "$(ls -A artifacts/*.pth 2>/dev/null)" ]; then
    echo "⚠ 警告: 未找到模型文件 (*.pth)"
    echo "   请确保 artifacts 目录包含以下文件:"
    echo "   - best_model_fold0.pth"
    echo "   - best_model_fold1.pth"
    echo "   - best_model_fold2.pth"
    echo "   - best_model_fold3.pth"
    echo "   - best_model_fold4.pth"
    echo "   - label_map.pkl"
    echo "   - scaler_fold0.pkl"
else
    echo "✓ 模型文件检查通过"
fi

# 6. 设置脚本权限
chmod +x start.sh
chmod +x deploy.sh

echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 检查并修改 .env 文件（如果使用）"
echo "2. 确保 artifacts 目录包含所有模型文件"
echo "3. 运行启动命令（见下方）"
echo ""

