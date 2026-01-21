# AI 模型服务部署指南

## 快速开始

### 1. 上传文件到服务器

将以下文件上传到服务器（建议路径：`/opt/ai_service`）：
- `ai_service.py`
- `requirements.txt`
- `artifacts/` 目录（包含所有模型文件）
- `start.sh`
- `deploy.sh`
- `ai-service.service`
- `.env.example`（可选）

### 2. 运行部署脚本

```bash
cd /opt/ai_service
chmod +x deploy.sh
./deploy.sh
```

### 3. 配置环境变量（可选）

如果需要自定义配置，可以创建 `.env` 文件：

```bash
cp .env.example .env
# 编辑 .env 文件
nano .env
```

### 4. 启动服务

#### 方式一：直接运行（测试用）

```bash
./start.sh
```

#### 方式二：使用 systemd（生产环境推荐）

```bash
# 复制服务文件到 systemd 目录
sudo cp ai-service.service /etc/systemd/system/

# 修改服务文件中的路径和用户（如需要）
sudo nano /etc/systemd/system/ai-service.service

# 重新加载 systemd
sudo systemctl daemon-reload

# 启用并启动服务
sudo systemctl enable ai-service
sudo systemctl start ai-service

# 查看服务状态
sudo systemctl status ai-service

# 查看日志
sudo journalctl -u ai-service -f
```

#### 方式三：使用 screen（简单后台运行）

```bash
# 安装 screen（如果未安装）
sudo apt-get install screen

# 启动 screen 会话
screen -S ai_service

# 在 screen 中运行
./start.sh

# 按 Ctrl+A 然后按 D 退出 screen（服务继续运行）

# 重新连接
screen -r ai_service
```

## 常用命令

### systemd 管理命令

```bash
# 启动服务
sudo systemctl start ai-service

# 停止服务
sudo systemctl stop ai-service

# 重启服务
sudo systemctl restart ai-service

# 查看状态
sudo systemctl status ai-service

# 查看日志
sudo journalctl -u ai-service -f

# 查看最近 100 行日志
sudo journalctl -u ai-service -n 100

# 禁用开机自启
sudo systemctl disable ai-service

# 启用开机自启
sudo systemctl enable ai-service
```

## 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| ENV | prod | 环境标识：dev/test/prod |
| REDIS_HOST | 47.97.19.58 | Redis 服务器地址 |
| REDIS_PORT | 6379 | Redis 端口 |
| REDIS_DB | 0 | Redis 数据库编号 |
| REDIS_PASSWORD | petpypkj2025 | Redis 密码 |
| REDIS_DECODE | False | 是否自动解码（Stream 建议 False） |
| ARTIFACTS_DIR | ./artifacts | 模型文件目录 |
| TEMP_DIR | ./temp | 临时文件目录 |
| LOG_LEVEL | INFO | 日志级别 |

## 文件结构

```
/opt/ai_service/
├── ai_service.py          # 主程序
├── requirements.txt       # Python 依赖
├── start.sh              # 启动脚本
├── deploy.sh             # 部署脚本
├── ai-service.service    # systemd 服务文件
├── .env.example          # 环境变量示例
├── venv/                 # Python 虚拟环境（部署后生成）
├── artifacts/            # 模型文件目录
│   ├── best_model_fold0.pth
│   ├── best_model_fold1.pth
│   ├── best_model_fold2.pth
│   ├── best_model_fold3.pth
│   ├── best_model_fold4.pth
│   ├── label_map.pkl
│   └── scaler_fold0.pkl
├── temp/                 # 临时文件目录
└── logs/                 # 日志目录（如果配置了文件日志）
```

## 故障排查

### 1. 服务无法启动

```bash
# 检查 Python 环境
python3 --version

# 检查依赖是否安装
source venv/bin/activate
pip list

# 检查模型文件
ls -la artifacts/
```

### 2. Redis 连接失败

```bash
# 测试 Redis 连接
redis-cli -h 47.97.19.58 -p 6379 -a petpypkj2025 ping

# 检查防火墙
sudo ufw status
```

### 3. 查看详细日志

```bash
# systemd 日志
sudo journalctl -u ai-service -f

# 如果使用直接运行，日志会输出到控制台
```

## 注意事项

1. **模型文件**：确保 `artifacts/` 目录包含所有必需的模型文件
2. **权限**：确保服务有权限读取模型文件和写入临时文件
3. **资源**：根据实际情况调整 systemd 服务文件中的资源限制
4. **GPU**：如果使用 GPU，确保 CUDA 和 PyTorch GPU 版本正确安装
5. **网络**：确保服务器可以访问 Redis 服务器

