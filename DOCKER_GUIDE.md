# LoRA Easy Training Scripts Backend - Docker 使用指南

## 概述

本專案已配置為使用 **uv** 作為 Python 套件管理器的現代化 Docker 解決方案。這個配置提供了：

- ✅ 使用 uv 進行快速依賴項管理
- ✅ GPU 支援（CUDA 12.8）
- ✅ 多階段 Docker 建構優化
- ✅ 開發和生產環境分離
- ✅ 整合 TensorBoard 和 Jupyter Lab

## 系統需求

### 必要條件
- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA Docker Runtime（用於 GPU 支援）
- GNU Make（可選，用於便捷指令）

### 建議硬體
- NVIDIA GPU（支援 CUDA 12.8+）
- 至少 16GB RAM
- 50GB+ 可用儲存空間

## 快速開始

### 1. 環境設定

```bash
# 複製環境變數範例檔案
cp env.example .env

# 編輯環境變數（設定你的 API 金鑰）
nano .env
```

### 2. 建構和啟動

```bash
# 使用 Makefile（推薦）
make build
make up

# 或直接使用 Docker Compose
docker-compose build
docker-compose up -d
```

### 3. 檢查狀態

```bash
# 查看服務狀態
make health

# 查看日誌
make logs

# 進入容器 shell
make shell
```

## 詳細使用說明

### Makefile 指令

```bash
# 基本操作
make help        # 顯示所有可用指令
make build       # 建構 Docker 映像
make up          # 啟動服務
make down        # 停止服務
make restart     # 重啟服務
make logs        # 查看日誌
make shell       # 進入容器 shell

# 開發環境
make dev         # 啟動開發環境（包含 Jupyter + TensorBoard）
make prod        # 啟動生產環境

# 套件管理
make sync        # 同步依賴項
make update      # 更新依賴項
make lock        # 生成 uv.lock 檔案
make install PACKAGE=package_name  # 安裝新套件

# 程式碼品質
make test        # 執行測試
make format      # 格式化程式碼
make lint        # 檢查程式碼風格

# 監控和維護
make health      # 檢查容器健康狀態
make stats       # 顯示資源使用狀況
make gpu         # 顯示 GPU 使用狀況
make backup      # 備份資料卷
make clean       # 清理容器和映像
```

### 服務配置

#### 主要服務（lora-backend）
- **連接埠**: 8000（Web 介面）
- **GPU**: 支援所有可用 GPU
- **磁碟區**:
  - `./datasets` → `/dataset`（資料集）
  - `./workspace` → `/workspace`（工作空間）
  - `./models` → `/app/models`（模型檔案）
  - `./outputs` → `/app/outputs`（輸出結果）

#### 開發服務

**Jupyter Lab**:
```bash
make dev  # 啟動開發環境
# 訪問 http://localhost:8888
# 密碼: lora
```

**TensorBoard**:
```bash
make dev  # 啟動開發環境
# 訪問 http://localhost:6007
```

### 環境變數配置

編輯 `.env` 檔案設定以下變數：

```bash
# 使用者 ID（避免權限問題）
UID=1000

# API 金鑰
WANDB_API_KEY=your_wandb_api_key
NGROK_AUTH_TOKEN=your_ngrok_token
HF_TOKEN=your_huggingface_token

# GPU 配置
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
```

## 開發工作流程

### 1. 設定開發環境

```bash
# 啟動完整開發環境
make dev

# 進入容器進行開發
make shell

# 在容器內使用 uv
uv add new-package        # 添加新套件
uv remove old-package     # 移除套件
uv sync                   # 同步依賴項
uv run python script.py   # 執行腳本
```

### 2. 程式碼開發

```bash
# 格式化程式碼
make format

# 檢查程式碼風格
make lint

# 執行測試
make test

# 安裝新依賴
make install PACKAGE=torch-audio
```

### 3. 監控和除錯

```bash
# 實時查看日誌
make logs

# 檢查 GPU 使用狀況
make gpu

# 查看容器資源使用
make stats

# 檢查健康狀態
make health
```

## 生產部署

### 1. 優化配置

```bash
# 編輯 docker-compose.yml
# 註解掉開發用的磁碟區掛載
# volumes:
#   - .:/app:rw  # 註解掉這行

# 設定生產環境變數
DEBUG=false
WORKERS=4
```

### 2. 啟動生產服務

```bash
make prod
```

### 3. 負載平衡（可選）

使用 nginx 或其他反向代理：

```nginx
upstream lora_backend {
    server localhost:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://lora_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 疑難排解

### 常見問題

1. **權限問題**
   ```bash
   # 確保 UID 設定正確
   echo $UID
   # 修改 .env 檔案中的 UID
   ```

2. **GPU 不可用**
   ```bash
   # 檢查 NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi
   ```

3. **記憶體不足**
   ```bash
   # 增加 Docker 記憶體限制
   # 或減少批次大小
   BATCH_SIZE=1
   ```

4. **依賴項衝突**
   ```bash
   # 重新建構映像
   make clean
   make build
   ```

### 日誌分析

```bash
# 查看特定服務日誌
docker-compose logs lora-backend
docker-compose logs tensorboard
docker-compose logs jupyter

# 跟蹤實時日誌
docker-compose logs -f lora-backend
```

### 效能優化

1. **使用 pillow-simd**（已預設啟用於 x86_64）
2. **啟用記憶體預分配**
3. **使用 tcmalloc**（已預設啟用）
4. **調整批次大小和學習率**

## 安全注意事項

1. **不要在生產環境中暴露 Jupyter**
2. **定期更新基礎映像**
3. **使用 secrets 管理 API 金鑰**
4. **限制容器權限**

## 備份和恢復

```bash
# 建立備份
make backup

# 恢復備份（手動）
docker run --rm -v lora_datasets:/data -v $(PWD)/backups:/backup ubuntu \
  tar xzf /backup/datasets-YYYYMMDD-HHMMSS.tar.gz -C /data
```

## 更新和維護

```bash
# 更新 uv 和依賴項
make update

# 重新建構映像（包含系統更新）
make clean
make build

# 更新 uv.lock 檔案
make lock
```

## 效能監控

- **GPU 使用**: `make gpu`
- **系統資源**: `make stats`
- **應用程式日誌**: `make logs`
- **TensorBoard**: http://localhost:6007
- **健康檢查**: `make health`

## 支援和社群

如果遇到問題：
1. 檢查日誌：`make logs`
2. 查看健康狀態：`make health`
3. 參考疑難排解章節
4. 提交 Issue 到專案 GitHub

---

**注意**: 這個 Docker 配置針對 LoRA 訓練進行了優化，包含了所有必要的 ML 函式庫和 GPU 支援。首次建構可能需要較長時間來下載依賴項。