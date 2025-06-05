# 🚀 LoRA Easy Training Scripts Backend - 快速啟動

## 📋 前置需求

- Docker 和 Docker Compose
- NVIDIA GPU + NVIDIA Docker Runtime
- 至少 16GB RAM

## ⚡ 5 分鐘快速啟動

### 1️⃣ 複製環境設定
```bash
cp env.example .env
```

### 2️⃣ 編輯 API 金鑰（可選）
```bash
nano .env
# 設定 WANDB_API_KEY, HF_TOKEN 等
```

### 3️⃣ 建構和啟動
```bash
make build  # 建構 Docker 映像（首次需要較長時間）
make up     # 啟動服務
```

### 4️⃣ 檢查狀態
```bash
make health  # 檢查服務狀態
make logs    # 查看啟動日誌
```

## 🌐 存取服務

- **主應用程式**: http://localhost:8000
- **健康檢查**: http://localhost:8000/health

## 🛠️ 常用指令

```bash
# 基本操作
make help     # 顯示所有指令
make shell    # 進入容器
make logs     # 查看日誌
make restart  # 重啟服務
make down     # 停止服務

# 開發環境（包含 Jupyter + TensorBoard）
make dev      # 啟動開發環境
# Jupyter: http://localhost:8888 (密碼: lora)
# TensorBoard: http://localhost:6007

# 套件管理
make sync     # 同步依賴項
make install PACKAGE=新套件名稱
```

## 🐛 遇到問題？

1. **權限問題**: 檢查 `.env` 中的 `UID` 設定
2. **GPU 不可用**: 確認 NVIDIA Docker Runtime 已安裝
3. **記憶體不足**: 關閉其他應用程式或調整 Docker 記憶體限制
4. **建構失敗**: 運行 `make clean` 然後 `make build`

詳細說明請參考 [DOCKER_GUIDE.md](DOCKER_GUIDE.md)

## 🎯 接下來可以做什麼？

1. 上傳你的資料集到 `./datasets/` 目錄
2. 配置訓練參數
3. 開始訓練 LoRA 模型
4. 使用 TensorBoard 監控訓練進度
5. 在 Jupyter 中進行實驗和分析

---

**快速幫助**: `make help` 或查看 [完整文件](DOCKER_GUIDE.md)