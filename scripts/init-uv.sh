#!/bin/bash
# LoRA Easy Training Scripts Backend - UV Initialization Script

set -e

echo "🚀 初始化 uv 環境..."

# 檢查是否在 Docker 容器中
if [ -f /.dockerenv ]; then
    echo "✅ 在 Docker 容器中運行"
else
    echo "⚠️  建議在 Docker 容器中運行此腳本"
fi

# 檢查 uv 是否安裝
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安裝，請先安裝 uv"
    exit 1
fi

echo "📦 uv 版本: $(uv --version)"

# 建立虛擬環境（如果不存在）
if [ ! -d "/venv" ]; then
    echo "🔧 建立虛擬環境..."
    uv venv /venv
fi

# 啟用虛擬環境
source /venv/bin/activate

# 生成 lock 檔案
echo "🔒 生成 uv.lock 檔案..."
uv lock

# 安裝依賴項
echo "📥 安裝依賴項..."
uv sync

# 安裝本地套件
echo "🔧 安裝本地套件..."
uv pip install -e ./sd_scripts
uv pip install -e ./lycoris
uv pip install -e ./custom_scheduler

echo "✅ uv 環境初始化完成！"

# 顯示已安裝的套件
echo "📋 已安裝的套件數量: $(uv pip list | wc -l)"

# 驗證關鍵套件
echo "🔍 驗證關鍵套件..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "❌ PyTorch 未安裝"
python -c "import starlette; print('✅ Starlette 已安裝')" || echo "❌ Starlette 未安裝"
python -c "import uvicorn; print('✅ Uvicorn 已安裝')" || echo "❌ Uvicorn 未安裝"

echo "🎉 初始化完成！你現在可以使用 'make dev' 或 'make up' 啟動服務。"