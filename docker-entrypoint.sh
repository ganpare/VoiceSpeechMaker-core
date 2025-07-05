#!/bin/bash
set -e

echo "=== Modern BERT Voice Speech Maker Training Container ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# GPU情報を表示
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU Information ==="
    nvidia-smi
fi

# バッチ学習用の初期化

# 引数が与えられた場合はそれを実行、そうでなければデフォルトコマンド
if [ $# -eq 0 ]; then
    echo "No command provided, starting default training..."
    exec python train_ms_jp_extra.py
else
    echo "Executing: $@"
    exec "$@"
fi
