#!/bin/bash

# Modern BERT Voice Speech Maker - Docker Volume Setup
# このスクリプトは学習用の永続ボリュームを事前に作成します

set -e

echo "=== Modern BERT Voice Speech Maker - Docker Volume Setup ==="

# ボリューム名を定義
DATA_VOLUME="voice_speech_maker_data"
MODEL_VOLUME="voice_speech_maker_models"
LOGS_VOLUME="voice_speech_maker_logs"

# 既存のボリュームをチェック
echo "Checking existing volumes..."

if docker volume ls | grep -q "$DATA_VOLUME"; then
    echo "⚠️  Volume '$DATA_VOLUME' already exists"
else
    echo "Creating volume '$DATA_VOLUME'..."
    docker volume create "$DATA_VOLUME"
    echo "✅ Volume '$DATA_VOLUME' created"
fi

if docker volume ls | grep -q "$MODEL_VOLUME"; then
    echo "⚠️  Volume '$MODEL_VOLUME' already exists"
else
    echo "Creating volume '$MODEL_VOLUME'..."
    docker volume create "$MODEL_VOLUME"
    echo "✅ Volume '$MODEL_VOLUME' created"
fi

if docker volume ls | grep -q "$LOGS_VOLUME"; then
    echo "⚠️  Volume '$LOGS_VOLUME' already exists"
else
    echo "Creating volume '$LOGS_VOLUME'..."
    docker volume create "$LOGS_VOLUME"
    echo "✅ Volume '$LOGS_VOLUME' created"
fi

echo ""
echo "=== Volume Information ==="
docker volume ls | grep voice_speech_maker

echo ""
echo "=== Volume Details ==="
echo "📁 Data Volume: $DATA_VOLUME"
docker volume inspect "$DATA_VOLUME" | grep -E '"Mountpoint"|"CreatedAt"'

echo ""
echo "📁 Model Volume: $MODEL_VOLUME"
docker volume inspect "$MODEL_VOLUME" | grep -E '"Mountpoint"|"CreatedAt"'

echo ""
echo "📁 Logs Volume: $LOGS_VOLUME"
docker volume inspect "$LOGS_VOLUME" | grep -E '"Mountpoint"|"CreatedAt"'

echo ""
echo "=== Usage Instructions ==="
echo "🔧 To mount these volumes in your containers:"
echo "   -v $DATA_VOLUME:/workspace/Data"
echo "   -v $MODEL_VOLUME:/workspace/model_assets"
echo "   -v $LOGS_VOLUME:/workspace/logs"
echo ""
echo "🧹 To clean up all volumes:"
echo "   docker volume rm $DATA_VOLUME $MODEL_VOLUME $LOGS_VOLUME"
echo ""
echo "✅ Volume setup complete!"
