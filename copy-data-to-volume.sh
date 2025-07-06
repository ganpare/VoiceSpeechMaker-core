#!/bin/bash

# データをボリュームにコピーするスクリプト

echo "=== データをボリュームにコピー ==="

# ボリュームが存在するか確認
if ! docker volume ls | grep -q voice_speech_maker_data; then
    echo "ボリューム voice_speech_maker_data が存在しません"
    echo "先にボリュームを作成してください: docker volume create voice_speech_maker_data"
    exit 1
fi

# 現在のDataディレクトリが存在するか確認
if [ ! -d "./Data" ]; then
    echo "Dataディレクトリが見つかりません"
    exit 1
fi

echo "ホストのData/をボリュームにコピー中..."

# 一時コンテナでファイルコピー
docker run --rm \
  -v "$(pwd)/Data:/host_data" \
  -v voice_speech_maker_data:/volume_data \
  alpine sh -c "cp -r /host_data/. /volume_data/ && echo 'コピー完了'"

echo ""
echo "=== ボリューム内容確認 ==="
docker run --rm \
  -v voice_speech_maker_data:/volume_data \
  alpine ls -la /volume_data/

echo ""
echo "=== データ転送完了 ==="