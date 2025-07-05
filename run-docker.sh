#!/bin/bash

# Modern BERT Voice Speech Maker Docker 学習起動スクリプト

set -e

echo "=== Modern BERT Voice Speech Maker Docker Training Setup ==="

# 必要なディレクトリを作成
echo "Creating necessary directories..."
mkdir -p Data model_assets configs logs

# GPU確認
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Docker Compose確認
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed."
    exit 1
fi

# 引数に応じて実行モードを選択
case "${1:-run}" in
    "build")
        echo "Building Docker image..."
        docker-compose build --no-cache
        ;;
    "run")
        echo "Starting training..."
        docker-compose up
        ;;
    "shell")
        echo "Starting interactive shell..."
        docker-compose run --rm voice-speech-maker-train bash
        ;;
    "logs")
        echo "Showing logs..."
        docker-compose logs -f
        ;;
    "stop")
        echo "Stopping containers..."
        docker-compose down
        ;;
    "clean")
        echo "Cleaning up..."
        docker-compose down --rmi all --volumes
        ;;
    *)
        echo "Usage: $0 [build|run|shell|logs|stop|clean]"
        echo ""
        echo "Commands:"
        echo "  build  - Build Docker image"
        echo "  run    - Start training (default)"
        echo "  shell  - Start interactive shell"
        echo "  logs   - Show logs"
        echo "  stop   - Stop containers"
        echo "  clean  - Clean up everything"
        exit 1
        ;;
esac

echo "Done!"
