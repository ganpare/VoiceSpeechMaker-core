# Modern BERT Voice Speech Maker Docker Training

このプロジェクトでは、DockerとGitクローンを使用してModern BERT Voice Speech Makerの学習環境を再現可能な形で構築します。**Modal環境への移行準備として、バッチ処理での学習に特化しています。**

## 前提条件

- Docker & Docker Compose
- NVIDIA Container Toolkit (GPU使用時)
- Git

## セットアップ

### 1. 基本的な使用方法

```bash
# イメージをビルド
docker-compose build

# 学習を開始
docker-compose up
```

### 2. カスタム設定での使用

#### 特定のGitブランチを使用
```bash
# docker-compose.ymlの args セクションを編集
GIT_BRANCH: "your-branch-name"
```

#### 特定のコミットハッシュを使用
```bash
# docker-compose.ymlの args セクションを編集
GIT_COMMIT_HASH: "abc123def456"
```

### 3. データの準備

```bash
# データディレクトリを作成
mkdir -p Data/your_model_name
mkdir -p model_assets
mkdir -p configs

# データセットを配置
cp -r your_dataset/* Data/your_model_name/

# 設定ファイルを配置
cp your_config.json configs/config.json
```

### 4. インタラクティブモードでの実行

```bash
# コンテナ内でシェルを起動
docker-compose run --rm voice-speech-maker-train bash

# 個別のコマンドを実行
docker-compose run --rm voice-speech-maker-train python train_ms_jp_extra.py --help
```

### 5. バッチ学習の監視

学習ログの確認：
```bash
# ログをリアルタイム表示
docker-compose logs -f

# 永続ボリュームの確認
docker volume ls | grep voice_speech_maker
```

## ディレクトリ構造

```
.
├── Dockerfile.git           # Git clone版Dockerfile
├── docker-compose.yml       # Docker Compose設定
├── docker-entrypoint.sh     # エントリーポイントスクリプト
├── Data/                    # データセット用ディレクトリ
│   └── your_model_name/     # モデル固有のデータ
├── model_assets/            # 学習済みモデル出力
├── configs/                 # 設定ファイル
└── logs/                    # ログファイル
```

## トラブルシューティング

### GPU認識されない場合
```bash
# NVIDIA Container Toolkitの確認
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

### メモリ不足の場合
```bash
# Docker Composeでメモリ制限を設定
services:
  style-bert-vits2-train:
    mem_limit: 16g
    shm_size: 8g
```

### ポート競合の場合
```bash
# docker-compose.ymlのポート設定を変更
ports:
  - "6007:6006"  # TensorBoard
```

## Modal準備

この Docker 設定は Modal 環境への移行準備として設計されています：

1. **Git クローン方式**: Modal でも同様の方式を使用
2. **環境変数管理**: Modal の環境設定と互換
3. **ボリューム構造**: Modal Volume の設計参考
4. **再現性**: 同一環境の保証

## 次のステップ

1. Docker での学習動作確認
2. Modal 用コードの作成
3. Modal での学習実行
4. コスト最適化とスケーリング
