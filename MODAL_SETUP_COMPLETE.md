# Modern BERT Voice Speech Maker - Modal対応Docker環境構築

## 概要

このプロジェクトは、Modern BERT-based音声合成システムの学習環境をModal対応のDockerコンテナとして構築したものです。

## プロジェクト目標

- ✅ Modal-compatible Docker環境の構築
- ✅ 永続ボリュームによるデータ管理  
- ✅ Git クローンベースの再現可能なデプロイ
- ✅ 学習専用（推論なし）の最適化環境
- ✅ esd.listファイルパス形式の確認と文書化

## アーキテクチャ

### ボリューム構成

```text
voice_speech_maker_data     → /workspace/Data (訓練データ)
voice_speech_maker_models   → /workspace/model_assets (モデル保存)
voice_speech_maker_logs     → /workspace/logs (ログ出力)
```

### ファイル構成

```text
voicespeechmaker_modal/
├── Dockerfile.git          # Gitクローンベースの学習用Docker
├── docker-compose.yml      # ボリューム管理とサービス定義
├── docker-entrypoint.sh    # コンテナ起動スクリプト
├── docker-volumes-setup.sh # ボリューム事前作成スクリプト
├── run-docker.sh          # Linux/Mac起動スクリプト
├── run-docker.bat         # Windows起動スクリプト
└── README.docker.md       # Docker環境詳細説明
```

## 実装済み機能

### 1. Dockerfile.git

- **ベース**: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
- **リポジトリ**: <https://github.com/ganpare/VoiceSpeechMaker-core.git>
- **特徴**:
  - Gitクローンによるコード取得（COPYではない）
  - GPU学習対応
  - 永続ボリューム対応
  - バッチ学習専用（GUI/ポート露出なし）

### 2. Docker Compose設定

```yaml
services:
  voice-speech-maker-train:
    build:
      context: .
      dockerfile: Dockerfile.git
    container_name: voice-speech-maker-train
    volumes:
      - voice_speech_maker_data:/workspace/Data
      - voice_speech_maker_models:/workspace/model_assets
      - voice_speech_maker_logs:/workspace/logs
    command: ["python", "train_ms_jp_extra.py"]
```

### 3. 永続ボリューム管理

**作成スクリプト**: `docker-volumes-setup.sh`

```bash
# ボリューム作成
./docker-volumes-setup.sh

# 状態確認
docker volume ls | grep voice_speech_maker
```

### 4. データパス形式の確認

**esd.listファイル形式**: `raw/`からの相対パス

```text
# 正しい形式
foo.wav|speaker1|JP|こんにちは|...
style1/bar.wav|speaker2|JP|さようなら|...

# train_ms_jp_extra.pyが自動的にwavs/パスに変換
```

## 実行手順

### Phase 1: 環境準備

```powershell
# 1. ボリューム作成
./docker-volumes-setup.sh

# 2. 現在のデータをボリュームにコピー（必要に応じて）
docker run --rm -v ${PWD}/Data:/host_data -v voice_speech_maker_data:/container_data alpine cp -r /host_data/. /container_data/
```

### Phase 2: Docker環境起動

```powershell
# Windows
./run-docker.bat

# または直接docker-compose
docker-compose up -d --build
```

### Phase 3: 学習実行（準備中）

```powershell
# コンテナ内で学習実行
docker-compose exec voice-speech-maker-train python train_ms_jp_extra.py
```

## 技術的詳細

### データローディング検証

- `data_utils.py`のTextAudioSpeakerLoaderを分析済み
- `preprocess_text.py`の`--correct_path`オプションでパス変換確認済み
- esd.listの相対パス形式が正しく処理されることを確認

### Modal適合性

- ポート露出なし（学習専用）
- 永続ボリューム使用
- バッチ処理対応
- 再現可能なGitベースデプロイ

## 現在の状態

### ✅ 完了

1. Dockerfile.git作成（Gitクローンベース）
2. docker-compose.yml設定（ボリューム管理）
3. ボリューム作成スクリプト完成
4. 起動スクリプト作成（Linux/Windows対応）
5. データパス形式確認・文書化
6. Dockerビルド成功

### 🔄 進行中

1. 実際の学習実行テスト
2. ボリュームデータ初期化
3. Modal環境への移植準備

### 📋 次のステップ

1. 学習データの準備と前処理
2. 実際の学習実行とログ確認
3. Modal環境でのテスト実行

## トラブルシューティング

### 一般的な問題

1. **Dockerビルドエラー**: `docker-compose up --build`で再ビルド
2. **ボリューム権限**: Docker Desktop設定でファイル共有を確認
3. **GPU認識**: nvidia-docker-runtimeの設定確認

### ログ確認

```powershell
# コンテナログ
docker-compose logs -f voice-speech-maker-train

# ボリューム内容確認
docker run --rm -v voice_speech_maker_logs:/logs alpine ls -la /logs
```

## リポジトリ情報

- **プロジェクト**: VoiceSpeechMaker Modal対応版
- **リポジトリ**: <https://github.com/ganpare/VoiceSpeechMaker-core.git>
- **目的**: Modern BERT音声合成の学習環境構築
- **Modal対応**: 永続ボリューム、バッチ処理、再現可能性

---

**作成日**: 2025年7月6日  
**最終更新**: Docker環境構築完了、学習実行準備中
