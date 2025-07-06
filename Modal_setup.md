# Modal セットアップガイド

Modern BERT Voice Speech Maker のModal環境構築手順

## 概要

Modalは**CLI中心**の設計で、インフラ管理はコマンド、実行ロジックはPythonで分業します。

```
管理 = CLI、実行 = Python
```

## 1. Modal環境のセットアップ

### 1.1 Modal CLIインストール
```bash
pip install modal
```

### 1.2 認証
```bash
modal setup
```
- ブラウザが開いてModal認証
- ワークスペースに接続完了
- 認証情報は `~/.modal.toml` に保存

## 2. ボリューム作成（一度のみ）

### 2.1 3つのボリューム作成
```bash
# 学習データ用
modal volume create voice_speech_maker_data

# モデル出力用
modal volume create voice_speech_maker_models

# ログ用
modal volume create voice_speech_maker_logs
```

### 2.2 ボリューム確認
```bash
modal volume list
```

## 3. データアップロード

### 3.1 テストファイルアップロード
```bash
# 単一ファイル
modal volume put voice_speech_maker_data "./Data/大蔵衣遠/esd.list" "/esd.list"

# アップロード確認
modal volume ls voice_speech_maker_data
```

### 3.2 全データアップロード（時間要注意）
```bash
# ディレクトリ全体アップロード（数分〜数十分）
modal volume put voice_speech_maker_data "./Data/大蔵衣遠" "/大蔵衣遠"
```

## 4. ボリューム管理コマンド

### 基本操作
```bash
# 作成
modal volume create <volume_name>

# ファイルアップロード
modal volume put <volume_name> <local_path> <remote_path>

# ファイル一覧
modal volume ls <volume_name>

# ファイルダウンロード
modal volume get <volume_name> <remote_path> <local_path>

# 削除
modal volume delete <volume_name>
```

### よく使うコマンド
```bash
# 全ボリューム一覧
modal volume list

# 特定ボリュームの中身確認
modal volume ls voice_speech_maker_data

# ディレクトリ構造確認
modal volume ls voice_speech_maker_data /大蔵衣遠
```

## 5. Python実行スクリプト

### 5.1 ボリューム参照（Python）
```python
import modal

# 既存ボリューム参照
volume = modal.Volume.from_name("voice_speech_maker_data")

# 関数でマウント
@app.function(volumes={"/workspace/Data": volume})
def train():
    # /workspace/Data/大蔵衣遠/raw/file.wav でアクセス
    pass
```

### 5.2 作成済みスクリプト
```bash
# ボリューム初期化（初回のみ）
python modal_scripts/setup_volume.py

# 学習実行（何度でも）
python modal_scripts/train_modal.py
```

## 6. DockerとModalの対応関係

### ボリューム管理
| 操作 | Docker | Modal |
|------|--------|-------|
| **作成** | `docker volume create` | `modal volume create` |
| **データ投入** | `docker run -v ... cp` | `modal volume put` |
| **確認** | `docker volume ls` | `modal volume ls` |
| **マウント** | `docker-compose.yml` | `@app.function()` |

### パス構造（同じ）
```
ホスト:     ./Data/大蔵衣遠/raw/file.wav
ボリューム: /大蔵衣遠/raw/file.wav
コンテナ:   /workspace/Data/大蔵衣遠/raw/file.wav
```

## 7. トラブルシューティング

### よくあるエラー

#### ボリュームが見つからない
```bash
# エラー: Volume 'xxx' not found
# 解決: ボリューム作成
modal volume create voice_speech_maker_data
```

#### アップロードがタイムアウト
```bash
# 大きなファイルは時間がかかる
# 小さなファイルでテストしてから実行
modal volume put voice_speech_maker_data "./test.txt" "/test.txt"
```

#### 認証エラー
```bash
# 認証し直し
modal setup
```

### デバッグ方法
```bash
# ボリューム状態確認
modal volume list
modal volume ls voice_speech_maker_data

# 実行ログ確認
modal app list
modal app logs <app_name>
```

## 8. 料金体系

### ボリューム料金
- ストレージ: $0.10/GB/月
- 転送: 無料（一般的な使用範囲）

### 実行料金
- GPU使用時のみ課金
- 使わない時は0円

## 9. 学習結果のダウンロード

### 9.1 チェックポイント保存先

学習中に以下の場所にファイルが保存されます：

#### **推論用モデル（.safetensors）**
```
voice_speech_maker_models/
└── LUNA0712/
    └── LUNA0712_e{epoch}_s{step}.safetensors
```

#### **訓練チェックポイント（.pth）**
```
voice_speech_maker_data/
└── LUNA0712/models/
    ├── G_{step}.pth      # Generator
    ├── D_{step}.pth      # Discriminator
    ├── DUR_{step}.pth    # Duration判別器
    └── WD_{step}.pth     # WavLM判別器
```

#### **ログファイル**
```
voice_speech_maker_logs/
└── train_{timestamp}.log
```

### 9.2 ダウンロードコマンド

#### **推論用モデルのダウンロード**
```bash
# 最新の推論用モデル（.safetensors）をダウンロード
modal volume get voice_speech_maker_models "/LUNA0712" "./downloaded_models/"
```

#### **訓練チェックポイントのダウンロード**
```bash
# 訓練再開用チェックポイント（.pth）をダウンロード
modal volume get voice_speech_maker_data "/LUNA0712/models" "./downloaded_checkpoints/"
```

#### **ログファイルのダウンロード**
```bash
# 学習ログをダウンロード
modal volume get voice_speech_maker_logs "/" "./downloaded_logs/"
```

#### **特定ファイルのダウンロード**
```bash
# 最新モデルのみダウンロード
modal volume get voice_speech_maker_models "/LUNA0712/LUNA0712_e30_s8000.safetensors" "./latest_model.safetensors"
```

### 9.3 ダウンロード後の使用方法

#### **推論での使用**
```bash
# ダウンロードしたモデルをmodel_assets/に配置
cp ./downloaded_models/LUNA0712/*.safetensors ./model_assets/LUNA0712/
```

#### **学習再開での使用**
```bash
# チェックポイントをData/に配置
cp ./downloaded_checkpoints/*.pth ./Data/LUNA0712/models/
```

## 10. 次のステップ

1. **全データアップロード**
   ```bash
   modal volume put voice_speech_maker_data "./Data" "/"
   ```

2. **学習実行**
   ```bash
   python modal_scripts/train_modal.py
   ```

3. **学習結果の確認**
   ```bash
   modal volume ls voice_speech_maker_models
   modal volume ls voice_speech_maker_logs
   ```

4. **モデルのダウンロード**
   ```bash
   modal volume get voice_speech_maker_models "/LUNA0712" "./downloaded_models/"
   ```

---

**作成日**: 2025年7月6日  
**最終更新**: Modal CLI環境構築完了

**メモ**: CLIが直感的で、DockerからModalへの移行が簡単