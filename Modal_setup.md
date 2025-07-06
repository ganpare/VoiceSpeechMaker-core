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

## 10. 🎯 LUNA0712継続学習：決定版手順

### 10.1 試行錯誤から学んだポイント

#### ❌ 失敗パターン
1. **不完全なデータ構造**: ローカルの豊富なデータ（4,830ファイル）を一部のみアップロード
2. **ディレクトリ構造の不一致**: Style-Bert-VITS2が期待する`models/`ディレクトリが欠如
3. **新規学習の繰り返し**: 継続学習が失敗して毎回e1から開始
4. **CLIタイムアウト**: `modal run`で2分制限にかかり中断

#### ✅ 成功パターン
1. **完全データセット**: ローカル`Data/LUNA0712/`の全ファイルをアップロード
2. **正しいディレクトリ構造**: `models/`、`models_backup/`を含む完全構造
3. **detachedモード**: `modal run --detach`で長時間学習を継続
4. **複数モデル形式**: PyTorch(.pth)、SafeTensors(.safetensors)、バックアップを全て配置

### 10.2 🚀 決定版：完全データセットでの継続学習手順

#### Step 1: ローカルデータ構造の確認
```bash
# ローカルData/LUNA0712/の構造確認
ls -la Data/LUNA0712/
# 期待される構造:
# ├── config.json, esd.list, train.list, val.list
# ├── raw/ (4,830+ .wav files)
# ├── models/ (PyTorchモデル: G_8000.pth, D_8000.pth, WD_8000.pth)
# └── models_backup/ (SafeTensors: G_0.safetensors, D_0.safetensors, WD_0.safetensors)
```

#### Step 2: 完全データセットアップロード
```bash
# 1. 全音声ファイル (4,830+ファイル、数分かかる)
modal volume put voice_speech_maker_data Data/LUNA0712/raw Data/LUNA0712/raw

# 2. 設定ファイル群
modal volume put voice_speech_maker_data Data/LUNA0712/esd.list Data/LUNA0712/esd.list
modal volume put voice_speech_maker_data Data/LUNA0712/train.list Data/LUNA0712/train.list
modal volume put voice_speech_maker_data Data/LUNA0712/val.list Data/LUNA0712/val.list

# 3. 学習済みモデル (PyTorchモデル)
modal volume put voice_speech_maker_models Data/LUNA0712/models LUNA0712/models

# 4. バックアップモデル (SafeTensorsモデル)  
modal volume put voice_speech_maker_models Data/LUNA0712/models_backup LUNA0712/models_backup
```

#### Step 3: データ構造確認
```bash
# アップロード確認
modal volume ls voice_speech_maker_data Data/LUNA0712/raw | wc -l  # 4,830+行
modal volume ls voice_speech_maker_models LUNA0712/models          # PyTorchモデル確認
modal volume ls voice_speech_maker_models LUNA0712/models_backup   # SafeTensorsモデル確認
modal volume ls voice_speech_maker_models LUNA0712                 # 既存チェックポイント確認
```

#### Step 4: 継続学習実行
```bash
# Detachedモードで長時間学習実行
modal run --detach modal_scripts/train_modal.py::train_luna0712
```

#### Step 5: 学習進捗確認
```bash
# 実行中アプリ確認
modal app list

# ログ確認 (app_idは上記で確認)
modal app logs <app_id>

# 新しいチェックポイント確認
modal volume ls voice_speech_maker_models LUNA0712 | grep "e27\|e28\|e29"
```

### 10.3 📊 期待される結果

#### 🎯 学習進行パターン
```
継続学習開始: LUNA0712_e26_s8000.safetensors
↓
新チェックポイント生成:
├── LUNA0712_e27_s9000.safetensors
├── LUNA0712_e28_s10000.safetensors  
├── LUNA0712_e29_s11000.safetensors
└── ... (継続)
```

#### 💾 コンテナ内ディレクトリ構造
```
/workspace/
├── Data/                          ← voice_speech_maker_data
│   └── LUNA0712/
│       ├── config.json, *.list
│       └── raw/ (4,830+ files)
├── model_assets/                  ← voice_speech_maker_models  
│   └── LUNA0712/
│       ├── models/ (PyTorch)      # G_8000.pth, D_8000.pth, WD_8000.pth
│       ├── models_backup/ (ST)    # G_0.safetensors, D_0.safetensors, WD_0.safetensors
│       ├── LUNA0712_e26_s8000.safetensors (既存最新)
│       └── LUNA0712_e27_s9000.safetensors (新規生成)
└── logs/                          ← voice_speech_maker_logs
```

#### ⚡ A100 GPU最適化設定
- **GPU**: A100 80GB
- **Batch Size**: 12 (config.jsonで設定済み)
- **継続学習**: e26_s8000から継続
- **学習データ**: 4,830音声ファイル完全活用

### 10.4 🔧 トラブルシューティング（決定版）

#### 問題1: "train from scratch"メッセージ
```
原因: models/G_0.safetensorsが見つからない
解決: models_backup/をアップロードして正しいディレクトリ構造にする
```

#### 問題2: BERTファイルエラー
```
原因: .bert.ptファイルが存在しない
解決: bert_gen.pyで事前生成（スクリプトに組み込み済み）
```

#### 問題3: モデル名がmodel_nameになる
```
原因: config.jsonのmodel_name設定が不正
解決: スクリプトで自動的に"LUNA0712"に修正
```

#### 問題4: CLIタイムアウト
```
原因: modal runが2分でタイムアウト
解決: modal run --detachを使用
```

## 11. 次のステップ

1. **完全データセットアップロード** ✅ 完了
   ```bash
   # 4,830音声ファイル + 学習済みモデル + バックアップモデル
   ```

2. **継続学習実行** ✅ 実行中
   ```bash
   modal run --detach modal_scripts/train_modal.py::train_luna0712
   ```

3. **学習結果の確認** ⏳ 待機中
   ```bash
   modal volume ls voice_speech_maker_models LUNA0712
   ```

4. **改良モデルのダウンロード** 📋 予定
   ```bash
   modal volume get voice_speech_maker_models "LUNA0712/LUNA0712_e30_s12000.safetensors" "./improved_luna0712.safetensors"
   ```

---

**作成日**: 2025年7月6日  
**最終更新**: LUNA0712継続学習決定版手順追加（試行錯誤完了）

**重要な学び**: 
- ローカルの豊富なデータ構造をそのまま活用することが成功の鍵
- PyTorch + SafeTensors + バックアップの3形式すべてアップロードが重要
- detachedモードで長時間学習を安定実行
- 4,830音声ファイル完全活用でLUNA0712の品質向上を実現