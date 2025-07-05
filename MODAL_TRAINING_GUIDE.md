# Style-Bert-VITS2 Modal学習システム 完全ガイド

## 🎯 概要

このプロジェクトは、Style-Bert-VITS2音声合成モデルをModalクラウドプラットフォーム上で学習するシステムです。ローカルマシン（Mac）で前処理を行い、Modal上のGPUで効率的に学習を実行します。

## 🏗️ システム構成

- **ローカル環境**: 前処理（CPU集約的タスク）
- **Modal環境**: 学習（GPU集約的タスク）
- **ModernBert**: 日本語テキスト処理用のBERT後継モデル
- **WavLM**: 音声判別器モデル

## 📋 前提条件

1. **Python 3.10以上**
2. **Modal アカウント** - [modal.com](https://modal.com)でアカウント作成
3. **十分なストレージ** - 音声データとモデル用

## 🛠️ セットアップ手順

### 1. 環境構築

```bash
# リポジトリをクローン
git clone <your-repo-url>
cd VoiceSpeechMaker-core

# 必要な分岐に切り替え
git checkout feature-new  # または該当ブランチ

# 依存関係をインストール
pip install -e .
```

### 2. Modal設定

```bash
# Modalにログイン
modal setup

# 認証情報を確認
modal auth current
```

### 3. データセット準備

キャラクター用のディレクトリ構造：
```
Data/
└── <キャラクター名>/
    ├── <音声ファイル名>.wav (複数)
    └── esd.list (音声ファイルとテキストの対応リスト)
```

#### esd.listの形式例：
```
<音声ファイル名>|<発話者名>|JP|<読み上げテキスト>
voice001.wav|大蔵衣遠|JP|おはようございます。今日もよろしくお願いします。
voice002.wav|大蔵衣遠|JP|ありがとうございました。
```

## 🚀 学習実行手順

### ステップ1: ローカル前処理

```bash
# テキスト前処理を実行
python preprocess_text.py --transcription_path Data/<キャラクター名>/esd.list --correct_path

# 音声前処理を実行  
python preprocess_all.py --model_name <キャラクター名> --use_jp_extra
```

**前処理完了の確認項目：**
- `Data/<キャラクター名>/train.list` が生成されている
- `Data/<キャラクター名>/val.list` が生成されている
- `Data/<キャラクター名>/config.json` の `n_speakers` が1になっている
- `spk2id` に正しいキャラクター名が設定されている

### ステップ2: データをModalボリュームにアップロード

```bash
# ローカルデータセットをModalボリュームにアップロード
python3 -m modal run modal_training.py --action=upload-local --dataset-name=<キャラクター名>
```

### ステップ3: Modal学習実行

```bash
# 学習開始（10エポック）
python3 -m modal run modal_training.py --action=train --dataset-name=<キャラクター名>
```

### ステップ4: 学習済みモデルのダウンロード

```bash
# 学習済みモデルをダウンロード
python3 -m modal run modal_training.py --action=download --dataset-name=<キャラクター名>
```

## 📊 利用可能なコマンド

### データセット管理
```bash
# 利用可能なデータセットとモデルをリスト表示
python3 -m modal run modal_training.py --action=list

# ボリューム内の特定パスをリスト表示
python3 -m modal run modal_training.py --action=list-path --path-to-list=/data/Data

# データセットの設定を確認
python3 -m modal run modal_training.py --action=read-config --dataset-name=<キャラクター名>
```

### エラー診断
```bash
# テキスト前処理エラーログを確認
python3 -m modal run modal_training.py --action=read-error-log --dataset-name=<キャラクター名>
```

## 🔧 設定詳細

### 学習パラメータ

`modal_training.py`の主要設定：
- **エポック数**: 10（テスト用）、本格学習では100-1000
- **バッチサイズ**: 2（GPUメモリに応じて調整）
- **学習率**: 0.0001
- **GPU**: A10G（Modal標準）
- **メモリ**: 32GB RAM

### ModernBert設定

- **モデル**: `ruri-v3-30m`（HuggingFace cl-nagoya）
- **トークナイザー**: SentencePiece（`tokenizer.model`）
- **対応言語**: 日本語メイン

### WavLM設定

- **モデル**: `microsoft/wavlm-base-plus`
- **用途**: 音声判別器として使用
- **自動ダウンロード**: Modalイメージビルド時に実行

## 🎭 新しいキャラクターの追加

### 1. 音声データ準備
- 44.1kHz、16bit WAV形式推奨
- 1ファイルあたり2-10秒程度
- 最低100ファイル、理想的には300-500ファイル

### 2. esd.list作成
- UTF-8エンコーディング
- パイプ区切り形式：`ファイル名|話者名|言語|テキスト`
- テキストは正しい日本語表記

### 3. 前処理実行
```bash
python preprocess_text.py --transcription_path Data/<新キャラクター名>/esd.list --correct_path
python preprocess_all.py --model_name <新キャラクター名> --use_jp_extra
```

### 4. Modal学習
```bash
python3 -m modal run modal_training.py --action=upload-local --dataset-name=<新キャラクター名>
python3 -m modal run modal_training.py --action=train --dataset-name=<新キャラクター名>
```

## 🐛 トラブルシューティング

### よくあるエラーと解決法

#### 1. `n_speakers must be > 0`
- **原因**: config.jsonのn_speakersが0
- **解決**: 前処理を正しく実行し、train.listが生成されているか確認

#### 2. `expected str, bytes or os.PathLike object, not NoneType`
- **原因**: ModernBertトークナイザーの問題
- **解決**: `bert/ruri-v3-30m/tokenizer.model`が存在するか確認

#### 3. `WavLM model not found`
- **原因**: WavLMモデルファイルの不足
- **解決**: 自動ダウンロードが有効化されているため、最新版では解決済み

#### 4. 音声ファイルが見つからない
- **原因**: ファイルパスの不一致
- **解決**: `--correct_path`フラグを使用してpreprocess_text.pyを実行

### ログ確認方法

```bash
# Modal実行ログはコンソールに表示
# 詳細ログはModal Webコンソールで確認: https://modal.com/

# ローカルエラーログ確認
cat Data/<キャラクター名>/text_error.log
```

## 📈 性能最適化

### 学習時間短縮
- バッチサイズを4-8に増加（GPUメモリが許す限り）
- Mixed precision（bf16）を有効化（設定済み）
- 適切なエポック数設定（品質と時間のバランス）

### 品質向上
- より多くの音声データを準備
- 音声品質の統一（ノイズ除去、音量正規化）
- 適切なテキスト正規化

## 💰 コスト管理

- **Modal料金**: GPU使用時間に基づく従量課金
- **A10G使用例**: 10エポック学習で約$2-5
- **長時間学習**: 事前に予算設定を推奨

## 🔐 セキュリティ注意事項

- Modalトークンは適切に管理
- 機密データのアップロード前に確認
- 公開リポジトリでのトークン露出に注意

## 📚 参考資料

- [Modal Documentation](https://modal.com/docs)
- [Style-Bert-VITS2 GitHub](https://github.com/litagin02/Style-Bert-VITS2)
- [ModernBert Model](https://huggingface.co/cl-nagoya/ruri-v3-30m)
- [WavLM Model](https://huggingface.co/microsoft/wavlm-base-plus)

## 🎉 成功事例

### 大蔵衣遠キャラクター
- **データ**: 367音声ファイル
- **学習時間**: 約4分（10エポック）
- **結果**: 正常に学習完了、モデル保存成功

---

**🚀 Happy Voice Training! 🎤**