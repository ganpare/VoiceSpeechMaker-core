# Style-Bert-VITS2 Modal学習システム 完全ガイド

## 🎯 概要

このプロジェクトは、Style-Bert-VITS2音声合成モデルをModalクラウドプラットフォーム上で学習するシステムです。ローカルマシン（Mac）で前処理を行い、Modal上のGPUで効率的に学習を実行します。

### 🔥 重要な発見：分散学習機能の復活

**実は、このプロジェクトには元々完全な分散学習（マルチGPU）機能が実装されていました！**

従来は単一GPU用に設定が固定されていましたが、調査の結果、以下の証拠が発見されました：

- ✅ `torch.distributed`の完全なimport
- ✅ `DistributedLengthGroupedSampler`（分散学習用データローダー）
- ✅ NCCL/Gloo backend設定（コメントアウトされていた）
- ✅ 分散学習用環境変数の読み込み処理

これらの機能を復活させることで、**真の並列GPU学習**が可能になりました。A100-80GB:2やH100:2設定で、大幅な学習速度向上が期待できます。

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
# 標準学習（A10G GPU - 24GB）
python3 -m modal run modal_training.py --action=train --dataset-name=<キャラクター名>

# 高性能学習（A100-80GB GPU - 80GB）
python3 -m modal run modal_training.py --action=train --dataset-name=<キャラクター名> --gpu-type=A100-80GB

# 超高速学習（2x A100-80GB GPU - 160GB）
python3 -m modal run modal_training.py --action=train --dataset-name=<キャラクター名> --gpu-type=A100-80GB:2

# 最高性能学習（2x H100 GPU - 160GB）
python3 -m modal run modal_training.py --action=train --dataset-name=<キャラクター名> --gpu-type=H100:2
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

### GPU設定とパフォーマンス

| GPU設定 | GPU Memory | RAM | 推奨用途 | バッチサイズ | 学習速度 | コスト |
|---------|-----------|-----|----------|-------------|---------|--------|
| **A10G** | 24GB | 32GB | 標準学習 | 2-4 | 標準 | 低 |
| **A100-80GB** | 80GB | 64GB | 大規模データ | 4-8 | 高速 | 中 |
| **A100-80GB:2** | 160GB | 128GB | 超高速学習 | 8-16 | 超高速 | 高 |
| **H100:2** | 160GB | 128GB | 最高性能 | 16-32 | 最高速 | 最高 |

### 🚀 分散学習の技術詳細

**復活した分散学習機能**：

1. **自動検出システム**: `WORLD_SIZE > 1`で分散学習を自動有効化
2. **NCCL Backend**: NVIDIA GPU間の高速通信プロトコル
3. **環境変数の自動設定**: Modal環境で必要な設定を自動注入
4. **同期学習**: 複数GPU間でのグラディエント同期

**コード変更点**：
```python
# 修正前（単一GPU固定）
rank = 0  # 単一GPU用に固定
local_rank = 0  # 単一GPU用に固定
n_gpus = 1  # 単一GPU用に固定

# 修正後（分散学習対応）
if int(os.environ.get("WORLD_SIZE", 1)) > 1:
    # マルチGPU環境で分散学習を自動有効化
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = int(os.environ["WORLD_SIZE"])
```

**学習速度の理論値**：
- **2x A100-80GB**: 約1.8倍高速化（通信オーバーヘッドを考慮）
- **2x H100**: 約1.9倍高速化（より効率的な通信）

### 学習パラメータ

各GPU設定での最適パラメータ：

**A10G (標準)**：
- エポック数: 100-1000
- バッチサイズ: 2-4
- 学習時間: 30-60分/100エポック

**A100-80GB (高性能)**：
- エポック数: 100-1000
- バッチサイズ: 4-8
- 学習時間: 10-20分/100エポック

**マルチGPU (超高速)**：
- エポック数: 100-1000
- バッチサイズ: 8-32
- 学習時間: 5-10分/100エポック
- ✅ **分散学習が自動有効化されます**

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
# データアップロード
python3 -m modal run modal_training.py --action=upload-local --dataset-name=<新キャラクター名>

# GPU設定を選択して学習
python3 -m modal run modal_training.py --action=train --dataset-name=<新キャラクター名> --gpu-type=A100-80GB
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

#### 5. 分散学習が有効化されない
- **原因**: 環境変数`WORLD_SIZE`が正しく設定されていない
- **解決**: マルチGPU関数使用時は自動で設定されます
- **確認方法**: ログで"Multi-GPU training ENABLED!"メッセージを確認

### ログ確認方法

```bash
# Modal実行ログはコンソールに表示
# 詳細ログはModal Webコンソールで確認: https://modal.com/

# ローカルエラーログ確認
cat Data/<キャラクター名>/text_error.log
```

## 📈 性能最適化

### GPU選択指針

**データ量による推奨GPU**：
- **〜200ファイル**: A10G（コスト効率重視）
- **200-500ファイル**: A100-80GB（バランス型）
- **500+ファイル**: A100-80GB:2（高速処理）
- **1000+ファイル**: H100:2（最高性能）

### 学習時間短縮
- **GPU上位選択**: A10G → A100-80GB → マルチGPU
- **バッチサイズ最適化**: GPUメモリ容量に応じて調整
- **Mixed precision**: bf16有効化（設定済み）
- **並列学習**: マルチGPU環境での分散学習（✅ 実装済み）

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
- **学習時間**: 約4分（10エポック、A10G単体）
- **結果**: 正常に学習完了、モデル保存成功

### 分散学習機能の発見と復活
- **発見**: train_ms_jp_extra.py内に完全な分散学習実装を発見
- **復活作業**: 168-170行目の固定設定を動的設定に変更
- **技術要素**: torch.distributed、NCCL backend、環境変数自動設定
- **期待効果**: 2x A100-80GBで約1.8倍、2x H100で約1.9倍の高速化

### 分散学習の実装詳細
- **元のコード**: `DistributedLengthGroupedSampler`等、分散学習用コンポーネントが既存
- **コメントアウト**: NCCLバックエンド初期化がコメントアウトされていた
- **復活手法**: 環境変数`WORLD_SIZE`による条件分岐で自動有効化
- **Modal統合**: マルチGPU関数で環境変数を自動設定

---

**🚀 Happy Voice Training! 🎤**