# Modal Training for Style-Bert-VITS2

このプロジェクトは、[Modal](https://modal.com/)サーバレスGPUプラットフォームを使用してStyle-Bert-VITS2モデルをクラウドで学習するためのシステムです。

## 特徴

🚀 **サーバレスGPU**: Modalの高性能GPU（A100等）を使用した効率的な学習  
📦 **永続ストレージ**: データセットとモデルの永続的な保存  
⚡ **オートスケーリング**: 需要に応じた自動的なリソース管理  
🔄 **学習再開**: 中断された学習の再開機能  
📊 **リアルタイム監視**: TensorBoardによる学習進捗の可視化  

## セットアップ

### 1. Modalアカウントの作成

1. [Modal](https://modal.com/)でアカウントを作成
2. Modalトークンを取得
3. Modal CLIをインストール:

```bash
pip install modal
```

4. 認証:

```bash
modal token set --token-id <YOUR_TOKEN_ID> --token-secret <YOUR_TOKEN_SECRET>
```

### 2. 環境構築

```bash
git clone <your-repo>
cd VoiceSpeechMaker-core-feature
pip install modal
```

## 使用方法

### データセットのアップロード

データセットをModalボリュームにアップロード:

```bash
# URLからアップロード
modal run modal_training.py --action=upload --dataset-name=my_voice --dataset-url=https://example.com/dataset.zip

# 手動アップロード用ディレクトリ作成
modal run modal_training.py --action=upload --dataset-name=my_voice
```

### 学習の実行

```bash
# 基本的な学習
modal run modal_training.py --action=train --dataset-name=my_voice

# カスタム設定での学習
modal run modal_training.py \
  --action=train \
  --dataset-name=my_voice \
  --epochs=500 \
  --batch-size=4 \
  --learning-rate=0.0002 \
  --use-jp-extra=true
```

### 学習済みモデルのダウンロード

```bash
modal run modal_training.py --action=download --dataset-name=my_voice
```

### 利用可能なリソースの確認

```bash
modal run modal_training.py --action=list
```

## パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `--action` | 実行するアクション (`train`, `upload`, `download`, `list`) | `train` |
| `--dataset-name` | データセット名 | `default` |
| `--config-path` | 設定ファイルパス | `configs/config.json` |
| `--epochs` | エポック数 | `1000` |
| `--batch-size` | バッチサイズ | `2` |
| `--learning-rate` | 学習率 | `0.0001` |
| `--dataset-url` | データセットのダウンロードURL | `None` |
| `--use-jp-extra` | 日本語拡張版を使用 | `true` |
| `--resume` | 学習を再開 | `false` |

## ディレクトリ構造

```
/data/               # データセット用永続ボリューム
  └── my_voice/      # データセット
      ├── wavs/      # 音声ファイル
      ├── esd.list   # ファイルリスト
      └── configs/   # 設定

/models/             # モデル用永続ボリューム
  ├── my_voice/      # 学習中のモデル
  │   ├── G_*.pth    # Generator checkpoints
  │   ├── D_*.pth    # Discriminator checkpoints
  │   └── ...
  └── my_voice_assets/  # 推論用アセット
      ├── model.safetensors
      ├── config.json
      └── style_vectors.npy
```

## コスト最適化

### GPU利用時間の最小化

- **事前処理の最適化**: データセットの前処理はローカルで実行
- **チェックポイント間隔**: 適切な保存間隔で学習を中断・再開可能
- **混合精度学習**: `bf16_run=true`で高速化とメモリ効率化

### バッチサイズの調整

A100 GPU使用時の推奨設定:

| モデルサイズ | バッチサイズ | メモリ使用量 |
|-------------|-------------|-------------|
| 標準 | 4-6 | ~30GB |
| 大規模 | 2-4 | ~35GB |
| 最大 | 1-2 | ~40GB |

## 監視とデバッグ

### TensorBoard

学習中のログはModalボリュームに保存され、以下でアクセス可能:

```bash
# ログの取得（別途実装が必要）
modal run modal_training.py --action=get-logs --dataset-name=my_voice
```

### ログの確認

```bash
# Modalダッシュボードでリアルタイムログを確認
# https://modal.com/apps
```

## トラブルシューティング

### よくある問題

1. **GPU メモリ不足**
   - バッチサイズを減らす
   - 混合精度学習を有効化
   - モデルサイズを調整

2. **ネットワークエラー**
   - データセットの再アップロード
   - 学習の再開機能を使用

3. **設定エラー**
   - 設定ファイルの構文確認
   - パスの確認

### サポート

- [Modal Documentation](https://modal.com/docs)
- [Style-Bert-VITS2 Issues](https://github.com/litagin02/Style-Bert-VITS2/issues)

## ライセンス

このプロジェクトはStyle-Bert-VITS2のライセンスに従います。
