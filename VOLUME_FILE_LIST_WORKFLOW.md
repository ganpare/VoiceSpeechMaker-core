# Modal Volume ファイル一覧取得・ダウンロード手順

このドキュメントは、Modalクラウド環境でVolume（永続ストレージ）内のファイル・ディレクトリ一覧を取得し、ローカルPCにダウンロードするまでの一連のワークフローをまとめたものです。

---

## 1. 概要

- ModalのVolume（例: `sbv2-data-volume`）内の全ファイル・ディレクトリ一覧をPythonスクリプトで取得
- その一覧をテキストファイル（例: `volume_file_list.txt`）として**別のVolume（例: `style-bert-vits2-volume`）に保存**
- Modal Web UIからそのファイルをダウンロード

---

## 2. スクリプト例

`list_volume_files.py` というファイル名で以下のようなスクリプトを作成

```python
import modal
import os

image = modal.Image.debian_slim(python_version="3.10")
app = modal.App("list-volume-files", image=image)
data_volume = modal.Volume.from_name("sbv2-data-volume")
output_volume = modal.Volume.from_name("style-bert-vits2-volume")

@app.function(
    volumes={
        "/mnt/gamedata": data_volume,
        "/mnt/output": output_volume
    }
)
def list_files():
    out_path = "/mnt/output/volume_file_list.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk("/mnt/gamedata"):
            level = root.replace("/mnt/gamedata", "").count(os.sep)
            indent = "  " * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = "  " * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")
    print(f"{out_path} に出力しました")

@app.local_entrypoint()
def main():
    list_files.remote()
```

---

## 3. 実行方法

1. 上記スクリプトをリポジトリ直下に保存
2. ターミナルで以下を実行

```sh
modal run list_volume_files.py
```

- `sbv2-data-volume`の全ファイル一覧が`style-bert-vits2-volume`の`/mnt/output/volume_file_list.txt`に保存されます

---

## 4. ダウンロード方法

1. Modal Web UI（[https://modal.com/volumes](https://modal.com/volumes) など）にアクセス
2. `style-bert-vits2-volume`を選択
3. `/mnt/output/volume_file_list.txt` を探してクリックし、ダウンロード

---

## 5. 注意点

- ModalのVolumeは**ローカルPCのディレクトリとは自動同期されません**
- 必ずVolume経由でファイルをやり取りしてください
- Dockerのマウントとは仕組みが異なります

---

## 6. 参考: 現在のVolume一覧取得

```sh
modal volume list
```

---

## 7. 追加メモ

- 他のVolumeやパスでも同様の方法で一覧取得・保存が可能
- さらに自動化したい場合は、Functionのreturnで内容を返す・API経由で取得するなども応用できます 

---

## 8. Modal関連ファイル・設定ファイルの役割と現状

### modal_training.py
- Modalクラウド上でStyle-Bert-VITS2の分散学習・データアップロード・モデル保存などを一括管理するメインスクリプト。
- Modal Functionとして学習（train_model）、データアップロード（upload_dataset）、モデルダウンロード（download_model）、**データセット前処理（preprocess_dataset）**、**ボリュームパス内容リスト（list_volume_path）** などを提供。
- Modalクラウドで本格的な学習・データ管理を行う場合は必須。
- **現在の状況**:
    - GPUタイプをA10Gに設定済み。
    - `train_model` および `preprocess_dataset` 関数内のデータセットパスを `/data/Data/キャラクター名` に修正済み。
    - `train_model` 関数内のエポック数をテスト用に `10` に設定済み。
    - `main` 関数に `preprocess` および `list-path` アクションを追加済み。
    - `config.json` 内の `training_files` と `validation_files` のパスを動的に更新するように修正済み。

### modal_utils.py
- Modal学習用の設定管理・コスト見積もり・データセット検証などのユーティリティ集。
- Modalクラウドでの運用設計やコスト見積もり、データ検証を自動化したい場合に有用。

### modal_manager.py
- Modal学習システムのローカル管理ツール。コマンドラインからデータセット準備・検証・コスト見積もり・学習設定生成などを実行。
- Modalクラウドでの大規模運用や複数データセット管理を行う場合に便利。

### setup_modal.py
- Modal学習システムのクイックセットアップスクリプト。依存インストール・ディレクトリ作成・サンプルファイル生成・認証セットアップなどを自動化。
- 新規環境構築や初回セットアップ時に便利。

### upload_to_modal.py
- ローカルのデータセットをModalクラウドのボリュームにアップロードするためのスクリプト。
- データセットをクラウドにアップロードする場合は有用。

### list_volume_files.py
- Modal Volume（永続ストレージ）内のファイル・ディレクトリ一覧を取得し、別のVolumeにテキスト出力するためのスクリプト。
- Volumeの中身を可視化・管理したい場合に便利。

### train_ms_jp_extra.py
- Style-Bert-Vits2の日本語拡張モデル学習用スクリプト。Modalクラウド対応のイメージ定義・ボリュームマウントも含む。
- 日本語モデルの学習・Modalクラウドでの実行に必須。

### requirements.txt
- 依存パッケージ管理ファイル。Modalクラウド用にtorch/torchaudio等のバージョン調整済み。
- Modalクラウド・ローカル両方で依存解決に必須。

### modal_env/
- Python仮想環境（venv）ディレクトリ。ローカル開発・依存解決用。
- 仮想環境の生成物なのでgit管理から除外（.gitignore済み）。Modalクラウド実行には直接不要。

### その他
- setup_modal_auth.py：空ファイル。認証セットアップ用の名残で、現状不要。
- docs/MODAL_TRAINING.md：Modal学習運用の詳細ドキュメント。運用設計や手順の参考に。

---

### 【現在の課題とデバッグ状況】
- **課題**: 前処理ステップで `Step 1: pretrained folder not found.` エラーが継続して発生。
- **状況**:
    - `pretrained_jp_extra` ディレクトリはローカルに存在し、`modal_training.py` の `modal.Image` 定義でModalコンテナの `/app/pretrained_jp_extra` に追加されていることを確認済み。
    - Modalコンテナ内の `/app/pretrained_jp_extra` にファイルが存在することもデバッグ出力で確認済み。
    - 問題は、`gradio_tabs/train.py` が `subprocess.run` で実行される際のパス解決、または `shutil.copytree` の挙動にある可能性が高い。
- **次の対応方針**:
    - `modal_training.py` の `preprocess_dataset` 関数内で、データセットをボリュームからコピーした後、**明示的に `/app/pretrained_jp_extra` の内容を `Data/{dataset_name}/models` ディレクトリにコピーする**ロジックを追加する。これにより、`gradio_tabs/train.py` の内部コピーロジックをバイパスする。
    - デバッグ用の `print` ステートメントは役割を終えたので削除する。

---

### 【運用のポイントまとめ】
- Modalクラウドで本格運用する場合は、modal_training.py・modal_utils.py・modal_manager.py・requirements.txtが中心。
- データアップロードやVolume管理が必要な場合は、upload_to_modal.py・list_volume_files.pyも活用。
- ローカル開発や初期セットアップのみなら、setup系スクリプトは不要になることも多い。
- 仮想環境（modal_env/）はgit管理から除外し、依存はrequirements.txtで一元管理。