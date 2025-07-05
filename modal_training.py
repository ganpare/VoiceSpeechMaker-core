"""
Modal-based distributed training for Style-Bert-VITS2
サーバレスGPUプラットフォームModalを使用した分散学習システム
"""

import modal
import os
import subprocess
import sys
import json
import shutil
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "torchaudio", 
        "transformers==4.48",
        "accelerate",
        "tensorboard",
        "librosa==0.9.2",
        "gradio>=4.32",
        "numpy<2",
        "loguru",
        "protobuf==4.25",
        "psutil",
        "pyopenjtalk-mod",
        "pyworld-prebuilt",
        "stable_ts",
        "faster-whisper>=1.0.0",
        "ctranslate2>=4.0",
        "pyloudnorm",
        "pyannote.audio>=3.1.0",
        "punctuators",
        "umap-learn",
        "setuptools<80.9",
    ])
    .apt_install(["git", "wget", "unzip", "ffmpeg"])
    .run_commands([
        "pip install --upgrade pip",
    ])
    # 現在のプロジェクトディレクトリをコピー
    .add_local_dir("pretrained_jp_extra", "/app/pretrained_jp_extra") # この行を追加
    .add_local_dir(".", "/app")
)

# Modal App定義
app = modal.App("style-bert-vits2-training", image=image)

# 永続ボリューム定義（データとモデルを保存）
data_volume = modal.Volume.from_name("sbv2-data-volume", create_if_missing=True)
model_volume = modal.Volume.from_name("style-bert-vits2-models", create_if_missing=True)

@app.function(
    gpu="A10G",  # A10G GPU x1を使用
    volumes={
        "/data": data_volume,
        "/models": model_volume,
    },
    timeout=86400,  # 24時間のタイムアウト
    memory=32768,   # 32GB RAM
    cpu=8,          # 8 CPUs
)
def train_model(
    config_path: str,
    dataset_name: str,
    epochs: int = 1000,
    batch_size: int = 2,
    learning_rate: float = 0.0001,
    use_jp_extra: bool = True,
    resume_training: bool = False
):
    """
    Modal上でStyle-Bert-VITS2モデルを学習する関数
    
    Args:
        config_path: 設定ファイルのパス
        dataset_name: データセット名
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        use_jp_extra: 日本語拡張版を使用するか
        resume_training: 学習を再開するか
    """
    print("🚀 Starting Style-Bert-Vits2 training on Modal...")
    
    # 作業ディレクトリを設定（イメージに含まれたソースコードを使用）
    os.chdir("/app")
    
    # ディレクトリ構造を修正：/data/Data を /Data にシンボリックリンク
    if not os.path.exists("/Data") and os.path.exists("/data/Data"):
        os.symlink("/data/Data", "/Data")
        print("✅ Created symlink: /data/Data -> /Data")
    
    # 必要に応じて追加パッケージをインストール
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    # Debug: List contents of /app and /app/pretrained_jp_extra
    print(f"Contents of /app: {os.listdir('/app')}")
    try:
        print(f"Contents of /app/pretrained_jp_extra: {os.listdir('/app/pretrained_jp_extra')}")
    except FileNotFoundError:
        print("Warning: /app/pretrained_jp_extra not found in container.")
    
    # データセットをボリュームからコピー
    volume_source_path = Path("/data/Data") / dataset_name
    local_destination_path = Path("Data") / dataset_name

    try:
        print(f"📂 Attempting to copy dataset from volume: {volume_source_path}")
        # Debug: List contents of the parent directory
        print(f"Contents of {volume_source_path.parent}: {os.listdir(volume_source_path.parent)}")
        if local_destination_path.exists():
            shutil.rmtree(local_destination_path)
        shutil.copytree(volume_source_path, local_destination_path)
        print(f"✅ Dataset copied successfully to {local_destination_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset {dataset_name} not found in volume at {volume_source_path}. Please ensure the dataset exists at this path on the Modal volume.")
    
    # キャラクター固有の設定ファイルを読み込み、更新
    character_config_path = local_destination_path / "config.json"
    
    if character_config_path.exists():
        # 既存のconfig.jsonが存在する場合（前処理済み）は、学習パラメータのみ更新
        print(f"📄 Using existing config.json with preprocessed speaker information")
        with open(character_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # 前処理されていない場合のみデフォルトをコピー
        shutil.copy(config_path, character_config_path)
        print(f"📝 Copied {config_path} to {character_config_path}")
        with open(character_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # training_files と validation_files のパスを設定
        config["data"]["training_files"] = f"Data/{dataset_name}/train.list"
        config["data"]["validation_files"] = f"Data/{dataset_name}/val.list"
    
    # GPUと学習設定を更新（既存の話者情報は保持）
    config["train"]["epochs"] = epochs
    config["train"]["batch_size"] = batch_size
    config["train"]["learning_rate"] = learning_rate
    config["train"]["bf16_run"] = True  # Mixed precision有効化
    
    with open(character_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"⚙️ Updated training configuration in {character_config_path}")

    # 事前学習済みモデルをボリュームからコピー（存在する場合）
    model_path = Path("/models") / dataset_name
    local_model_path = Path("Data") / dataset_name / "models"
    
    if model_path.exists() and resume_training:
        print("📦 Loading pre-trained models from volume...")
        if local_model_path.exists():
            shutil.rmtree(local_model_path)
        local_model_path.parent.mkdir(exist_ok=True)
        shutil.copytree(model_path, local_model_path)
    
    # 学習スクリプトを実行
    train_script = "train_ms_jp_extra.py" if use_jp_extra else "train_ms.py"
    
    print(f"🔥 Starting training with {train_script}...")
    
    cmd = [
        sys.executable, train_script,
        "--config", str(character_config_path), # キャラクター固有のconfig.jsonを渡す
        "--model", str(local_destination_path.parent / dataset_name),
    ]
    
    if not resume_training:
        cmd.append("--skip_default_style")
    
    # 学習実行
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Training completed successfully!")
        print(result.stdout)
        
        # 学習済みモデルをボリュームに保存
        if local_model_path.exists():
            print("💾 Saving trained models to volume...")
            if model_path.exists():
                shutil.rmtree(model_path)
            model_path.parent.mkdir(exist_ok=True)
            shutil.copytree(local_model_path, model_path)
        
        # 最終的なモデルアセットもボリュームに保存
        model_assets_path = Path("model_assets") / dataset_name
        volume_assets_path = Path("/models") / f"{dataset_name}_assets"
        
        if model_assets_path.exists():
            print("📦 Saving model assets to volume...")
            if volume_assets_path.exists():
                shutil.rmtree(volume_assets_path)
            volume_assets_path.parent.mkdir(exist_ok=True)
            shutil.copytree(model_assets_path, volume_assets_path)
        
        return {
            "status": "success",
            "message": "Training completed successfully",
            "dataset": dataset_name,
            "epochs_completed": epochs
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print(f"Error output: {e.stderr}")
        return {
            "status": "error", 
            "message": f"Training failed: {e}",
            "error_output": e.stderr
        }


@app.function(
    volumes={"/data": data_volume},
    timeout=7200,  # 2時間のタイムアウト（大きなデータセット用）
    cpu=4, # 前処理はCPUで十分
)
def preprocess_dataset(
    dataset_name: str,
    normalize: bool = True,
    trim: bool = True,
    num_processes: int = 4,
    use_jp_extra: bool = True,
):
    """
    Modal上でデータセットの前処理を行う関数
    
    Args:
        dataset_name: データセット名
        normalize: 音声のラウドネス正規化を行うか
        trim: 無音部分をトリミングするか
        num_processes: 並列処理のプロセス数
    """
    print(f"🚀 Starting preprocessing for dataset: {dataset_name}")
    
    os.chdir("/app")
    
    # ディレクトリ構造を修正：/data/Data を /Data にシンボリックリンク
    if not os.path.exists("/Data") and os.path.exists("/data/Data"):
        os.symlink("/data/Data", "/Data")
        print("✅ Created symlink: /data/Data -> /Data")
    
    # 必要に応じて追加パッケージをインストール
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    # Debug: List contents of /app and /app/pretrained_jp_extra
    print(f"Contents of /app: {os.listdir('/app')}")
    try:
        print(f"Contents of /app/pretrained_jp_extra: {os.listdir('/app/pretrained_jp_extra')}")
    except FileNotFoundError:
        print("Warning: /app/pretrained_jp_extra not found in container.")
    
    # データセットをボリュームからコピー
    volume_source_path = Path("/data/Data") / dataset_name
    local_destination_path = Path("Data") / dataset_name

    try:
        print(f"📂 Attempting to copy dataset from volume: {volume_source_path}")
        # Debug: List contents of the parent directory
        print(f"Contents of {volume_source_path.parent}: {os.listdir(volume_source_path.parent)}")
        if local_destination_path.exists():
            shutil.rmtree(local_destination_path)
        shutil.copytree(volume_source_path, local_destination_path)
        print(f"✅ Dataset copied successfully to {local_destination_path}")
        
        # ファイル構造を整理：wavファイルをrawディレクトリに移動
        raw_dir = local_destination_path / "raw"
        wav_files = list(local_destination_path.glob("*.wav"))
        
        if wav_files:
            raw_dir.mkdir(exist_ok=True)
            print(f"📁 Created raw directory: {raw_dir}")
            
            for wav_file in wav_files:
                target_path = raw_dir / wav_file.name
                shutil.move(str(wav_file), str(target_path))
                
            print(f"✅ Moved {len(wav_files)} wav files to raw directory")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset {dataset_name} not found in volume at {volume_source_path}. Please ensure the dataset exists at this path on the Modal volume.")

    # Debug: Check for esd.list
    esd_list_path = local_destination_path / "esd.list"
    if not esd_list_path.exists():
        print(f"⚠️ Warning: {esd_list_path} not found. Please ensure esd.list exists in your dataset directory.")
        # Optionally, raise an error or return early if esd.list is mandatory
        # raise FileNotFoundError(f"esd.list not found in {local_destination_path}. Preprocessing requires this file.")
    
    # preprocess_all.py を実行
    cmd = [
        sys.executable, "preprocess_all.py",
        "--model_name", dataset_name,
        "--num_processes", str(num_processes),
    ]
    if normalize:
        cmd.append("--normalize")
    if trim:
        cmd.append("--trim")
    if use_jp_extra:
        cmd.append("--use_jp_extra")
        
    print(f"🔥 Running preprocessing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Preprocessing completed successfully!")
        print(result.stdout)
        
        # 前処理結果をボリュームに保存
        print("💾 Saving preprocessed data back to volume...")
        # local_destination_path から volume_source_path へコピー
        if volume_source_path.exists():
            shutil.rmtree(volume_source_path)
        shutil.copytree(local_destination_path, volume_source_path)
        
        return {
            "status": "success",
            "message": "Preprocessing completed successfully",
            "dataset": dataset_name,
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed: {e}")
        print(f"Error output: {e.stderr}")
        return {
            "status": "error", 
            "message": f"Preprocessing failed: {e}",
            "error_output": e.stderr
        }


@app.function(
    volumes={"/data": data_volume},
    timeout=3600,
)
def upload_dataset(dataset_name: str, dataset_archive_url: str = None):
    """
    データセットをModalボリュームにアップロードする関数
    
    Args:
        dataset_name: データセット名
        dataset_archive_url: データセットのアーカイブURL（zip形式）
    """
    print(f"📤 Uploading dataset: {dataset_name}")
    
    data_path = Path("/data") / dataset_name
    
    if dataset_archive_url:
        # URLからデータセットをダウンロード
        print(f"📥 Downloading dataset from {dataset_archive_url}")
        archive_path = Path("/tmp") / f"{dataset_name}.zip"
        
        urllib.request.urlretrieve(dataset_archive_url, archive_path)
        
        # ZIPファイルを展開
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        print(f"✅ Dataset uploaded successfully to {data_path}")
        return {"status": "success", "path": str(data_path)}
    
    else:
        # 手動アップロード用のディレクトリを作成
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory for manual upload: {data_path}")
        return {"status": "ready", "path": str(data_path)}


@app.function(
    volumes={"/data": data_volume},
    timeout=7200,  # 2時間のタイムアウト（大きなデータセット用）
)
def upload_dataset_direct(dataset_name: str, dataset_files: dict):
    """
    ローカルからデータセットを直接アップロードする関数
    
    Args:
        dataset_name: データセット名
        dataset_files: {"filename": file_content} の辞書
    """
    print(f"📤 Uploading dataset directly: {dataset_name}")
    
    data_path = Path("/data") / dataset_name
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 各ファイルを保存
    for filename, file_content in dataset_files.items():
        file_path = data_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Base64デコードして保存
        if isinstance(file_content, str):
            # テキストファイルの場合
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
        else:
            # バイナリファイルの場合（base64エンコード済み）
            file_data = base64.b64decode(file_content)
            with open(file_path, 'wb') as f:
                f.write(file_data)
        
        print(f"  📁 Saved: {filename}")
    
    print(f"✅ Dataset uploaded successfully to {data_path}")
    return {"status": "success", "path": str(data_path), "files_count": len(dataset_files)}


@app.function(
    volumes={"/models": model_volume},
    timeout=1800,
)
def download_model(dataset_name: str, download_path: str = None):
    """
    学習済みモデルをダウンロードする関数
    
    Args:
        dataset_name: データセット名
        download_path: ダウンロード先パス（ローカル）
    """
    print(f"📦 Preparing model download for: {dataset_name}")
    
    model_path = Path("/models") / dataset_name
    assets_path = Path("/models") / f"{dataset_name}_assets"
    
    if not model_path.exists() and not assets_path.exists():
        return {"status": "error", "message": "Model not found"}
    
    # 一時ディレクトリでZIPファイルを作成
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / f"{dataset_name}_model.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # モデルファイルを追加
            if model_path.exists():
                for file_path in model_path.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(model_path.parent)
                        zipf.write(file_path, arcname)
            
            # アセットファイルを追加
            if assets_path.exists():
                for file_path in assets_path.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(assets_path.parent)
                        zipf.write(file_path, arcname)
        
        # ZIPファイルの内容を読み取り
        with open(zip_path, 'rb') as f:
            zip_content = f.read()
    
    print(f"✅ Model package created (size: {len(zip_content)} bytes)")
    
    return {
        "status": "success",
        "dataset": dataset_name,
        "zip_content": zip_content,
        "filename": f"{dataset_name}_model.zip"
    }


@app.function()
def list_datasets():
    """
    利用可能なデータセットと学習済みモデルをリストアップ
    """
    from pathlib import Path
    
    data_path = Path("/data")
    model_path = Path("/models")
    
    datasets = []
    if data_path.exists():
        datasets = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    models = []
    if model_path.exists():
        models = [m.name for m in model_path.iterdir() if m.is_dir()]
    
    return {
        "datasets": datasets,
        "models": models,
        "status": "success"
    }

@app.function(
    volumes={"/data": data_volume},
    timeout=600,
)
def list_volume_path(path_to_list: str):
    """
    Modalボリューム内の指定されたパスの内容をリストアップ
    """
    from pathlib import Path
    
    target_path = Path(path_to_list)
    
    if not target_path.is_dir():
        return {"status": "error", "message": f"Path is not a directory or does not exist: {path_to_list}"}
        
    contents = []
    for item in target_path.iterdir():
        contents.append(str(item))
        
    return {"status": "success", "path": path_to_list, "contents": contents}


@app.function(
    volumes={"/data": data_volume},
    timeout=600,
)
def read_text_error_log(dataset_name: str):
    """
    指定されたデータセットのtext_error.logファイルを読み込む関数
    """
    from pathlib import Path
    
    config_path = Path(f"/data/Data/{dataset_name}/text_error.log")
    
    if not config_path.exists():
        return {"status": "error", "message": f"text_error.log not found for dataset {dataset_name}"}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "status": "success",
            "log_path": str(config_path),
            "content": content,
            "line_count": len(content.splitlines()) if content else 0
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to read text_error.log: {str(e)}"}

@app.function(
    volumes={"/data": data_volume},
    timeout=600,
)
def read_config_file(dataset_name: str):
    """
    指定されたデータセットのconfig.jsonファイルを読み込んで内容を表示
    
    Args:
        dataset_name: データセット名
    """
    from pathlib import Path
    import json
    
    config_path = Path("/data/Data") / dataset_name / "config.json"
    
    if not config_path.exists():
        return {
            "status": "error", 
            "message": f"config.json not found at {config_path}",
            "dataset": dataset_name
        }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = json.load(f)
        
        # 話者情報を抽出
        speaker_info = {}
        if "data" in config_content:
            data_config = config_content["data"]
            speaker_info["n_speakers"] = data_config.get("n_speakers", "NOT_SET")
            speaker_info["spk2id"] = data_config.get("spk2id", "NOT_SET")
            speaker_info["training_files"] = data_config.get("training_files", "NOT_SET")
            speaker_info["validation_files"] = data_config.get("validation_files", "NOT_SET")
        
        return {
            "status": "success",
            "dataset": dataset_name,
            "config_path": str(config_path),
            "speaker_info": speaker_info,
            "full_config": config_content
        }
        
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": f"Invalid JSON in config.json: {str(e)}",
            "dataset": dataset_name,
            "config_path": str(config_path)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading config.json: {str(e)}",
            "dataset": dataset_name,
            "config_path": str(config_path)
        }


# コマンドライン実行用
@app.local_entrypoint()
def main(
    action: str = "train",
    dataset_name: str = "default",
    config_path: str = "configs/config.json",
    epochs: int = 1000,
    batch_size: int = 2,
    learning_rate: float = 0.0001,
    dataset_url: str = None,
    use_jp_extra: bool = True,
    resume: bool = False,
    normalize: bool = True, # 前処理用引数
    trim: bool = True, # 前処理用引数
    num_processes: int = 4, # 前処理用引数
    path_to_list: str = "", # list_volume_path用引数
):
    """
    Modal学習システムのメインエントリーポイント
    
    使用例:
    # ローカルデータセットをアップロード
    modal run modal_training.py --action=upload-local --dataset-name=my_voice
    
    # URLからデータセットをアップロード
    modal run modal_training.py --action=upload --dataset-name=my_voice --dataset-url=https://example.com/dataset.zip
    
    # データセットの前処理
    modal run modal_training.py --action=preprocess --dataset-name=my_voice
    
    # 学習開始
    modal run modal_training.py --action=train --dataset-name=my_voice --epochs=500
    
    # モデルダウンロード
    modal run modal_training.py --action=download --dataset-name=my_voice
    
    # リソース確認
    modal run modal_training.py --action=list

    # ボリューム内のパスをリスト
    modal run modal_training.py --action=list-path --path-to-list=/data/Data
    
    # データセットのconfig.jsonを読み込み
    modal run modal_training.py --action=read-config --dataset-name=大蔵衣遠
    """
    
    if action == "train":
        print(f"🚀 Starting training for dataset: {dataset_name}")
        result = train_model.remote(
            config_path=config_path,
            dataset_name=dataset_name,
            epochs=10, # テスト用にエポック数を10に設定
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_jp_extra=use_jp_extra,
            resume_training=resume
        )
        print(f"Training result: {result}")
        
    elif action == "preprocess":
        print(f"⚙️ Starting preprocessing for dataset: {dataset_name}")
        result = preprocess_dataset.remote(
            dataset_name=dataset_name,
            normalize=normalize,
            trim=trim,
            num_processes=num_processes,
        )
        print(f"Preprocessing result: {result}")
        
    elif action == "upload":
        print(f"📤 Uploading dataset: {dataset_name}")
        result = upload_dataset.remote(
            dataset_name=dataset_name,
            dataset_archive_url=dataset_url
        )
        print(f"Upload result: {result}")
        
    elif action == "download":
        print(f"📦 Downloading model: {dataset_name}")
        result = download_model.remote(dataset_name=dataset_name)
        
        if result["status"] == "success":
            # ローカルにファイルを保存
            filename = result["filename"]
            with open(filename, 'wb') as f:
                f.write(result["zip_content"])
            print(f"✅ Model downloaded: {filename}")
        else:
            print(f"❌ Download failed: {result['message']}")
            
    elif action == "list":
        result = list_datasets.remote()
        print(f"📋 Available resources:")
        print(f"  Datasets: {result['datasets']}")
        print(f"  Models: {result['models']}")
        
    elif action == "list-path":
        print(f"🔍 Listing contents of: {path_to_list}")
        result = list_volume_path.remote(path_to_list=path_to_list)
        print(f"List result: {result}")
        
    elif action == "read-config":
        print(f"📄 Reading config.json for dataset: {dataset_name}")
        result = read_config_file.remote(dataset_name=dataset_name)
        
        if result["status"] == "success":
            print(f"✅ Config file found at: {result['config_path']}")
            print(f"📋 Speaker Information:")
            for key, value in result["speaker_info"].items():
                print(f"  {key}: {value}")
            
            print(f"\n📄 Full Config Content:")
            print(json.dumps(result["full_config"], indent=2, ensure_ascii=False))
        else:
            print(f"❌ Error reading config: {result['message']}")
            
    elif action == "read-error-log":
        print(f"📋 Reading text_error.log for dataset: {dataset_name}")
        result = read_text_error_log.remote(dataset_name=dataset_name)
        
        if result["status"] == "success":
            print(f"✅ Error log found at: {result['log_path']}")
            print(f"📋 Log contains {result['line_count']} lines")
            print(f"\n📄 Error Log Content:")
            print(result["content"])
        else:
            print(f"❌ Error reading log: {result['message']}")
        
    elif action == "upload-local":
        print(f"📤 Uploading local dataset: {dataset_name}")
        
        # ローカルのデータセットディレクトリをチェック
        local_dataset_path = Path(f"Data/{dataset_name}")
        if not local_dataset_path.exists():
            print(f"❌ Local dataset not found: {local_dataset_path}")
            return
        
        # ファイルを読み込んでBase64エンコード
        dataset_files = {}
        for file_path in local_dataset_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dataset_path)
                
                if file_path.suffix.lower() in ['.txt', '.list', '.json', '.yml', '.yaml']:
                    # テキストファイルの場合
                    with open(file_path, 'r', encoding='utf-8') as f:
                        dataset_files[str(relative_path)] = f.read()
                else:
                    # バイナリファイルの場合（base64エンコード済み）
                    import base64
                    with open(file_path, 'rb') as f:
                        file_data = base64.b64encode(f.read()).decode('utf-8')
                        dataset_files[str(relative_path)] = file_data
        
        print(f"📁 Prepared {len(dataset_files)} files for upload")
        
        # データセットをアップロード
        result = upload_dataset_direct.remote(
            dataset_name=dataset_name,
            dataset_files=dataset_files
        )
        print(f"Upload result: {result}")
        
    else:
        print(f"❌ Unknown action: {action}")
        print("Available actions: train, upload-local, upload, download, list, preprocess, list-path, read-config, read-error-log")


if __name__ == "__main__":
    # 直接実行時のサンプル
    print("Style-Bert-Vits2 Modal Training System")
    print("Use: modal run modal_training.py --help for usage information")