#!/usr/bin/env python3
"""
Modal Training Script
学習実行（何度でも実行可能）
"""

import modal

# Modal App定義
app = modal.App("voice-speech-maker-train")

# 既存ボリュームを参照
data_volume = modal.Volume.from_name("voice_speech_maker_data")
models_volume = modal.Volume.from_name("voice_speech_maker_models")
logs_volume = modal.Volume.from_name("voice_speech_maker_logs")

# 軽量化：必要最小限のパッケージのみ
image = (
    modal.Image.from_dockerfile("Dockerfile.simple")
    .pip_install([
        # 基本ライブラリ（高速インストール）
        "torch>=2.1.0",
        "torchaudio>=2.1.0", 
        "transformers==4.48.0",
        "librosa==0.9.2",
        "numpy<2.0.0",
        "safetensors>=0.3.0",
        "tqdm>=4.64.0",
        "loguru>=0.7.0",
    ])
    .env({
        # CUDA設定
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        
        # パス設定
        "PYTHONPATH": "/workspace",
        
        # キャッシュ設定 (メモリ効率化)
        "TRANSFORMERS_CACHE": "/tmp/transformers_cache",
        "HF_HOME": "/tmp/huggingface_cache",
        "NUMBA_CACHE_DIR": "/tmp/numba_cache",
    })
    .run_commands([
        # システム依存関係 (Style-Bert-VITS2音声処理用)
        "apt-get update && apt-get install -y git ffmpeg libsndfile1-dev build-essential python3-dev",
        
        # 基本環境確認
        "python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'",
    ])
)

@app.function(
    image=image,
    volumes={
        "/workspace/Data": data_volume,
        "/workspace/model_assets": models_volume,
        "/workspace/logs": logs_volume,
    },
    gpu="A100",  # GPU種類を指定
    timeout=7200,  # 2時間タイムアウト（学習時間を考慮）
    memory=32768,  # 32GB RAM（学習には大容量メモリが必要）
)
def train_luna0712():
    """LUNA0712継続学習実行関数"""
    import subprocess
    import os
    
    print("=== Modal LUNA0712 Training Start ===")
    
    # 作業ディレクトリ確認
    print(f"Current directory: {os.getcwd()}")
    
    # 現在のリポジトリをGitHubからクローン
    print("\\n=== Cloning current repository ===")
    subprocess.run([
        "git", "clone", "https://github.com/ganpare/VoiceSpeechMaker-core.git", 
        "/tmp/voicespeechmaker"
    ], check=True)
    
    # 必要なスクリプトファイルのみをコピー（Data, model_assetsは除外）
    subprocess.run("cp -r /tmp/voicespeechmaker/style_bert_vits2 /workspace/", shell=True, check=True)
    subprocess.run("cp /tmp/voicespeechmaker/*.py /workspace/", shell=True, check=True)
    subprocess.run("cp /tmp/voicespeechmaker/*.yml /workspace/", shell=True, check=True)
    subprocess.run("cp /tmp/voicespeechmaker/requirements.txt /workspace/", shell=True, check=True)
    subprocess.run("cp -r /tmp/voicespeechmaker/configs /workspace/", shell=True, check=True)
    subprocess.run("cp -r /tmp/voicespeechmaker/dict_data /workspace/", shell=True, check=True)
    subprocess.run("cp -r /tmp/voicespeechmaker/slm /workspace/", shell=True, check=True)
    
    # default_style.pyのエラーを修正（空配列対応）
    print("\\n=== Fixing default_style.py ===")
    with open("/workspace/default_style.py", "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        "x = np.concatenate(embs, axis=0)  # (N, 256)",
        "x = np.concatenate(embs, axis=0) if embs else np.array([])  # (N, 256)"
    )
    with open("/workspace/default_style.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ default_style.py fixed")
    
    # 追加の必要パッケージをインストール
    print("\\n=== Installing additional packages ===")
    subprocess.run([
        "pip", "install", 
        "pyopenjtalk-mod>=0.3.0",
        "jaconv>=0.3.4", 
        "pyworld-prebuilt>=0.3.0",
        "accelerate>=0.24.0",
        "protobuf==4.25.0",
        "tensorboard>=2.10.0",
        "onnxruntime-gpu>=1.16.0",
        "pydantic==1.10.12"
    ], check=True)
    
    # WavLM-base-plusモデルをHugging Faceからダウンロード
    print("\\n=== Downloading WavLM model from Hugging Face ===")
    from transformers import AutoModel, AutoConfig
    
    # Hugging FaceからWavLMモデルをダウンロードしてローカルに保存
    model_name = "microsoft/wavlm-base-plus"
    local_model_path = "/workspace/slm/wavlm-base-plus"
    os.makedirs(local_model_path, exist_ok=True)
    
    # モデル設定とモデルをダウンロード
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # ローカルに保存
    config.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print(f"✅ WavLM model saved to {local_model_path}")
    
    # データ確認
    print("\n=== Data Check ===")
    if os.path.exists("/workspace/Data/LUNA0712"):
        print("✅ LUNA0712 data directory found")
        print("Config files:")
        config_files = ["config.json", "esd.list", "train.list", "val.list"]
        for file in config_files:
            file_path = f"/workspace/Data/LUNA0712/{file}"
            if os.path.exists(file_path):
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} missing")
        
        # 音声ファイル確認
        raw_dir = "/workspace/Data/LUNA0712/raw"
        if os.path.exists(raw_dir):
            audio_count = len([f for f in os.listdir(raw_dir) if f.endswith('.wav')])
            print(f"  ✅ Audio files: {audio_count} wav files")
        else:
            print("  ❌ raw/ directory not found")
            return False
    else:
        print("❌ LUNA0712 data directory not found")
        return False
    
    # 学習済みモデル確認
    print("\n=== Model Check ===")
    model_dir = "/workspace/model_assets/LUNA0712"
    if os.path.exists(model_dir):
        model_files = os.listdir(model_dir)
        print("Available models:")
        for file in model_files:
            if file.endswith('.safetensors'):
                print(f"  ✅ {file}")
    else:
        print("❌ Pre-trained models not found")
        return False
    
    # スタイルベクトルファイルをデータディレクトリにコピー（エラー回避）
    print("\n=== Copying style vectors ===")
    style_vec_src = "/workspace/model_assets/LUNA0712/style_vectors.npy"
    style_vec_dst = "/workspace/Data/LUNA0712/style_vectors.npy"
    if os.path.exists(style_vec_src):
        subprocess.run(["cp", style_vec_src, style_vec_dst], check=True)
        print("✅ Style vectors copied")
    
    # スタイルベクトル生成（デフォルトベクトルを作成）
    print("\n=== Generating missing style vectors ===")
    try:
        import numpy as np
        
        # デフォルトスタイルベクトルを読み込み
        if os.path.exists("/workspace/Data/LUNA0712/style_vectors.npy"):
            default_style = np.load("/workspace/Data/LUNA0712/style_vectors.npy")
            print(f"Default style vector shape: {default_style.shape}")
        else:
            # デフォルトスタイルベクトル（256次元）を作成
            default_style = np.zeros(256, dtype=np.float32)
            print("Created zero default style vector")
        
        # train.listから音声ファイルリストを取得
        train_list_path = "/workspace/Data/LUNA0712/train.list"
        val_list_path = "/workspace/Data/LUNA0712/val.list"
        
        generated_count = 0
        for list_file in [train_list_path, val_list_path]:
            if os.path.exists(list_file):
                with open(list_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) > 0:
                            audio_path = parts[0]
                            style_vector_path = f"{audio_path}.npy"
                            
                            # スタイルベクトルファイルが存在しない場合のみ作成
                            if not os.path.exists(style_vector_path):
                                os.makedirs(os.path.dirname(style_vector_path), exist_ok=True)
                                np.save(style_vector_path, default_style)
                                generated_count += 1
        
        print(f"✅ Generated {generated_count} missing style vector files")
            
    except Exception as e:
        print(f"⚠️ Style vector generation error: {e}, continuing with training")
    
    # config.jsonのパス区切り文字を修正（Windows -> Linux + 継続学習設定）
    print("\n=== Fixing config.json paths and model settings ===")
    config_path = "/workspace/Data/LUNA0712/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_content = f.read()
    config_content = config_content.replace("Data\\\\LUNA0712\\\\", "Data/LUNA0712/")
    config_content = config_content.replace("Data\\LUNA0712\\", "Data/LUNA0712/")
    
    # 継続学習用の設定を追加
    import json
    config_data = json.loads(config_content)
    config_data["model_name"] = "LUNA0712"  # モデル名を明示的に設定
    config_content = json.dumps(config_data, indent=2, ensure_ascii=False)
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    print("✅ Config paths and model name fixed")
    
    # train.listとval.listのパス修正（Windows -> Linux + wavs -> raw）
    print("\n=== Fixing train/val list paths ===")
    for list_file in ["train.list", "val.list"]:
        list_path = f"/workspace/Data/LUNA0712/{list_file}"
        if os.path.exists(list_path):
            with open(list_path, "r", encoding="utf-8") as f:
                list_content = f.read()
            # Windowsパス区切り文字とディレクトリ名を修正
            list_content = list_content.replace("Data\\LUNA0712\\wavs\\", "Data/LUNA0712/raw/")
            list_content = list_content.replace("Data\\\\LUNA0712\\\\wavs\\\\", "Data/LUNA0712/raw/")
            # 大文字小文字の修正（A -> a）
            list_content = list_content.replace("v_lun0019A.wav", "v_lun0019a.wav")
            with open(list_path, "w", encoding="utf-8") as f:
                f.write(list_content)
            print(f"✅ {list_file} paths fixed")
    
    # BERTファイル生成（必要に応じて）
    print("\n=== Generating BERT features ===")
    try:
        result = subprocess.run([
            "python", "bert_gen.py",
            "--config", "/workspace/Data/LUNA0712/config.json"
        ], 
        cwd="/workspace",
        capture_output=True,
        text=True,
        timeout=300  # 5分タイムアウト
        )
        
        if result.returncode == 0:
            print("✅ BERT features generated successfully")
        else:
            print("⚠️ BERT feature generation failed, continuing anyway")
            print("Error:", result.stderr[-300:])
            
    except Exception as e:
        print(f"⚠️ BERT generation error: {e}, continuing anyway")
    
    # 学習実行
    print("\n=== LUNA0712 Continuation Training Start ===")
    try:
        # 継続学習の準備（ローカルデータ構造を活用）
        model_dir = "/workspace/model_assets/LUNA0712"
        
        # 利用可能なモデルを確認
        print("\\n=== Available Models for Continuation ===")
        
        # 1. 最新チェックポイント確認
        checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors') and f.startswith('LUNA0712_e')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_e')[1].split('_')[0]))
            print(f"📍 Latest checkpoint: {latest_checkpoint}")
        
        # 2. PyTorchモデル確認
        pytorch_models_dir = os.path.join(model_dir, "models")
        if os.path.exists(pytorch_models_dir):
            pytorch_files = [f for f in os.listdir(pytorch_models_dir) if f.endswith('.pth')]
            print(f"🔥 PyTorch models: {pytorch_files}")
        
        # 3. バックアップモデル確認  
        backup_models_dir = os.path.join(model_dir, "models_backup")
        if os.path.exists(backup_models_dir):
            backup_files = [f for f in os.listdir(backup_models_dir) if f.endswith('.safetensors')]
            print(f"💾 Backup models: {backup_files}")
        
        # 継続学習実行（Style-Bert-VITS2が自動でモデルを検出）
        print(f"🚀 Ready for continuation training from model directory")
        
        # train_ms_jp_extra.py を継続学習設定で実行
        result = subprocess.run([
            "python", "train_ms_jp_extra.py",
            "--config", "/workspace/Data/LUNA0712/config.json",
            "--model", "/workspace/model_assets/LUNA0712",  # 既存モデルディレクトリ（継続学習）
            "--assets_root", "/workspace/model_assets",
            # "--no_progress_bar",  # 進捗バーを有効化（Modal web画面で確認可能）
            # "--speedup",  # 高速化オプション（必要に応じて）
        ], 
        cwd="/workspace",
        capture_output=True,
        text=True,
        timeout=7200  # 2時間タイムアウト
        )
        
        if result.returncode == 0:
            print("✅ LUNA0712 training completed successfully")
            print("Training output:")
            print(result.stdout[-2000:])  # 最後の2000文字のみ表示
        else:
            print("❌ LUNA0712 training failed")
            print("Error output:")
            print(result.stderr[-2000:])  # 最後の2000文字のみ表示
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out after 2 hours")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False
    
    print("\n=== LUNA0712 Training Complete ===")
    return True

# Web画面から直接実行推奨
# Modal Web UI: https://modal.com/apps/ganpare/main/deployed/voice-speech-maker-train

@app.local_entrypoint()
def main():
    """ローカルエントリーポイント（オプション）"""
    result = train_luna0712.remote()
    print(f"Training result: {result}")