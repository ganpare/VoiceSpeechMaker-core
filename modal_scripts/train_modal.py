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

# Dockerイメージ定義
image = modal.Image.from_dockerfile("Dockerfile.simple")

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
    
    # 学習実行
    print("\n=== LUNA0712 Continuation Training Start ===")
    try:
        # train_ms_jp_extra.py を LUNA0712 設定で実行
        result = subprocess.run([
            "python", "train_ms_jp_extra.py",
            "--config", "/workspace/Data/LUNA0712/config.json",
            "--model", "/workspace/Data/LUNA0712",
            "--assets_root", "/workspace/model_assets",
            "--no_progress_bar",  # Modal環境では進捗バーを無効化
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

@app.local_entrypoint()
def main():
    """メイン関数"""
    print("=== Modal LUNA0712 Training Job ===")
    success = train_luna0712.remote()
    
    if success:
        print("✅ LUNA0712 継続学習完了")
    else:
        print("❌ LUNA0712 継続学習失敗")

if __name__ == "__main__":
    main()