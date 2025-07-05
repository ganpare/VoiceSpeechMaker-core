#!/usr/bin/env python3
"""
Modal学習用のシンプルなアップローダー
ローカルのデータセットを簡単にModalにアップロードする
"""

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Dict

def prepare_dataset_for_upload(dataset_path: Path) -> Dict[str, str]:
    """
    ローカルデータセットをModal用に準備
    
    Args:
        dataset_path: データセットのパス
        
    Returns:
        dict: ファイル名とコンテンツの辞書
    """
    dataset_files = {}
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"📂 Preparing dataset: {dataset_path}")
    
    # サポートされているファイル拡張子
    text_extensions = {'.txt', '.list', '.json', '.yml', '.yaml', '.py', '.md'}
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    file_count = 0
    total_size = 0
    
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(dataset_path)
            file_size = file_path.stat().st_size
            total_size += file_size
            
            print(f"  📄 Processing: {relative_path} ({file_size:,} bytes)")
            
            if file_path.suffix.lower() in text_extensions:
                # テキストファイル
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        dataset_files[str(relative_path)] = f.read()
                except UnicodeDecodeError:
                    # UTF-8で読めない場合はバイナリとして処理
                    with open(file_path, 'rb') as f:
                        file_data = base64.b64encode(f.read()).decode('utf-8')
                        dataset_files[str(relative_path)] = file_data
            
            elif file_path.suffix.lower() in audio_extensions or file_size > 0:
                # 音声ファイルやその他のバイナリファイル
                with open(file_path, 'rb') as f:
                    file_data = base64.b64encode(f.read()).decode('utf-8')
                    dataset_files[str(relative_path)] = file_data
            
            file_count += 1
            
            # ファイルサイズ制限（100MB以上のファイルは警告）
            if file_size > 100 * 1024 * 1024:
                print(f"  ⚠️  Large file detected: {relative_path} ({file_size:,} bytes)")
    
    print(f"✅ Prepared {file_count} files ({total_size:,} bytes total)")
    
    # 総サイズ制限の確認（1GB）
    if total_size > 1024 * 1024 * 1024:
        print(f"⚠️  Large dataset detected: {total_size:,} bytes")
        print("Consider splitting the dataset or using external storage for very large files")
    
    return dataset_files

def upload_dataset_to_modal(dataset_name: str, dataset_files: Dict[str, str]):
    """
    データセットをModalにアップロード
    """
    try:
        # modal_training.pyをインポート
        sys.path.insert(0, '.')
        from modal_training import upload_dataset_direct
        
        print(f"🚀 Uploading dataset '{dataset_name}' to Modal...")
        
        result = upload_dataset_direct.remote(
            dataset_name=dataset_name,
            dataset_files=dataset_files
        )
        
        print(f"📊 Upload result: {result}")
        return result
        
    except ImportError as e:
        print(f"❌ Failed to import modal_training: {e}")
        print("Make sure you're running this from the project root directory")
        return None
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def validate_dataset_structure(dataset_path: Path) -> bool:
    """
    データセット構造を検証
    """
    print("🔍 Validating dataset structure...")
    
    required_files = ["esd.list"]
    required_dirs = ["wavs"]
    
    errors = []
    warnings = []
    
    # 必須ファイルの確認
    for file_name in required_files:
        file_path = dataset_path / file_name
        if not file_path.exists():
            errors.append(f"Missing required file: {file_name}")
    
    # 必須ディレクトリの確認
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            errors.append(f"Missing required directory: {dir_name}")
        elif dir_name == "wavs":
            # 音声ファイル数をカウント
            audio_files = list(dir_path.glob("*.wav"))
            if len(audio_files) == 0:
                errors.append("No audio files found in wavs directory")
            elif len(audio_files) < 50:
                warnings.append(f"Only {len(audio_files)} audio files found. Recommend at least 50 files for good quality.")
            else:
                print(f"  ✅ Found {len(audio_files)} audio files")
    
    # エラーがある場合
    if errors:
        print("❌ Dataset validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    # 警告がある場合
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✅ Dataset structure validation passed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Modal用データセットアップローダー")
    parser.add_argument("dataset_name", help="データセット名")
    parser.add_argument("--dataset-path", type=str, help="データセットのパス（デフォルト: Data/{dataset_name}）")
    parser.add_argument("--validate-only", action="store_true", help="検証のみ実行（アップロードしない）")
    parser.add_argument("--skip-validation", action="store_true", help="検証をスキップ")
    
    args = parser.parse_args()
    
    # データセットパスを決定
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = Path("Data") / args.dataset_name
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    # データセット構造を検証
    if not args.skip_validation:
        if not validate_dataset_structure(dataset_path):
            sys.exit(1)
    
    if args.validate_only:
        print("✅ Validation completed (upload skipped)")
        return
    
    try:
        # データセットを準備
        dataset_files = prepare_dataset_for_upload(dataset_path)
        
        # Modalにアップロード
        result = upload_dataset_to_modal(args.dataset_name, dataset_files)
        
        if result and result.get("status") == "success":
            print(f"🎉 Dataset '{args.dataset_name}' uploaded successfully!")
            print(f"📁 Files uploaded: {result.get('files_count', 'unknown')}")
            print(f"📍 Modal path: {result.get('path', 'unknown')}")
            print("\n🚀 Next steps:")
            print(f"1. Start training: modal run modal_training.py --action=train --dataset-name={args.dataset_name}")
            print(f"2. Monitor progress: Check Modal dashboard")
        else:
            print("❌ Upload failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Upload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
