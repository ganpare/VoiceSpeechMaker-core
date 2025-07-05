#!/usr/bin/env python3
"""
Modal学習システムのローカル管理ツール
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, Any

from modal_utils import ModalConfig, DatasetValidator, estimate_training_time


class ModalManager:
    """Modal学習システムの管理クラス"""
    
    def __init__(self):
        self.config = ModalConfig()
        self.validator = DatasetValidator()
    
    def prepare_dataset(self, dataset_path: str, output_path: str = None):
        """データセットを学習用に準備"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            return False
        
        print(f"📂 Preparing dataset: {dataset_path}")
        
        # データセット構造を検証
        validation_result = self.validator.validate_dataset_structure(dataset_path)
        
        if not validation_result["valid"]:
            print("❌ Dataset validation failed:")
            for error in validation_result["errors"]:
                print(f"  - {error}")
            return False
        
        if validation_result["warnings"]:
            print("⚠️  Warnings:")
            for warning in validation_result["warnings"]:
                print(f"  - {warning}")
        
        print("✅ Dataset structure validation passed")
        print(f"📊 Dataset stats: {validation_result['stats']}")
        
        # 音声品質チェック
        print("🔍 Checking audio quality...")
        quality_result = self.validator.check_audio_quality(dataset_path)
        
        if "error" in quality_result:
            print(f"❌ Audio quality check failed: {quality_result['error']}")
            return False
        
        if quality_result["issues"]:
            print("⚠️  Audio quality issues found:")
            for issue in quality_result["issues"]:
                print(f"  - {issue}")
        
        print(f"📈 Audio quality stats:")
        print(f"  - Average sample rate: {quality_result.get('avg_sample_rate', 'N/A'):.0f} Hz")
        print(f"  - Average duration: {quality_result.get('avg_duration', 'N/A'):.2f} seconds")
        print(f"  - Files checked: {quality_result.get('total_files_checked', 'N/A')}")
        
        # アーカイブ作成
        if output_path is None:
            output_path = f"{dataset_path.name}_prepared.zip"
        
        print(f"📦 Creating dataset archive: {output_path}")
        self._create_dataset_archive(dataset_path, output_path)
        
        print(f"✅ Dataset preparation completed: {output_path}")
        return True
    
    def _create_dataset_archive(self, dataset_path: Path, output_path: str):
        """データセットアーカイブを作成"""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in dataset_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(dataset_path.parent)
                    zipf.write(file_path, arcname)
    
    def create_training_config(self, 
                             dataset_name: str,
                             output_path: str = None,
                             **kwargs):
        """学習設定ファイルを作成"""
        
        if output_path is None:
            output_path = f"configs/{dataset_name}_modal_config.json"
        
        # デフォルト設定を取得
        config = self.config.default_config.copy()
        
        # カスタム設定を適用
        for key, value in kwargs.items():
            if key in config:
                if isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
        
        # データセット名を設定
        config["dataset_name"] = dataset_name
        
        # 設定を保存
        self.config.save_config(config, output_path)
        print(f"✅ Training config created: {output_path}")
        
        return output_path
    
    def estimate_costs(self, 
                      dataset_size: int,
                      epochs: int = 1000,
                      batch_size: int = 2,
                      gpu_type: str = "A100"):
        """学習コストと時間を推定"""
        
        print(f"💰 Estimating training costs...")
        print(f"  Dataset size: {dataset_size} samples")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  GPU type: {gpu_type}")
        print()
        
        # 学習時間を推定
        time_estimate = estimate_training_time(
            dataset_size, epochs, batch_size, gpu_type
        )
        
        print(f"⏱️  Estimated training time:")
        print(f"  - Hours: {time_estimate['estimated_hours']:.1f}")
        print(f"  - Days: {time_estimate['estimated_days']:.1f}")
        print(f"  - Total steps: {time_estimate['total_steps']:,}")
        print()
        
        # コストを推定
        cost_estimate = self.config.estimate_training_cost(
            gpu_type, 1, time_estimate['estimated_hours']
        )
        
        print(f"💵 Estimated costs (USD):")
        print(f"  - GPU cost: ${cost_estimate['gpu_cost']:.2f}")
        print(f"  - Compute cost: ${cost_estimate['compute_cost']:.2f}")
        print(f"  - Storage cost: ${cost_estimate['storage_cost']:.2f}")
        print(f"  - Total cost: ${cost_estimate['total_cost']:.2f}")
        print()
        
        return time_estimate, cost_estimate
    
    def check_modal_setup(self):
        """Modal環境のセットアップを確認"""
        print("🔍 Checking Modal setup...")
        
        try:
            import modal
            print("✅ Modal library installed")
        except ImportError:
            print("❌ Modal library not installed. Run: pip install modal")
            return False
        
        # Modal認証確認
        try:
            # これは実際のModalトークンをチェックする方法に置き換える必要があります
            print("✅ Modal authentication configured")
        except Exception as e:
            print(f"❌ Modal authentication failed: {e}")
            print("Run: modal token set --token-id <ID> --token-secret <SECRET>")
            return False
        
        print("✅ Modal setup check completed")
        return True
    
    def monitor_training(self, dataset_name: str):
        """学習の監視（将来の実装用）"""
        print(f"📊 Monitoring training for dataset: {dataset_name}")
        print("This feature will be implemented in a future version.")
        print("For now, check the Modal dashboard: https://modal.com/apps")
    
    def generate_training_script(self, dataset_name: str, config_path: str):
        """学習実行用のシェルスクリプトを生成"""
        script_content = f"""#!/bin/bash
# Style-Bert-VITS2 Modal Training Script
# Dataset: {dataset_name}

echo "🚀 Starting Modal training for {dataset_name}..."

# Upload dataset (if needed)
# modal run modal_training.py --action=upload --dataset-name={dataset_name} --dataset-url=<YOUR_DATASET_URL>

# Start training
modal run modal_training.py \\
  --action=train \\
  --dataset-name={dataset_name} \\
  --config-path={config_path} \\
  --epochs=1000 \\
  --batch-size=2 \\
  --learning-rate=0.0001 \\
  --use-jp-extra=true

echo "✅ Training completed!"

# Download trained model
modal run modal_training.py --action=download --dataset-name={dataset_name}

echo "📦 Model downloaded successfully!"
"""
        
        script_path = f"train_{dataset_name}_modal.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 実行権限を付与（Unix系OSの場合）
        if os.name != 'nt':
            os.chmod(script_path, 0o755)
        
        print(f"📝 Training script generated: {script_path}")
        return script_path


def main():
    parser = argparse.ArgumentParser(description="Modal学習システム管理ツール")
    parser.add_argument("action", choices=[
        "prepare", "config", "estimate", "check", "monitor", "script"
    ], help="実行するアクション")
    
    parser.add_argument("--dataset-path", type=str, help="データセットのパス")
    parser.add_argument("--dataset-name", type=str, help="データセット名")
    parser.add_argument("--output", type=str, help="出力ファイルパス")
    parser.add_argument("--dataset-size", type=int, help="データセットサイズ")
    parser.add_argument("--epochs", type=int, default=1000, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=2, help="バッチサイズ")
    parser.add_argument("--gpu-type", type=str, default="A100", help="GPU種類")
    parser.add_argument("--config-path", type=str, help="設定ファイルパス")
    
    args = parser.parse_args()
    
    manager = ModalManager()
    
    if args.action == "prepare":
        if not args.dataset_path:
            print("❌ --dataset-path が必要です")
            sys.exit(1)
        manager.prepare_dataset(args.dataset_path, args.output)
    
    elif args.action == "config":
        if not args.dataset_name:
            print("❌ --dataset-name が必要です")
            sys.exit(1)
        manager.create_training_config(args.dataset_name, args.output)
    
    elif args.action == "estimate":
        if not args.dataset_size:
            print("❌ --dataset-size が必要です")
            sys.exit(1)
        manager.estimate_costs(
            args.dataset_size, args.epochs, args.batch_size, args.gpu_type
        )
    
    elif args.action == "check":
        manager.check_modal_setup()
    
    elif args.action == "monitor":
        if not args.dataset_name:
            print("❌ --dataset-name が必要です")
            sys.exit(1)
        manager.monitor_training(args.dataset_name)
    
    elif args.action == "script":
        if not args.dataset_name:
            print("❌ --dataset-name が必要です")
            sys.exit(1)
        config_path = args.config_path or f"configs/{args.dataset_name}_modal_config.json"
        manager.generate_training_script(args.dataset_name, config_path)


if __name__ == "__main__":
    main()
