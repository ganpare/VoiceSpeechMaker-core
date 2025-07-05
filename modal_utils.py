"""
Modal用の設定とユーティリティ関数
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ModalConfig:
    """Modal学習設定クラス"""
    
    def __init__(self):
        self.default_config = {
            "gpu": {
                "type": "A100",  # A100, T4, RTX4090等
                "count": 1,
                "memory_gb": 40
            },
            "compute": {
                "cpu_count": 8,
                "memory_gb": 32,
                "timeout_hours": 24
            },
            "training": {
                "epochs": 1000,
                "batch_size": 2,
                "learning_rate": 0.0001,
                "mixed_precision": True,
                "gradient_checkpointing": False,
                "save_interval": 1000,
                "eval_interval": 1000
            },
            "data": {
                "preprocessing": {
                    "resample_rate": 44100,
                    "trim_silence": True,
                    "normalize_audio": True
                },
                "augmentation": {
                    "enable": False,
                    "pitch_shift": False,
                    "time_stretch": False,
                    "noise_injection": False
                }
            },
            "model": {
                "use_jp_extra": True,
                "use_wavlm_discriminator": True,
                "use_duration_discriminator": True,
                "freeze_bert": False
            },
            "optimization": {
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 5.0,
                "warmup_steps": 1000,
                "lr_scheduler": "exponential"
            }
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
            
            # デフォルト設定とマージ
            config = self.default_config.copy()
            config.update(custom_config)
            return config
        else:
            return self.default_config
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """設定ファイルを保存"""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def get_gpu_config(self, gpu_type: str = "A100", count: int = 1):
        """GPU設定を取得"""
        gpu_configs = {
            "A100": {"memory": 40, "compute_capability": "8.0"},
            "V100": {"memory": 32, "compute_capability": "7.0"},
            "T4": {"memory": 16, "compute_capability": "7.5"},
            "RTX4090": {"memory": 24, "compute_capability": "8.9"},
            "A10G": {"memory": 24, "compute_capability": "8.6"}
        }
        
        return {
            "type": gpu_type,
            "count": count,
            **gpu_configs.get(gpu_type, gpu_configs["A100"])
        }
    
    def estimate_training_cost(self, 
                             gpu_type: str = "A100", 
                             gpu_count: int = 1,
                             training_hours: float = 10.0) -> Dict[str, float]:
        """学習コストを推定"""
        
        # Modal GPU料金 (USD/hour, 2024年概算)
        gpu_costs = {
            "A100": 3.20,
            "V100": 1.60,
            "T4": 0.60,
            "RTX4090": 1.20,
            "A10G": 1.00
        }
        
        gpu_cost_per_hour = gpu_costs.get(gpu_type, gpu_costs["A100"])
        total_gpu_cost = gpu_cost_per_hour * gpu_count * training_hours
        
        # CPU・メモリコスト（概算）
        compute_cost = 0.10 * training_hours
        
        # ストレージコスト（概算）
        storage_cost = 0.05 * training_hours
        
        total_cost = total_gpu_cost + compute_cost + storage_cost
        
        return {
            "gpu_cost": total_gpu_cost,
            "compute_cost": compute_cost,
            "storage_cost": storage_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }


class DatasetValidator:
    """データセットの検証クラス"""
    
    @staticmethod
    def validate_dataset_structure(dataset_path: Path) -> Dict[str, Any]:
        """データセット構造を検証"""
        required_files = ["esd.list"]
        required_dirs = ["wavs"]
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # 必須ファイルの確認
        for file_name in required_files:
            file_path = dataset_path / file_name
            if not file_path.exists():
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required file: {file_name}")
        
        # 必須ディレクトリの確認
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required directory: {dir_name}")
            elif dir_name == "wavs":
                # 音声ファイル数をカウント
                audio_files = list(dir_path.glob("*.wav"))
                validation_result["stats"]["audio_files"] = len(audio_files)
                
                if len(audio_files) == 0:
                    validation_result["valid"] = False
                    validation_result["errors"].append("No audio files found in wavs directory")
                elif len(audio_files) < 100:
                    validation_result["warnings"].append(f"Only {len(audio_files)} audio files found. Recommend at least 100 files for good quality.")
        
        # esd.listファイルの検証
        esd_path = dataset_path / "esd.list"
        if esd_path.exists():
            try:
                with open(esd_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                validation_result["stats"]["transcript_lines"] = len(lines)
                
                # サンプル行の検証
                if lines:
                    sample_line = lines[0].strip()
                    parts = sample_line.split("|")
                    if len(parts) < 3:
                        validation_result["warnings"].append("esd.list format might be incorrect. Expected format: filename|speaker|text")
            except Exception as e:
                validation_result["errors"].append(f"Error reading esd.list: {str(e)}")
        
        return validation_result
    
    @staticmethod
    def check_audio_quality(dataset_path: Path, sample_count: int = 10) -> Dict[str, Any]:
        """音声ファイルの品質をチェック"""
        import librosa
        import numpy as np
        
        wavs_path = dataset_path / "wavs"
        audio_files = list(wavs_path.glob("*.wav"))
        
        if not audio_files:
            return {"error": "No audio files found"}
        
        sample_files = np.random.choice(audio_files, min(sample_count, len(audio_files)), replace=False)
        
        quality_stats = {
            "sample_rates": [],
            "durations": [],
            "channels": [],
            "bit_depths": [],
            "issues": []
        }
        
        for audio_file in sample_files:
            try:
                y, sr = librosa.load(audio_file, sr=None)
                duration = len(y) / sr
                
                quality_stats["sample_rates"].append(sr)
                quality_stats["durations"].append(duration)
                quality_stats["channels"].append(1 if y.ndim == 1 else y.shape[0])
                
                # 品質チェック
                if duration < 0.5:
                    quality_stats["issues"].append(f"{audio_file.name}: Too short ({duration:.2f}s)")
                elif duration > 30.0:
                    quality_stats["issues"].append(f"{audio_file.name}: Too long ({duration:.2f}s)")
                
                if sr not in [22050, 44100, 48000]:
                    quality_stats["issues"].append(f"{audio_file.name}: Unusual sample rate ({sr}Hz)")
                
            except Exception as e:
                quality_stats["issues"].append(f"{audio_file.name}: Error loading - {str(e)}")
        
        # 統計情報
        if quality_stats["sample_rates"]:
            quality_stats["avg_sample_rate"] = np.mean(quality_stats["sample_rates"])
            quality_stats["avg_duration"] = np.mean(quality_stats["durations"])
            quality_stats["total_files_checked"] = len(sample_files)
        
        return quality_stats


def create_modal_config_template(output_path: str = "modal_config.json"):
    """Modal設定テンプレートを作成"""
    config = ModalConfig()
    config.save_config(config.default_config, output_path)
    print(f"Modal configuration template created: {output_path}")


def estimate_training_time(dataset_size: int, 
                         epochs: int = 1000, 
                         batch_size: int = 2,
                         gpu_type: str = "A100") -> Dict[str, float]:
    """学習時間を推定"""
    
    # GPU別の処理速度（steps/second, 概算）
    gpu_speeds = {
        "A100": 2.0,
        "V100": 1.2,
        "T4": 0.6,
        "RTX4090": 1.5,
        "A10G": 1.0
    }
    
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * epochs
    
    speed = gpu_speeds.get(gpu_type, gpu_speeds["A100"])
    training_time_seconds = total_steps / speed
    
    return {
        "estimated_hours": training_time_seconds / 3600,
        "estimated_days": training_time_seconds / (3600 * 24),
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "gpu_speed_steps_per_sec": speed
    }
