#!/usr/bin/env python3
"""
Modal Volume Setup Script
ボリューム作成とデータアップロード（初回のみ実行）
"""

import modal

# Modal App定義
app = modal.App("voice-speech-maker-setup")

def main():
    """3つのボリューム作成とデータアップロード"""
    
    print("=== Modal Volume Setup ===")
    
    # 作成するボリューム一覧
    volumes_config = {
        "voice_speech_maker_data": {"path": "./Data", "description": "学習データ"},
        "voice_speech_maker_models": {"path": None, "description": "モデル出力"},
        "voice_speech_maker_logs": {"path": None, "description": "ログ"}
    }
    
    created_volumes = {}
    
    # 1. ボリューム作成
    print("1. ボリューム作成中...")
    for volume_name, config in volumes_config.items():
        try:
            volume = modal.Volume.from_name(volume_name, create_if_missing=True)
            print(f"✅ {volume_name} 作成/参照完了 ({config['description']})")
            created_volumes[volume_name] = volume
        except Exception as e:
            print(f"❌ {volume_name} 作成/参照失敗: {e}")
            return False
    
    # 2. データアップロード（voice_speech_maker_dataのみ）
    print("\n2. データアップロード中...")
    data_volume = created_volumes["voice_speech_maker_data"]
    try:
        # CLIでアップロードする方法を案内
        print("ℹ️ データアップロードはCLIで実行してください:")
        print(f"modal volume put voice_speech_maker_data ./Data /")
        print("✅ ボリューム準備完了")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False
    
    print(f"\n=== セットアップ完了 ===")
    print("作成されたボリューム:")
    for volume_name, config in volumes_config.items():
        print(f"- {volume_name}: {config['description']}")
    print("\n次回からは train_modal.py で学習を実行してください")
    return True

if __name__ == "__main__":
    with app.run():
        success = main()
        if success:
            print("✅ セットアップ成功")
        else:
            print("❌ セットアップ失敗")