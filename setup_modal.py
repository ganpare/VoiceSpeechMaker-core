#!/usr/bin/env python3
"""
Modal学習システムのクイックセットアップスクリプト
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Python バージョンをチェック"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 以上が必要です")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_modal():
    """Modal をインストール"""
    print("📦 Installing Modal...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "modal"], check=True)
        print("✅ Modal installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Modal")
        return False

def setup_modal_auth():
    """Modal 認証をセットアップ"""
    print("\n🔑 Modal Authentication Setup")
    print("1. Visit https://modal.com/ and create an account")
    print("2. Go to https://modal.com/settings/tokens")
    print("3. Create a new token")
    print("4. Run the following command with your token:")
    print("   modal token set --token-id <YOUR_TOKEN_ID> --token-secret <YOUR_TOKEN_SECRET>")
    print("\nAfter setting up your token, run this script again to verify.")
    
    # Modal 認証確認
    try:
        result = subprocess.run([sys.executable, "-c", "import modal; print('OK')"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Modal authentication verified")
            return True
    except:
        pass
    
    print("⚠️  Modal authentication not yet configured")
    return False

def create_directories():
    """必要なディレクトリを作成"""
    print("📁 Creating directories...")
    directories = [
        "configs",
        "Data",
        "model_assets",
        "logs",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created")

def create_sample_files():
    """サンプルファイルを作成"""
    print("📝 Creating sample files...")
    
    # サンプル使用方法スクリプト
    usage_script = """#!/usr/bin/env python3
# Modal学習システムの使用例

# 1. データセットの準備
python modal_manager.py prepare --dataset-path ./Data/my_voice

# 2. 学習設定の作成
python modal_manager.py config --dataset-name my_voice

# 3. コスト推定（例：500サンプル、1000エポック）
python modal_manager.py estimate --dataset-size 500 --epochs 1000 --gpu-type A100

# 4. 学習実行スクリプトの生成
python modal_manager.py script --dataset-name my_voice

# 5. 実際の学習実行
# modal run modal_training.py --action=train --dataset-name=my_voice

print("Modal学習システムの使用例を確認してください")
"""
    
    with open("scripts/modal_usage_example.py", "w", encoding="utf-8") as f:
        f.write(usage_script)
    
    print("✅ Sample files created")

def run_quick_test():
    """クイックテストを実行"""
    print("🧪 Running quick test...")
    
    try:
        # Modal import テスト
        subprocess.run([sys.executable, "-c", "import modal; print('Modal import: OK')"], check=True)
        
        # Modal manager テスト
        if Path("modal_manager.py").exists():
            subprocess.run([sys.executable, "modal_manager.py", "check"], check=True)
        
        print("✅ Quick test passed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Quick test failed")
        return False

def print_next_steps():
    """次のステップを表示"""
    print("\n🎉 Setup completed!")
    print("\n📋 Next Steps:")
    print("1. Set up Modal authentication (if not done yet):")
    print("   modal token set --token-id <ID> --token-secret <SECRET>")
    print("\n2. Prepare your dataset:")
    print("   python modal_manager.py prepare --dataset-path ./Data/your_voice")
    print("\n3. Create training configuration:")
    print("   python modal_manager.py config --dataset-name your_voice")
    print("\n4. Estimate costs:")
    print("   python modal_manager.py estimate --dataset-size 500")
    print("\n5. Start training:")
    print("   modal run modal_training.py --action=train --dataset-name=your_voice")
    print("\n📖 For detailed documentation, see: docs/MODAL_TRAINING.md")

def main():
    print("🚀 Style-Bert-VITS2 Modal Training Setup")
    print("=" * 50)
    
    # Python バージョンチェック
    if not check_python_version():
        sys.exit(1)
    
    # Modal インストール
    try:
        import modal
        print("✅ Modal already installed")
    except ImportError:
        if not install_modal():
            sys.exit(1)
    
    # ディレクトリ作成
    create_directories()
    
    # サンプルファイル作成
    create_sample_files()
    
    # Modal 認証セットアップ
    auth_ok = setup_modal_auth()
    
    # クイックテスト
    if auth_ok:
        run_quick_test()
    
    # 次のステップを表示
    print_next_steps()

if __name__ == "__main__":
    main()
