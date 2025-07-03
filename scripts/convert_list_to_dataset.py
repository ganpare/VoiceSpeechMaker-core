#!/usr/bin/env python3
"""
listファイルをStyle-Bert-VITS2の訓練データ形式に変換するスクリプト

listファイル形式:
/path/to/audio.wav|speaker_name|language|transcript

Style-Bert-VITS2の期待する形式:
inputs/speaker_name/
├── wavs/          # 音声ファイル
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
├── esd.list       # 音声とテキストのペア情報
└── ...
"""

import argparse
import shutil
import os
from pathlib import Path
from typing import List, Tuple
import re


def parse_list_file(list_path: str) -> List[Tuple[str, str, str, str]]:
    """listファイルを解析"""
    entries = []
    with open(list_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) != 4:
                print(f"警告: 行 {line_num} の形式が正しくありません: {line}")
                continue
                
            audio_path, speaker, language, transcript = parts
            entries.append((audio_path.strip(), speaker.strip(), language.strip(), transcript.strip()))
    
    return entries


def sanitize_filename(filename: str) -> str:
    """ファイル名を安全な形式に変換"""
    # 危険な文字を除去
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 連続するアンダースコアを単一に
    filename = re.sub(r'_+', '_', filename)
    # 前後の空白とピリオドを除去
    filename = filename.strip(' .')
    return filename


def convert_dataset(list_path: str, output_dir: str, speaker_name: str = None):
    """listファイルをStyle-Bert-VITS2形式に変換"""
    entries = parse_list_file(list_path)
    
    if not entries:
        print("エラー: 有効なエントリが見つかりませんでした")
        return
    
    # 話者名を決定
    if speaker_name is None:
        speaker_name = entries[0][1]  # 最初のエントリから話者名を取得
    
    speaker_name = sanitize_filename(speaker_name)
    
    # 出力ディレクトリを作成
    output_path = Path(output_dir)
    wavs_dir = output_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # esd.listファイルを作成
    esd_list = []
    copied_files = 0
    
    print(f"話者名: {speaker_name}")
    print(f"出力ディレクトリ: {output_path}")
    print(f"処理対象: {len(entries)} ファイル")
    
    for i, (audio_path, speaker, language, transcript) in enumerate(entries, 1):
        # 音声ファイルの存在確認
        if not os.path.exists(audio_path):
            print(f"警告: 音声ファイルが見つかりません: {audio_path}")
            continue
        
        # ファイル名を生成 (ゼロパディング付き)
        base_name = f"{i:04d}"
        audio_ext = Path(audio_path).suffix
        new_audio_name = f"{base_name}{audio_ext}"
        new_audio_path = wavs_dir / new_audio_name
        
        try:
            # 音声ファイルをコピー
            shutil.copy2(audio_path, new_audio_path)
            
            # esd.listエントリを追加
            # Format: wavs/filename.wav|speaker_name|language|transcript
            esd_list.append(f"wavs/{new_audio_name}|{speaker_name}|{language}|{transcript}")
            copied_files += 1
            
            if copied_files % 100 == 0:
                print(f"処理中... {copied_files}/{len(entries)}")
                
        except Exception as e:
            print(f"エラー: {audio_path} のコピーに失敗: {e}")
            continue
    
    # esd.listファイルを書き出し
    esd_path = output_path / "esd.list"
    with open(esd_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(esd_list))
    
    print(f"\n変換完了!")
    print(f"コピーされたファイル数: {copied_files}")
    print(f"出力先: {output_path}")
    print(f"esd.list: {esd_path}")
    
    # 統計情報
    total_duration_estimate = copied_files * 3.0  # 平均3秒と仮定
    print(f"\n統計情報:")
    print(f"- 推定総時間: {total_duration_estimate/60:.1f}分")
    print(f"- 平均ファイル数/分: {copied_files/(total_duration_estimate/60):.1f}")


def main():
    parser = argparse.ArgumentParser(description="listファイルをStyle-Bert-VITS2形式に変換")
    parser.add_argument("--input", "-i", required=True, help="入力listファイルのパス")
    parser.add_argument("--output", "-o", required=True, help="出力ディレクトリのパス")
    parser.add_argument("--speaker", "-s", help="話者名 (指定しない場合はlistファイルから自動取得)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"エラー: 入力ファイルが見つかりません: {args.input}")
        return 1
    
    convert_dataset(args.input, args.output, args.speaker)
    return 0


if __name__ == "__main__":
    exit(main())