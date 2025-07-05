"""
Modal function to debug dataset list files (train.list and val.list)
This helps identify issues with speaker processing by examining file formats.
"""

import modal
import os
from pathlib import Path

# Create a simple image with basic Python requirements
image = modal.Image.debian_slim(python_version="3.11")

# Modal App definition
app = modal.App("debug-dataset-lists", image=image)

# Volume definition for data access
data_volume = modal.Volume.from_name("sbv2-data-volume")

@app.function(
    volumes={"/data": data_volume},
    timeout=600,
)
def read_dataset_lists(dataset_name: str, lines_to_show: int = 10):
    """
    Read and display the contents of train.list and val.list files
    from a dataset in the Modal volume.
    
    Args:
        dataset_name: Name of the dataset (e.g., "大蔵衣遠")
        lines_to_show: Number of lines to display from each file
    
    Returns:
        Dictionary containing file contents and analysis
    """
    print(f"🔍 Examining dataset lists for: {dataset_name}")
    
    # Define paths
    dataset_path = Path("/data/Data") / dataset_name
    train_list_path = dataset_path / "train.list"
    val_list_path = dataset_path / "val.list"
    esd_list_path = dataset_path / "esd.list"
    
    result = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "files_exist": {},
        "file_contents": {},
        "analysis": {}
    }
    
    # Check if dataset directory exists
    if not dataset_path.exists():
        result["error"] = f"Dataset directory not found: {dataset_path}"
        return result
    
    # List all files in the dataset directory
    try:
        all_files = [f.name for f in dataset_path.iterdir() if f.is_file()]
        all_dirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        result["directory_contents"] = {
            "files": all_files,
            "directories": all_dirs
        }
        print(f"📁 Directory contents:")
        print(f"  Files: {all_files}")
        print(f"  Directories: {all_dirs}")
    except Exception as e:
        result["directory_error"] = str(e)
    
    # Check each list file
    for file_name, file_path in [
        ("train.list", train_list_path),
        ("val.list", val_list_path),
        ("esd.list", esd_list_path)
    ]:
        result["files_exist"][file_name] = file_path.exists()
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Store basic info
                result["file_contents"][file_name] = {
                    "total_lines": len(lines),
                    "first_lines": [line.strip() for line in lines[:lines_to_show]],
                    "last_lines": [line.strip() for line in lines[-min(3, len(lines)):]] if len(lines) > lines_to_show else []
                }
                
                # Analyze format
                if lines:
                    first_line = lines[0].strip()
                    result["analysis"][file_name] = analyze_line_format(first_line)
                
                print(f"📄 {file_name}:")
                print(f"  Total lines: {len(lines)}")
                print(f"  First few lines:")
                for i, line in enumerate(lines[:lines_to_show]):
                    print(f"    {i+1}: {line.strip()}")
                    
            except Exception as e:
                result["file_contents"][file_name] = {"error": str(e)}
                print(f"❌ Error reading {file_name}: {e}")
        else:
            print(f"❌ {file_name} not found")
    
    # Additional analysis
    result["speaker_analysis"] = analyze_speaker_info(result)
    
    return result

def analyze_line_format(line: str) -> dict:
    """
    Analyze the format of a line from a dataset list file.
    
    Args:
        line: A line from train.list, val.list, or esd.list
        
    Returns:
        Dictionary with format analysis
    """
    analysis = {
        "raw_line": line,
        "parts_count": 0,
        "separator": None,
        "potential_format": "unknown",
        "has_speaker_info": False
    }
    
    if not line.strip():
        analysis["potential_format"] = "empty"
        return analysis
    
    # Try different separators
    separators = ['|', '\t', ',', ';']
    for sep in separators:
        if sep in line:
            parts = line.split(sep)
            if len(parts) > analysis["parts_count"]:
                analysis["parts_count"] = len(parts)
                analysis["separator"] = sep
                analysis["parts"] = [part.strip() for part in parts]
    
    # If no separator found, treat as single part
    if analysis["separator"] is None:
        analysis["parts_count"] = 1
        analysis["parts"] = [line.strip()]
    
    # Determine format based on parts count and content
    if analysis["parts_count"] >= 2:
        # Common formats:
        # wav_path|transcript|speaker_id (Style-Bert-VITS2)
        # wav_path|transcript (basic)
        # wav_path|speaker_id|transcript (some formats)
        
        parts = analysis["parts"]
        
        # Check if first part looks like a file path
        if parts[0].endswith('.wav') or '/' in parts[0] or '\\' in parts[0]:
            analysis["potential_format"] = "wav_path_first"
            
            if analysis["parts_count"] == 2:
                analysis["potential_format"] = "wav_path|transcript"
            elif analysis["parts_count"] == 3:
                analysis["potential_format"] = "wav_path|transcript|speaker_id"
                analysis["has_speaker_info"] = True
                analysis["potential_speaker"] = parts[2]
            elif analysis["parts_count"] > 3:
                analysis["potential_format"] = "wav_path|transcript|speaker_id|extra"
                analysis["has_speaker_info"] = True
                analysis["potential_speaker"] = parts[2]
        
        # Check if any part looks like a speaker ID
        for i, part in enumerate(parts):
            if part.isdigit() or (part.startswith('speaker') and part != parts[1]):
                analysis["has_speaker_info"] = True
                analysis["potential_speaker"] = part
                analysis["speaker_position"] = i
    
    return analysis

def analyze_speaker_info(result: dict) -> dict:
    """
    Analyze speaker information across all list files.
    
    Args:
        result: The result dictionary from read_dataset_lists
        
    Returns:
        Dictionary with speaker analysis
    """
    analysis = {
        "speakers_found": set(),
        "format_consistency": True,
        "issues": []
    }
    
    formats = []
    
    for file_name in ["train.list", "val.list", "esd.list"]:
        if file_name in result.get("analysis", {}):
            file_analysis = result["analysis"][file_name]
            formats.append(file_analysis.get("potential_format", "unknown"))
            
            if file_analysis.get("has_speaker_info"):
                speaker = file_analysis.get("potential_speaker", "unknown")
                analysis["speakers_found"].add(speaker)
            else:
                analysis["issues"].append(f"{file_name} appears to lack speaker information")
    
    # Check format consistency
    if len(set(formats)) > 1:
        analysis["format_consistency"] = False
        analysis["issues"].append(f"Inconsistent formats across files: {formats}")
    
    # Convert set to list for JSON serialization
    analysis["speakers_found"] = list(analysis["speakers_found"])
    
    # Check if speaker info is missing
    if not analysis["speakers_found"]:
        analysis["issues"].append("No speaker information found in any list file")
    
    return analysis

@app.function(
    volumes={"/data": data_volume},
    timeout=600,
)
def check_raw_audio_files(dataset_name: str):
    """
    Check the raw audio files in the dataset directory.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with audio file information
    """
    print(f"🎵 Checking raw audio files for: {dataset_name}")
    
    dataset_path = Path("/data/Data") / dataset_name
    result = {
        "dataset_name": dataset_name,
        "audio_files": [],
        "directories": []
    }
    
    if not dataset_path.exists():
        result["error"] = f"Dataset directory not found: {dataset_path}"
        return result
    
    # Check for raw directory
    raw_dir = dataset_path / "raw"
    if raw_dir.exists():
        print(f"📁 Found raw directory: {raw_dir}")
        audio_files = list(raw_dir.glob("*.wav"))
        result["audio_files"] = [f.name for f in audio_files]
        result["raw_directory_exists"] = True
    else:
        print(f"📁 No raw directory found, checking main directory")
        audio_files = list(dataset_path.glob("*.wav"))
        result["audio_files"] = [f.name for f in audio_files]
        result["raw_directory_exists"] = False
    
    # Check for other directories
    for item in dataset_path.iterdir():
        if item.is_dir():
            result["directories"].append(item.name)
    
    result["audio_count"] = len(result["audio_files"])
    
    print(f"🎵 Found {result['audio_count']} audio files")
    if result["audio_files"]:
        print(f"  Sample files: {result['audio_files'][:5]}")
    
    return result

# Command line entrypoint
@app.local_entrypoint()
def main(
    dataset_name: str = "大蔵衣遠",
    lines_to_show: int = 10,
    action: str = "read-lists"
):
    """
    Debug dataset list files.
    
    Usage:
    modal run debug_dataset_lists.py --dataset-name="大蔵衣遠" --lines-to-show=10
    modal run debug_dataset_lists.py --dataset-name="大蔵衣遠" --action="check-audio"
    """
    
    if action == "read-lists":
        print(f"🔍 Reading dataset lists for: {dataset_name}")
        result = read_dataset_lists.remote(
            dataset_name=dataset_name,
            lines_to_show=lines_to_show
        )
        
        print(f"\n📊 Analysis Results:")
        print(f"Dataset: {result['dataset_name']}")
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"\n📁 Directory Contents:")
        if "directory_contents" in result:
            contents = result["directory_contents"]
            print(f"  Files: {contents['files']}")
            print(f"  Directories: {contents['directories']}")
        
        print(f"\n📄 File Existence:")
        for file_name, exists in result["files_exist"].items():
            status = "✅" if exists else "❌"
            print(f"  {status} {file_name}")
        
        print(f"\n🧐 Speaker Analysis:")
        speaker_analysis = result.get("speaker_analysis", {})
        print(f"  Speakers found: {speaker_analysis.get('speakers_found', [])}")
        print(f"  Format consistency: {speaker_analysis.get('format_consistency', 'Unknown')}")
        
        if speaker_analysis.get("issues"):
            print(f"  Issues:")
            for issue in speaker_analysis["issues"]:
                print(f"    - {issue}")
        
        # Show detailed analysis for each file
        print(f"\n📋 Detailed File Analysis:")
        for file_name, content in result.get("file_contents", {}).items():
            if "error" not in content:
                print(f"  {file_name}:")
                print(f"    Total lines: {content['total_lines']}")
                if file_name in result.get("analysis", {}):
                    analysis = result["analysis"][file_name]
                    print(f"    Format: {analysis.get('potential_format', 'unknown')}")
                    print(f"    Has speaker info: {analysis.get('has_speaker_info', False)}")
                    if analysis.get("separator"):
                        print(f"    Separator: '{analysis['separator']}'")
    
    elif action == "check-audio":
        print(f"🎵 Checking audio files for: {dataset_name}")
        result = check_raw_audio_files.remote(dataset_name=dataset_name)
        
        print(f"\n📊 Audio File Analysis:")
        print(f"Dataset: {result['dataset_name']}")
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"Audio files found: {result['audio_count']}")
        print(f"Raw directory exists: {result['raw_directory_exists']}")
        print(f"Directories: {result['directories']}")
        
        if result["audio_files"]:
            print(f"Sample audio files:")
            for file in result["audio_files"][:10]:
                print(f"  - {file}")
    
    else:
        print(f"❌ Unknown action: {action}")
        print("Available actions: read-lists, check-audio")

if __name__ == "__main__":
    print("Dataset List Debug Tool")
    print("Use: modal run debug_dataset_lists.py --help for usage information")