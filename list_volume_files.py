import modal
import os

image = modal.Image.debian_slim(python_version="3.10")
app = modal.App("list-volume-files", image=image)
data_volume = modal.Volume.from_name("sbv2-data-volume")
output_volume = modal.Volume.from_name("style-bert-vits2-volume")

@app.function(
    volumes={
        "/mnt/gamedata": data_volume,
        "/mnt/output": output_volume
    }
)
def list_files():
    out_path = "/mnt/output/volume_file_list.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk("/mnt/gamedata"):
            level = root.replace("/mnt/gamedata", "").count(os.sep)
            indent = "  " * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = "  " * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")
    print(f"{out_path} に出力しました")

@app.local_entrypoint()
def main():
    list_files.remote() 