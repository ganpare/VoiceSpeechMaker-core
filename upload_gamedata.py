import modal

app = modal.App("gamedata-uploader")

gamedata_volume = modal.Volume.from_name("gamedata-volume", create_if_missing=True)

@app.local_entrypoint()
def main():
    try:
        print("アップロード開始")
        gamedata_volume.commit_local_dir("F:/gamedata")
        print("アップロード完了！")
    except Exception as e:
        print("エラーが発生しました:", e) 