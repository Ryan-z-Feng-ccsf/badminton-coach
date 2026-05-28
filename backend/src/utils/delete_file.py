import glob
import os
from config.core import config

VIDEO_DIR = config.get_path("VIDEO_DIR")

def batch_delete_file():
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    for video_file in video_files:
        print(f"🗑️ Deleting video: {video_file}")
        os.remove(video_file)
        print(f"✅ Deleted: {video_file}")

if __name__ == "__main__":
    batch_delete_file()
