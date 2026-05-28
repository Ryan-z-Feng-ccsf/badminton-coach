import glob
import os
from labeler import runAutomatedLabeler
from config.core import config

VIDEO_DIR = config.get_path("VIDEO_DIR")
print(VIDEO_DIR)
VIDEO_EXTENSION = ("*.mp4", "*.avi", "*.mov", "*.MP4", ".webm")  # Add more extensions if needed


def batch_process():
    for video_type in VIDEO_EXTENSION:
        video_files = glob.glob(os.path.join(VIDEO_DIR, video_type))

        for video_file in video_files:
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            print(f"🚀 Processing video: {video_file}")
            status = runAutomatedLabeler(video_file,video_name)
            if status == "QUIT":
                print("Batch processing interrupted by user.")
                return
            elif status == "NEXT":
                print("Skipping to next video...")
                continue
            print(f"✅ Finished processing: {video_file}")
    print("✅ Batch processed------")


if __name__ == "__main__":
    batch_process()
