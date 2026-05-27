import cv2
import numpy as np
import os
from collections import deque
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from config.core import config
from src.utils.extractor import PoseDetector, VideoCapture
from utils.generate_id import unique_id

MODEL_PATH = config.get_path("POSE_MODEL_PATH")
DATA_VIDEO_PATH = config.get_path("RAW_VIDEO_PATH")
SAVE_DIR = config.get_path("TRAIN_LUNGE_DIR")

os.makedirs(SAVE_DIR, exist_ok=True)

def runAutomatedLabeler():
    buffer = deque(maxlen=11)

    with PoseDetector(MODEL_PATH) as pose_detector:
        with VideoCapture(DATA_VIDEO_PATH) as video_capture:
            frame_idx = 0
            print("🕹️")

            while True:
                success, frame_bgr = video_capture.cap.read()
                if not success:
                    print("Video capture failed / end of video")
                    break

                result = pose_detector.process_frame(frame_idx, frame_bgr, video_capture.get_fps())
                buffer.append(result)  # Automated pop and append elements

                display_frame = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("Labeler", display_frame)
                key = cv2.waitKey(30)

                if key == ord('s'):
                    if len(buffer) == 11:
                        buffer_array = np.array(buffer)  # Convert deque to numpy array
                        data = buffer_array.reshape(buffer_array.shape[0],-1)
                        save_dir = os.path.abspath(os.path.join(SAVE_DIR, f"lunge_{unique_id()}"))
                        np.save(save_dir, data)

                    else:
                        print("⚠️ Not enough frames in buffer to save")

                elif key == ord('q'):
                    break

                frame_idx += 1


if __name__ == "__main__":
    print(runAutomatedLabeler())
