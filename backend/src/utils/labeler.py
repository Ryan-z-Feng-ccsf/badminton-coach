import numpy as np
import os
from collections import deque
from config.core import config
from src.utils.extractor import PoseDetector, VideoCapture
import cv2

MODEL_PATH = config.get_path("POSE_MODEL_PATH")

SAVE_DIR = config.get_path("TRAIN_LUNGE_DIR")
os.makedirs(SAVE_DIR, exist_ok=True)
TARGET_FPS = 30.0
MAX_DISPLAY_WIDTH = 2560
MAX_DISPLAY_HEIGHT = 1440


def runAutomatedLabeler(DATA_VIDEO_PATH, video_name):
    buffer = deque(maxlen=20)

    with PoseDetector(MODEL_PATH) as pose_detector:
        with VideoCapture(DATA_VIDEO_PATH) as video_capture:
            frame_idx = 0
            print("🕹️")

            while True:
                success, frame_bgr = video_capture.cap.read()
                if not success:
                    print("Video capture failed / end of video")
                    break

                stride = max(1, round(video_capture.get_fps() / TARGET_FPS))

                if frame_idx % stride == 0:
                    result = pose_detector.process_frame(frame_idx, frame_bgr, video_capture.get_fps())
                    buffer.append(result)  # Automated pop and append elements

                    display_frame = cv2.resize(frame_bgr, (1280,720))
                    cv2.imshow("Labeler", display_frame)
                    key = cv2.waitKey(30)

                    if key == ord('s'):
                        if len(buffer) == 20:
                            past_11_frames = list(buffer)[0:11]

                            buffer_array = np.array(past_11_frames)  # Convert deque to numpy array
                            data = buffer_array.reshape(buffer_array.shape[0], -1)
                            file_name = f"lunge_{video_name}_frame{frame_idx}.npy"
                            save_dir = os.path.abspath(os.path.join(SAVE_DIR, file_name))
                            np.save(save_dir, data)
                            buffer.clear()

                        else:
                            print("⚠️ Not enough frames in buffer to save")


                    elif key == ord('d'):
                        skip_frames = (video_capture.get_fps() * 2)
                        current_pos = int(video_capture.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        video_capture.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + skip_frames)

                        frame_idx = current_pos + skip_frames
                        buffer.clear()
                        print("⏩ Skipping 2 seconds ahead...")
                    elif key == ord('n'):
                        print("⏭️ Skipping to next video...")
                        return "NEXT"

                    elif key == ord('q'):
                        print("Exiting...")
                        return "QUIT"

                frame_idx += 1
    return "FINISHED"


if __name__ == "__main__":
    DATA_VIDEO_PATH = config.get_path("RAW_VIDEO_PATH")
    print(runAutomatedLabeler(DATA_VIDEO_PATH, "0yeX8D7Bo4k"))
