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
                buffer.append(result)
                
                
                
                
                frame_idx += 1