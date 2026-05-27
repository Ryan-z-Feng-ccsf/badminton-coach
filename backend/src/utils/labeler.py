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
