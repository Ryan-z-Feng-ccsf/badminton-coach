"""
Input:
    Raw video or live camera
output:
metadata={
    is_valid= True/False, # Indicate if there are people detected
    fps = ?
    pose_data_payload = {
    0: {
        "right_shoulder":[x,y,z],
        "right_elbow":[x,y,z],
        "right_wrist":[x,y,z],
        "right_hip":[x,y,z]
    },
    1: {
        "right_shoulder":[x,y,z],
        "right_elbow":[x,y,z],
        "right_wrist":[x,y,z],
        "right_hip":[x,y,z]
    },
    ...,
    }
    }
"""
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

load_dotenv()


class PoseDetector:
    def __init__(self, model_path_key="POSE_MODEL_PATH",
                 min_pose_detection_confidence=0.5, min_tracking_confidence=0.5, output_segmentation_masks=False):
        """
        initialize the PoseLandmarker object
        """
        # Get the relative path from env
        raw_relative_path = os.getenv(model_path_key)
        # get the absolute path of this python script directory
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # Concatenate the current directory with the relative path
        # and normalize it to the final correct absolute path
        self.model_path = os.path.abspath(os.path.join(cur_dir, raw_relative_path))
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.output_segmentation_masks = output_segmentation_masks
        self.detector = None

    def __enter__(self):
        """
        Initialize the PoseLandmarker object
        Set up phase: Allocate resources
        """
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=self.output_segmentation_masks
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.detector:
            self.detector.close()
        if exc_val is not None:
            print(f"An exception occurred: {exc_val}")
        # Return True allows the exception to propagate up
        # Return False would swallow the exception
        return False

    def process_frame(self, frame_idx: int, frame_rgb, fps: float) -> dict:
        """
        Extract parameters in one frame
        param: frame_idx: index of the frame
        param: frame_rgb: RGB frame
        param: fps: FPS
        return: result: metadata
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((frame_idx / fps) * 1000)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
        result: dict
        # ensure the detection result contains pose landmarks
        if detection_result.pose_landmarks:
            print("Pose landmarks detected")

            # Extract all the 33 points
            landmarks = detection_result.pose_landmarks[0]
            result = {
                "is_valid": detection_result.is_valid,
                "frame": {
                    "right_shoulder": [landmarks[12].x, landmarks[12].y, landmarks[12].z],
                    "right_elbow": [landmarks[14].x, landmarks[14].y, landmarks[14].z],
                    "right_wrist": [landmarks[16].x, landmarks[16].y, landmarks[16].z],
                    "right_hip": [landmarks[24].x, landmarks[24].y, landmarks[24].z]
                }
            }
        else:
            print("No pose landmarks detected")
            result = {
                "is_valid": False,
                "frame": None
            }
        return result


class VideoCapture:
    def __init__(self, video_path_key="VIDEO_PATH"):
        # Get the relative path
        raw_video_path = os.getenv(video_path_key)
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_path = os.path.abspath(os.path.join(cur_dir, raw_video_path))

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
        if exc_val is not None:
            print(f"An exception occurred: {exc_val}")
        return False

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)
