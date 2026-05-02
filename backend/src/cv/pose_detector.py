"""
Input:
    Raw video or live camera
output:
metadata={

    fps = ?
    pose_data_payload = {
    0: {
        "is_valid"= True/False, # Indicate if there are people detected
        "right_shoulder":[x,y,z],
        "right_elbow":[x,y,z],
        "right_wrist":[x,y,z],
        "right_hip":[x,y,z],
        'right_knee':[x,y,z],
        'right_ankle':[x,y,z]
    },
    1: {
        "right_shoulder":[x,y,z],
        "right_elbow":[x,y,z],
        "right_wrist":[x,y,z],
        "right_hip":[x,y,z],
        'right_knee':[x,y,z],
        'right_ankle':[x,y,z]
    },
    ...,
    }
    }
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

load_dotenv()


class PoseDetector:
    def __init__(self, model_path : str,
                 min_pose_detection_confidence=0.5, min_tracking_confidence=0.5, output_segmentation_masks=False):
        """
        initialize the PoseLandmarker object
        """  
        self._model_path = model_path
        self._min_pose_detection_confidence = min_pose_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._output_segmentation_masks = output_segmentation_masks
        self.detector = None

    def __enter__(self):
        """
        Initialize the PoseLandmarker object
        Set up phase: Allocate resources
        """
        if not self._model_path:
            raise FileNotFoundError("Model file not found")
        base_options = python.BaseOptions(model_asset_path=self._model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self._min_pose_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_segmentation_masks=self._output_segmentation_masks
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

    def _get_pt(self, landmarks):
        return [round(landmarks.x, 3), round(landmarks.y, 3), round(landmarks.z, 3)]

    def process_frame(self, frame_idx: int, frame_bgr, fps: float) -> dict:
        """
        Extract parameters in one frame
        param: frame_idx: index of the frame
        param: frame_bgr: BGR frame
        param: fps: FPS
        return: result: metadata
        """
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # Set up for timestamp in milliseconds for the current time
        timestamp_ms = int((frame_idx / fps) * 1000)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
        result: dict
        # ensure the detection result contains pose landmarks
        if detection_result.pose_landmarks:
            print("Pose landmarks detected")

            # Extract all the 33 points
            landmarks = detection_result.pose_landmarks[0]
            result = {
                "is_valid": True,
                'joints': {
                    "right_shoulder": self._get_pt(landmarks[12]),
                    "right_elbow": self._get_pt(landmarks[14]),
                    "right_wrist": self._get_pt(landmarks[16]),
                    "right_hip": self._get_pt(landmarks[24]),
                    'right_knee':self._get_pt(landmarks[26]),
                    'right_ankle':self._get_pt(landmarks[28])
                }
            }
        else:
            print("No pose landmarks detected")
            result = {
                "is_valid": False,
                'joints': None
            }
        return result


class VideoCapture:
    def __init__(self, video_path:str):
        # Get the relative path
        
        
        self._video_path = video_path

    def __enter__(self):
        self.cap = cv2.VideoCapture(self._video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError("Video file not found")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
        if exc_val is not None:
            print(f"An exception occurred: {exc_val}")
        return False

    def get_fps(self):
        return round(self.cap.get(cv2.CAP_PROP_FPS), 3)


class DetectorEngine:
    def get_metadata(self, model_path:str,video_path:str):
        metadata: dict = {}
        with PoseDetector(model_path) as pose_detector:
            with VideoCapture(video_path) as video_capture:
                pose_data_payload: dict = {}
                frame_idx = 0
                while True:
                    success, frame_bgr = video_capture.cap.read()
                    if not success:
                        print("Video capture failed / end of video")
                        break

                    pose_data_payload[frame_idx] = pose_detector.process_frame(frame_idx, frame_bgr,
                                                                               video_capture.get_fps())
                    frame_idx += 1
                metadata["fps"] = video_capture.get_fps()
                metadata["pose_data_payload"] = pose_data_payload

        return metadata


if __name__ == "__main__":
    from config.core import config
    engine = DetectorEngine(config.get_path("POSE_MODEL_PATH"))
    print(engine.get_metadata())
