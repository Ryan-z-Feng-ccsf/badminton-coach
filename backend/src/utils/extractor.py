import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class PoseDetector:
    def __init__(
            self,
            model_path: str,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
    ):
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
            output_segmentation_masks=self._output_segmentation_masks,
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
        detection_result = self.detector.detect_for_video(
            mp_image, timestamp_ms=timestamp_ms
        )
        result: np.ndarray
        # ensure the detection result contains pose landmarks
        if detection_result.pose_landmarks:
            print("Pose landmarks detected")

            # Extract all the 33 points
            landmarks = detection_result.pose_landmarks[0]
            target_indices = [
                11,
                12,
                15,
                16,
                23,
                24,
                25,
                26,
                27,
                28,
            ]
            coords = np.array([self._get_pt(landmarks[i]) for i in target_indices])

            midpoint = coords[[4, 5]].mean(axis=0)  # Midpoint between left hip and right hip
            normalized_coords = coords - midpoint  # Center the coordinates around the midpoint
            result = normalized_coords
        else:
            print("No pose landmarks detected")
            result = np.zeros((10, 3))  # Return an array of zeros if no landmarks are detected

        return result


class VideoCapture:
    def __init__(self, video_path: str):
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
