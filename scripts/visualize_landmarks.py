"""
intput: video file
output:
#  Data Adapter ：
pose_data_payload = {
    0: { # Frame 0
        "right_shoulder": {"x": 0.51, "y": 0.50, "z": 0.50},
        "right_elbow": {"x": 0.60, "y": 0.35, "z": 0.45},
        "right_wrist": {"x": 0.65, "y": 0.30, "z": 0.50},
        "right_hip": {"x": 0.50, "y": 0.80, "z": 0.50}
    },
    1: { # Frame 1
        "right_shoulder": {"x": 0.52, "y": 0.50, "z": 0.50},
        "right_elbow": {"x": 0.65, "y": 0.20, "z": 0.55},
        "right_wrist": {"x": 0.75, "y": 0.05, "z": 0.60},
        "right_hip": {"x": 0.51, "y": 0.80, "z": 0.50}
    },
    # ...
}
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '../models/pose_landmarker_heavy.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)
# Create a PoseLandmarker object
detector = vision.PoseLandmarker.create_from_options(options)

# init the pose data payload
pose_data_payload = {}
frame_idx = 0
# Open the webcam
cap = cv2.VideoCapture("../data/raw_videos/test_clear_trim2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")
while True:
    # Process the video frames
    success, frame = cap.read()
    if not success:
        print("End of video reached or failed to read the video frame.")
        break

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    # set up for timestamp in milliseconds for the current frame
    timestamp_ms = int((frame_idx / fps) * 1000)

    detection_result = detector.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
    # ensure the detection result contains pose landmarks
    if detection_result.pose_landmarks:
        print("Pose landmarks detected:")
        Landmarks = detection_result.pose_landmarks[0]
        height, width, _ = frame.shape
        right_shoulder_x = int(Landmarks[12].x * width)
        right_shoulder_y = int(Landmarks[12].y * height)
        cv2.circle(frame, (right_shoulder_x, right_shoulder_y), 5, (0, 255, 0), -1)
        right_elbow_x = int(Landmarks[14].x * width)
        right_elbow_y = int(Landmarks[14].y * height)
        cv2.circle(frame, (right_elbow_x, right_elbow_y), 5, (0, 0, 255), -1)
        right_wrist_x = int(Landmarks[16].x * width)
        right_wrist_y = int(Landmarks[16].y * height)
        cv2.circle(frame, (right_wrist_x, right_wrist_y), 5, (255, 0, 0), -1)
        right_hip_x = int(Landmarks[24].x * width)
        right_hip_y = int(Landmarks[24].y * height)
        cv2.circle(frame, (right_hip_x, right_hip_y), 5, (255, 255, 0), -1)


    else:
        pose_data_payload[frame_idx] = None
        print("No pose landmarks detected.")
    display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Pose Detection', display_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap.release()
detector.close()
