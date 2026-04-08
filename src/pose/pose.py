import cv2
import mediapipe as mp
import pandas as pd
import os
import urllib.request

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
model_path = "data/models/pose_landmarker.task"

def _ensure_model():
    os.makedirs("data/models", exist_ok=True)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)

def create_landmarker(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)

def extract_landmarks(video_path):
    rows = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    with create_landmarker(model_path) as landmarker:
        frame_idx = 0

        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            result = detect_pose(frame_bgr, landmarker, timestamp_ms)

            row = {
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                **result_to_row(result)
            }

            rows.append(row)
            frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)

def detect_pose(frame_bgr, landmarker, timestamp_ms):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )
    return landmarker.detect_for_video(mp_image, timestamp_ms)

def result_to_row(result):
    row = {}

    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks[0]):
            name = LANDMARK_NAMES[i]
            row[f"{name}_x"] = lm.x
            row[f"{name}_y"] = lm.y
            row[f"{name}_z"] = lm.z
            row[f"{name}_visibility"] = lm.visibility
            row[f"{name}_presence"] = lm.presence

    return row