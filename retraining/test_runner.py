import os
import sys
from pathlib import Path

import av
import cv2
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_processor import PoseProcessor
from src.mediapipe_pose import MediaPipePose
from src.State import State


CORRECT_DIR = ROOT / "posture_checker_offline/data/raw/videos/correct"
INCORRECT_DIR = ROOT / "posture_checker_offline/data/raw/videos/incorrect"
FEATURES_CSV = ROOT / "data/processed/mediapipe_2d_app_pipeline_features.csv"


def _load_test_files():
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=["elbow_mean", "elbow_std", "body_mean", "hip_mean"])

    correct_files = sorted(
        df.loc[df["label"] == "correct", "video_id"].dropna().unique().tolist()
    )
    incorrect_files = sorted(
        df.loc[df["label"] == "incorrect", "video_id"].dropna().unique().tolist()
    )
    return correct_files, incorrect_files


CORRECT_FILES, INCORRECT_FILES = _load_test_files()


def run_video(path):
    pose = MediaPipePose()
    state = State()
    processor = PoseProcessor(pose=pose, state=state)

    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        processor.recv(av_frame)

    cap.release()
    return state.snapshot()


@pytest.mark.parametrize("filename", CORRECT_FILES)
def test_correct_videos(filename):
    snap = run_video(os.path.join(CORRECT_DIR, filename))

    assert snap["incorrect_reps"] == 0
    assert snap["correct_reps"] > 0


@pytest.mark.parametrize("filename", INCORRECT_FILES)
def test_incorrect_videos(filename):
    snap = run_video(os.path.join(INCORRECT_DIR, filename))

    assert snap["incorrect_reps"] > 0
