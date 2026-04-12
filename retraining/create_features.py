from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import av
import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Pipeline import Pipeline
from src.State import State
from src.features import transform
from src.mediapipe_pose import MediaPipePose
from src.pose_processor import PoseProcessor


CORRECT_DIR = ROOT / "posture_checker_offline/data/raw/videos/correct"
INCORRECT_DIR = ROOT / "posture_checker_offline/data/raw/videos/incorrect"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


class FeatureCollectorPipeline(Pipeline):
    def __init__(self, mode, state):
        self.video_id: str | None = None
        self.label: str | None = None
        self.rep_rows: list[dict] = []
        super().__init__(mode=mode, state=state)

    def _predict_sequence(self):
        X = transform(pd.DataFrame(self.sequence))
        if X.empty:
            return

        row = X.iloc[0].to_dict()
        row["video_id"] = self.video_id
        row["label"] = self.label
        row["rep_id"] = len(self.rep_rows) + 1
        self.rep_rows.append(row)

    def finish(self):
        self._queue.join()
        self._stop_event.set()
        self._worker.join(timeout=2)


class FeatureCollectorProcessor(PoseProcessor):
    def _ensure_pipeline(self):
        if self.pipe is None:
            self.pipe = FeatureCollectorPipeline(mode=self.mode, state=self.state)


def run_video_collect_features(path: str, label: str) -> list[dict]:
    pose = MediaPipePose()
    state = State()
    processor = FeatureCollectorProcessor(pose=pose, state=state)
    processor._ensure_pipeline()

    if processor.pipe is None:
        return []

    processor.pipe.video_id = Path(path).name
    processor.pipe.label = label

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        processor.recv(av_frame)

    cap.release()
    processor.pipe.finish()

    rows = [dict(row) for row in processor.pipe.rep_rows]
    if len(rows) <= 1:
        for row in rows:
            row.pop("rep_id", None)

    return rows


def iter_videos(correct_dir: Path, incorrect_dir: Path) -> list[tuple[str, str]]:
    items = []

    for label, folder in (("correct", correct_dir), ("incorrect", incorrect_dir)):
        if not folder.exists():
            continue

        for filename in sorted(os.listdir(folder)):
            if filename.endswith(VIDEO_EXTENSIONS):
                items.append((str(folder / filename), label))

    return items


def build_features_dataframe(correct_dir: Path, incorrect_dir: Path, max_videos: int | None = None) -> pd.DataFrame:
    video_items = iter_videos(correct_dir, incorrect_dir)
    if max_videos is not None:
        video_items = video_items[:max_videos]

    rows = []
    for video_path, label in video_items:
        video_rows = run_video_collect_features(video_path, label)
        if not video_rows:
            video_rows = [{
                "video_id": Path(video_path).name,
                "label": label,
                "elbow_mean": pd.NA,
                "elbow_std": pd.NA,
                "body_mean": pd.NA,
                "hip_mean": pd.NA,
            }]
        rows.extend(video_rows)
        print(f"{Path(video_path).name}: collected {len(video_rows)} rep rows")

    if not rows:
        raise ValueError("No rep features were collected.")

    df = pd.DataFrame(rows)
    preferred = ["video_id", "label", "rep_id", "elbow_mean", "elbow_std", "body_mean", "hip_mean"]
    columns = [column for column in preferred if column in df.columns]
    return df[columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a MediaPipe 2D feature dataframe using the existing app pipeline."
    )
    parser.add_argument("--correct-dir", type=Path, default=CORRECT_DIR)
    parser.add_argument("--incorrect-dir", type=Path, default=INCORRECT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/mediapipe_2d_app_pipeline_features.csv",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_features_dataframe(
        correct_dir=args.correct_dir,
        incorrect_dir=args.incorrect_dir,
        max_videos=args.max_videos,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print()
    print(df.head().to_string(index=False))
    print()
    print(f"Saved features to: {args.output}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
