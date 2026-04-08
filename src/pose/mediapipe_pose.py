import os
import urllib.request
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "data/models/pose_landmarker.task"


class MediaPipePose:
    def __init__(self):
        self._ensure_model()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=False,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def _ensure_model(self):
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    def detect(self, mp_image, timestamp_ms):
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def draw(self, mp_image, detection_result):
        return self.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        annotated_image = np.copy(rgb_image)

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2
        )

        for pose_landmarks in detection_result.pose_landmarks:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style,
            )

        return annotated_image