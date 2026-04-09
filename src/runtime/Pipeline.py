import queue
import threading
import joblib
import pandas as pd
import numpy as np

from src.runtime.Counter import Counter
from src.feature.frame_features import extract
from src.feature.sequence_features import transform

MODEL_2D_PATH = "data/models/pushup_2d.joblib"
MODEL_3D_PATH = "data/models/pushup_3d.joblib"

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

class Pipeline:
    def __init__(self, mode, state):
        self.mode = mode
        self.state = state

        # internal
        self._queue = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self.sequence = []
        self.counter = Counter()

        # load model once
        self.model = self._load_model()

        # start worker thread
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # Add result to queue
    def submit(self, result):
        try:
            self._queue.put_nowait(result)
        except queue.Full:
            pass
        
    # Run loop
    def _run(self):
        while not self._stop_event.is_set():
            try:
                result = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.process(result)
            finally:
                self._queue.task_done()

    # Logic for processing frame
    def process(self, result):

        # Transform to general
        general = self._to_landmarks(result)

        if general is None:
            return
  
        # Transform to features
        frame_features = extract(general, self.mode)

        # Store current frame
        self.sequence.append(frame_features)

        # Give feedback
        feedback = frame_features["coach"]
        self._update_feedback(feedback)

        # Update counter logic
        phase = frame_features["phase"]
        complete = self.counter.update(phase)

        if complete:
            self._predict_sequence()
            self.sequence.clear()
  
    def stop(self):
        # Reset counter, state, pipe
        pass

    def _predict_sequence(self):
        
        # Build features
        X = transform(pd.DataFrame(self.sequence))

        # Predict
        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)
        confidence = float(max(proba[0]))

        # Update
        self._update_state(pred_label=str(pred), confidence=confidence)

    def _update_state(self, pred_label, confidence=None):
        with self.state.lock:
            self.state.total_reps += 1
            rep_id = self.state.total_reps

            self.state.last_prediction = pred_label
            self.state.last_confidence = confidence

            if pred_label == "correct":
                self.state.correct_reps += 1
            else:
                self.state.incorrect_reps += 1

            self.state.history.append({
                "rep": rep_id,
                "label": pred_label,
                "confidence": confidence,
            })

    def _update_feedback(self, feedback):
        with self.state.lock:
            self.state.feedback = feedback


    def _load_model(self):
        model_path = MODEL_2D_PATH if self.mode == 2 else MODEL_3D_PATH
        return joblib.load(model_path)
    
    def _to_landmarks(self, result):
        if result is None or not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        pose_dict = {}

        for i, name in enumerate(LANDMARK_NAMES):
            lm = landmarks[i]
            pose_dict[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": getattr(lm, "z", np.nan),
                "visibility": getattr(lm, "visibility", np.nan),
            }

        return pose_dict