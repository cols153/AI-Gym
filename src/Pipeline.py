import queue
import threading
import joblib
import pandas as pd
import numpy as np
from collections import deque

from src.Counter import Counter
from src.constants import LANDMARK_NAMES
from src.features import extract, transform
from src.coach import estimate_phase, give_feedback


MODEL_PATH = "posture_checker_offline/models/mlp_pipeline_mediapipe_2d_tuned.joblib"

class Pipeline:
    def __init__(self, state):
        self.state = state

        # internal
        self._queue = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self.sequence = []
        self.counter = Counter()
        self.phase_window = deque(maxlen=5)

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
        frame_features = extract(general)

        # Process phase and counter
        phase = estimate_phase(self.phase_window,frame_features["elbow_angle"]) 
        frame_features["phase"] = phase
        complete = self.counter.update(phase)

        # Process feedback
        feedback = give_feedback(frame_features)
        frame_features["coach"] = feedback
        self._update_feedback(feedback)

        # Store current frame
        self.sequence.append(frame_features)

        if complete:
            self._predict_sequence()
            self.sequence.clear()
  
    def stop(self):
        self._stop_event.set()
        self.sequence.clear()
        self.counter.reset()
        self.phase_window.clear()
        if self._worker.is_alive():
            self._worker.join(timeout=1)

    def _predict_sequence(self):
        
        # Build features
        X = transform(pd.DataFrame(self.sequence))

        # Predict
        pred = self.model.predict(X)
        pred_label = str(pred[0])
        proba = self.model.predict_proba(X)
        confidence = float(max(proba[0]))

        # Update
        self._update_state(pred_label=pred_label, confidence=confidence)

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
        return joblib.load(MODEL_PATH)
    
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
