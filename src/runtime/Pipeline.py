import queue
import threading
import joblib
import pandas as pd

from src.runtime.Counter import Counter
from src.feature.frame_features import extract
from src.feature.sequence_features import transform

MODEL_2D_PATH = "data/models/pushup_2d.joblib"
MODEL_3D_PATH = "data/models/pushup_3d.joblib"

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
        
        # Transform to features
        frame_features = extract(result, self.mode)

        # Store current frame
        self.sequence.append(frame_features)

        # Then update counter logic
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


    def _load_model(self):
        model_path = MODEL_2D_PATH if self.mode == 2 else MODEL_3D_PATH
        return joblib.load(model_path)