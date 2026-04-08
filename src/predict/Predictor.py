import joblib
import pandas as pd
from pathlib import Path


class Predictor:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        return joblib.load(self.model_path)

    def predict(self, feature_dict: dict):
        df = pd.DataFrame([feature_dict])
        return self.model.predict(df)[0]

    def predict_proba(self, feature_dict: dict):
        df = pd.DataFrame([feature_dict])
        return self.model.predict_proba(df)[0]