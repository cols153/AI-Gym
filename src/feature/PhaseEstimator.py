import pandas as pd

class PhaseEstimator:
    def __init__(self, config: dict, angle_mode: str = "3d"):
        rep_cfg = config["rep_phase"]

        self.angle_column = rep_cfg["angle_columns"][angle_mode]
        self.bottom_threshold = rep_cfg["bottom_threshold"]
        self.top_threshold = rep_cfg["top_threshold"]
        self.labels = rep_cfg["labels"]

    def estimate(self, angle: float) -> str:
        if pd.isna(angle):
            return self.labels["unknown"]
        if angle < self.bottom_threshold:
            return self.labels["bottom"]
        if angle > self.top_threshold:
            return self.labels["top"]
        return self.labels["transition"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rep_phase"] = df[self.angle_column].apply(self.estimate)
        return df