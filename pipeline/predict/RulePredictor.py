class RulePredictor:
    def predict(self, df):
        row = df.iloc[0]
        elbow_min = row.get("elbow_angle_min", 180)
        hip_drop = row.get("hip_drop", 0)

        if elbow_min > 110:
            return ["bad_depth"]
        if hip_drop > 0.2:
            return ["bad_hips"]
        return ["good"]

    def predict_proba(self, df):
        pred = self.predict(df)[0]
        labels = ["good", "bad_depth", "bad_hips"]
        probs = [1.0 if label == pred else 0.0 for label in labels]
        return [probs]