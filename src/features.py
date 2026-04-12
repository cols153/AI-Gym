import numpy as np
import pandas as pd


# Main entry point numeric features
def extract(pose_result):

    # Select best side
    side = _choose_body_side(pose_result)
    
    # Extract features    
    features = _extract(pose_result, side)

    return features

# Helper to select side
def _choose_body_side(pose_result):
    joints = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]

    left_score = _get_visibility(pose_result, "left", joints)
    right_score = _get_visibility(pose_result, "right", joints) 

    return "left" if left_score >= right_score else "right"

# Helper to get side visibility score
def _get_visibility(pose_result, side, joints):
    vals = [
        pose_result.get(f"{side}_{joint}", {}).get("visibility", np.nan)
        for joint in joints
    ]
    vals = [v for v in vals if not np.isnan(v)]

    if len(vals) == 0:
        return np.nan

    return float(np.mean(vals))

# Helper extract features
def _extract(pose_result, side):
    
    # Elbow
    a = _get_point(pose_result, side, "shoulder")
    b = _get_point(pose_result, side, "elbow")
    c = _get_point(pose_result, side, "wrist")
    elbow_angle = _compute_angle(a, b, c)

    # Body alignment
    a = _get_point(pose_result, side, "shoulder")
    b = _get_point(pose_result, side, "hip")
    c = _get_point(pose_result, side, "ankle")
    body_angle = _compute_angle(a, b, c)

    # Hip
    a = _get_point(pose_result, side, "shoulder")
    b = _get_point(pose_result, side, "hip")
    c = _get_point(pose_result, side, "knee")
    hip_angle = _compute_angle(a, b, c)

    return {
        "elbow_angle": elbow_angle,
        "body_angle": body_angle,
        "hip_angle": hip_angle,
    }

# Math helpers
def _compute_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    # Check for invalid input
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return np.nan

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return np.nan

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))

def _get_point(row, side, joint):
    landmark = row.get(f"{side}_{joint}")

    if landmark is None:
        return None

    return [landmark.get("x", np.nan), landmark.get("y", np.nan)]

# Main entry point sequence features
def transform(sequence_df):
    sequence_df.drop(columns=["phase"], errors="ignore", inplace=True)

    numeric_cols = sequence_df.select_dtypes(include="number").columns
    sequence_df.loc[:, numeric_cols] = (
        sequence_df[numeric_cols]
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )

    X = pd.DataFrame([{
        "elbow_mean": sequence_df["elbow_angle"].mean(),
        "elbow_std": sequence_df["elbow_angle"].std(),
        "body_mean": sequence_df["body_angle"].mean(),
        "hip_mean": sequence_df["hip_angle"].mean(),
    }])

    return X
