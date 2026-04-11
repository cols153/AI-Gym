import numpy as np
import pandas as pd


# Main entry point numeric features
def extract(pose_result, mode):

    # Select best side
    side = _choose_body_side(pose_result)
    
    # Extract features    
    features = _extract(pose_result, side, mode)

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
def _extract(pose_result, side, mode):
    
    # Elbow
    a = _get_point(pose_result, side, "shoulder", mode)
    b = _get_point(pose_result, side, "elbow", mode)
    c = _get_point(pose_result, side, "wrist", mode)
    elbow_angle = _compute_angle(a, b, c, mode)

    # Body alignment
    a = _get_point(pose_result, side, "shoulder", mode)
    b = _get_point(pose_result, side, "hip", mode)
    c = _get_point(pose_result, side, "ankle", mode)
    body_angle = _compute_angle(a, b, c, mode)

    # Hip
    a = _get_point(pose_result, side, "shoulder", mode)
    b = _get_point(pose_result, side, "hip", mode)
    c = _get_point(pose_result, side, "knee", mode)
    hip_angle = _compute_angle(a, b, c, mode)

    return {
        "elbow_angle": elbow_angle,
        "body_angle": body_angle,
        "hip_angle": hip_angle,
    }

# Math helpers
def _compute_angle(a, b, c, dims):
    a = np.array(a[:dims], dtype=float)
    b = np.array(b[:dims], dtype=float)
    c = np.array(c[:dims], dtype=float)

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

def _get_point(row, side, joint, dims):
    landmark = row.get(f"{side}_{joint}")

    if landmark is None:
        return None

    if dims == 2:
        return [landmark.get("x", np.nan), landmark.get("y", np.nan)]

    return [
        landmark.get("x", np.nan),
        landmark.get("y", np.nan),
        landmark.get("z", np.nan),
    ]

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