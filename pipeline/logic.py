import numpy as np
import pandas as pd

def run_logic(pose_df, feature_df):
    df = select_side_features(pose_df, feature_df)
    df = detect_phase(df)
    df = count_reps(df)
    df = check_form(df)
    return df

def choose_visible_side(row):
    left_score = np.nanmean([
        row.get("left_shoulder_visibility", np.nan),
        row.get("left_elbow_visibility", np.nan),
        row.get("left_wrist_visibility", np.nan),
        row.get("left_hip_visibility", np.nan),
        row.get("left_knee_visibility", np.nan),
        row.get("left_ankle_visibility", np.nan),
    ])

    right_score = np.nanmean([
        row.get("right_shoulder_visibility", np.nan),
        row.get("right_elbow_visibility", np.nan),
        row.get("right_wrist_visibility", np.nan),
        row.get("right_hip_visibility", np.nan),
        row.get("right_knee_visibility", np.nan),
        row.get("right_ankle_visibility", np.nan),
    ])

    if np.isnan(left_score) and np.isnan(right_score):
        return np.nan

    return "left" if left_score >= right_score else "right"

def select_side_features(pose_df, feature_df):
    df = feature_df.copy()

    df["selected_side"] = pose_df.apply(choose_visible_side, axis=1)

    paired_features = [
        "elbow_angle",
        "shoulder_angle",
        "hip_angle",
        "knee_angle",
        "body_angle",
    ]

    for feature in paired_features:
        df[feature] = df.apply(
            lambda row: row[f"left_{feature}"] if row["selected_side"] == "left"
            else row[f"right_{feature}"] if row["selected_side"] == "right"
            else np.nan,
            axis=1,
        )

    return df

def detect_phase(df, down_threshold=90, up_threshold=160):
    def get_phase(angle):
        if np.isnan(angle):
            return np.nan
        if angle < down_threshold:
            return "down"
        elif angle > up_threshold:
            return "up"
        else:
            return "mid"

    df["phase"] = df["elbow_angle"].apply(get_phase)
    return df

def count_reps(df):
    reps = 0
    prev = None
    rep_list = []

    for phase in df["phase"]:
        if prev == "down" and phase == "up":
            reps += 1
        rep_list.append(reps)
        prev = phase

    df["rep_count"] = rep_list
    return df

def check_form(df):
    df["bad_body_line"] = df["body_angle"] < 150
    df["bad_depth"] = df["elbow_angle"] > 100  # not going deep enough
    return df