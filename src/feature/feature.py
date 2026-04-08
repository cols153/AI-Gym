import numpy as np
import pandas as pd
import yaml

# Main entry point
def extract_features(pose_df, exercise="pushup", mode=2):
    # Open config    
    with open(f"data/exercises/{exercise}.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Extract features    
    rows = [extract_features_from_row(row, config, mode) for _, row in pose_df.iterrows()]
    features = pd.DataFrame(rows)

    # Apply smoothing
    return smooth_features(features, config)

def extract_features_from_row(row, config, mode=2):

    out = {
        "frame": row["frame"],
        "timestamp_ms": row["timestamp_ms"]
    }

    for side in ["left", "right"]:

        # visibility
        if "visibility" in config:
            for feature_name, spec in config["visibility"].items():
                out[f"{side}_{feature_name}"] = get_visibility(row, side, spec["joints"])

        # angles
        if "angles" in config:
            for feature_name, spec in config["angles"].items():
                p1 = get_point(row, side, spec["points"][0], mode)
                p2 = get_point(row, side, spec["points"][1], mode)
                p3 = get_point(row, side, spec["points"][2], mode)

                out[f"{side}_{feature_name}"] = compute_angle(p1, p2, p3,mode)

        # distances
        if "distances" in config:
            for feature_name, spec in config["distances"].items():
                p1 = get_point(row, side, spec["points"][0], mode)
                p2 = get_point(row, side, spec["points"][1], mode)

                out[f"{side}_{feature_name}"] = euclidean_distance(p1, p2,mode)

        # ratios
        if "ratios" in config:
            for feature_name, spec in config["ratios"].items():
                num = out.get(f"{side}_{spec['numerator']}", np.nan)
                den = out.get(f"{side}_{spec['denominator']}", np.nan)

                out[f"{side}_{feature_name}"] = (
                    num / den if pd.notna(den) and den > 1e-8 else np.nan
                )

    return out

def smooth_features(features, config):
    for signal in config.get("smoothing", {}).get("signals", []):
        for side in ["left", "right"]:
            col = f"{side}_{signal}"
            if col in features.columns:
                features[f"{col}_smooth"] = (
                    features[col]
                    .rolling(window=5, center=True, min_periods=1)
                    .mean()
                )
    return features            

# Math helpers
def euclidean_distance(p1, p2, dims=2):
    p1 = np.array(p1[:dims], dtype=float)
    p2 = np.array(p2[:dims], dtype=float)
    return np.linalg.norm(p1 - p2)

def compute_angle(a, b, c, dims=2):
    a = np.array(a[:dims], dtype=float)
    b = np.array(b[:dims], dtype=float)
    c = np.array(c[:dims], dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return np.nan

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def get_point(row, side, joint, dims=3):
    axes = ["x", "y"] if dims == 2 else ["x", "y", "z"]
    return [row[f"{side}_{joint}_{axis}"] for axis in axes]

def get_visibility(row, side, joints):
    vals = [row[f"{side}_{joint}_visibility"] for joint in joints]
    vals = [v for v in vals if pd.notna(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))
