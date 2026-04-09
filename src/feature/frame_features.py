import numpy as np

# Main entry point
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

    # Phase
    phase = _estimate_phase(elbow_angle)

    # Coach message
    feedback = _give_feedback(elbow_angle, body_angle, hip_angle, phase)

    return {
        "elbow_angle": elbow_angle,
        "body_angle": body_angle,
        "hip_angle": hip_angle,
        "phase": phase,
        "coach": feedback,
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

# Phase helper
def _estimate_phase(elbow_angle):
    if np.isnan(elbow_angle):
        return "unknown"
    if elbow_angle < 90:
        return "bottom"
    if elbow_angle > 140:
        return "top"
    return "transition"

def _give_feedback(elbow, body, hip, phase):
    messages = []

    if body < 150:
        messages.append("Back not straight")
    if hip < 150:
        messages.append("Hips too high")
    if body < 135:
        messages.append("Hips too low")
    if elbow > 110 and phase in ["bottom", "transition"]:
        messages.append("Not low enough")

    if not messages:
        return "Keep going"

    return " | ".join(messages)