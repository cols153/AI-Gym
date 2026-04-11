import numpy as np

# Phase helper
def estimate_phase(angle_window, current_angle):
    if current_angle is None or np.isnan(current_angle):
        return "unknown"

    angle_window.append(current_angle)
    smoothed_angle = sum(angle_window) / len(angle_window)

    if smoothed_angle < 90:
        return "bottom"
    if smoothed_angle > 140:
        return "top"
    return "transition"

def give_feedback(features):
    messages = []

    elbow = features.get("elbow_angle")
    body = features.get("body_angle")
    hip = features.get("hip_angle")
    phase = features.get("phase")

    # Safety checks
    if elbow is None or body is None or hip is None:
        return "Tracking..."

    # Torso / back alignment
    if body < 135:
        messages.append("Hips too low")
    elif body < 160:
        messages.append("Back not straight")

    # Hip position
    if hip < 150:
        messages.append("Hips too high")

    # Depth
    if elbow > 110 and phase in ("bottom", "transition"):
        messages.append("Not low enough")

    if not messages:
        return "Keep going"

    return " | ".join(messages)