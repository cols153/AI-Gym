import tempfile
from pipeline.pose.pose import extract_landmarks
from pipeline.feature import extract_features
from pipeline.logic import run_logic

def run_video_pipeline(input_source, draw=False, model="Rule", mode=2):

    # 1. Normalize input
    if hasattr(input_source, "read"):  # uploaded file
        video_path = save_uploaded_file(input_source)
    else:
        video_path = input_source  # already a path
    
    # 2. Detect landmarks
    pose_df = extract_landmarks(video_path)

    # 3. Extract features
    feature_df = extract_features(pose_df, mode=mode)

    # 4. Run predictor logic
    # logic_df = run_logic(pose_df, feature_df)

    return input_source, feature_df


def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name