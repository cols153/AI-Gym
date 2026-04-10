# AI-Gym

This repository contains:
- a real-time Streamlit application (AI-Gym)
- an offline pipeline for model development (posture_checker_offline)


A Streamlit app for real-time push-up coaching using webcam-based pose detection.

- **`MediaPipePose`** = pose detection logic
- **`PoseProcessor`** = handles webcam frames in Streamlit
- **`LiveState`** = shared runtime state / values

## Setup

Clone the repository:

git clone https://github.com/Hayo87/AI-Gym.git  
cd AI-Gym

Create and activate a conda environment (Python 3.10 required):

conda create -n aigym python=3.10 -y  
conda activate aigym

Install dependencies:

python -m pip install --upgrade pip  
python -m pip install -r requirements.txt

## Run the app

python -m streamlit run app.py

Then open your browser at:

http://localhost:8501

Allow webcam access when prompted.

## Notes

- Python 3.10 is required (compatibility with streamlit-webrtc)
- Make sure your webcam is enabled
- Stand far enough from the camera for proper pose detection 🙂
- The models used in this app were developed in the [posture_checker_offline](./posture_checker_offline) module, which contains the full pipeline for pose processing, feature engineering, and model training