# AI-Gym

This repository contains:
- A Streamlit app for real-time push-up coaching using webcam-based pose detection.
- An offline pipeline for model development (posture_checker_offline)


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

## Retraining Workflow

Generate a new feature dataset from the raw videos:

`.\venv\Scripts\python.exe retraining\create_features.py`

Train a new 2D MediaPipe model from the generated feature CSV:

`.\venv\Scripts\python.exe retraining\training.py --model-output data/models/pushup_2d.joblib`

Run the filtered video-based evaluation on videos where at least one rep was detected:

`.\venv\Scripts\python.exe -m pytest retraining\test_runner.py -s`

## Notes

- Python 3.10 is required (compatibility with streamlit-webrtc)
- Make sure your webcam is enabled
- Stand far enough from the camera for proper pose detection 🙂
- The models used in this app were developed in the [posture_checker_offline](./posture_checker_offline) module, which contains the full pipeline for pose processing, feature engineering, and model training
