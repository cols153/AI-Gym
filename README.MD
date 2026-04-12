# AI-Gym

AI-Gym is a Streamlit app for real-time push-up coaching using webcam-based pose detection and a MediaPipe 2D classification pipeline.

This repository contains:
- the live Streamlit app
- the active application pipeline in `src/`
- a retraining workflow in `retraining/`
- the original experimental notebooks and offline analysis in `posture_checker_offline/`

## Final App Pipeline

The final application is explicitly **2D-only**.

The live app uses:
- MediaPipe pose detection
- 2D landmark-based feature extraction
- repetition-level feature aggregation
- the trained model in `data/models/pushup_2d.joblib`

## Setup

Clone the repository:

```bash
git clone https://github.com/Hayo87/AI-Gym.git
cd AI-Gym
```

Create and activate a conda environment:

```bash
conda create -n aigym python=3.10 -y
conda activate aigym
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the App

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Allow webcam access when prompted.

## Project Structure

- `app.py`
  Streamlit front-end for the live push-up coach.
- `src/`
  Core application pipeline, pose processing, feedback logic, and shared utilities.
- `retraining/`
  Scripts to regenerate features, retrain the 2D model, and run evaluation on filtered videos.
- `data/models/`
  Saved models used by the app.
- `data/processed/`
  Generated feature CSV files used for retraining.
- `posture_checker_offline/`
  Experimental notebooks, offline analysis, and original project artifacts.

## Retraining Workflow

The retraining workflow lives in `retraining/` and reuses the live app pipeline so that training and inference stay aligned.

### 1. Generate Features From Raw Videos

This step:
- reads the raw videos from:
  - `posture_checker_offline/data/raw/videos/correct`
  - `posture_checker_offline/data/raw/videos/incorrect`
- runs them through the same app-side pose processing route
- creates a repetition-level feature CSV
- still writes a placeholder row for videos where no full repetition was detected

Run:

```powershell
.\venv\Scripts\python.exe retraining\create_features.py
```

Output:

```text
data/processed/mediapipe_2d_app_pipeline_features.csv
```

### 2. Train a New 2D Model

This step:
- reads the generated feature CSV
- removes rows without full repetition features
- trains an MLP on:
  - `elbow_mean`
  - `elbow_std`
  - `body_mean`
  - `hip_mean`
- saves the model for the live app

Run:

```powershell
.\venv\Scripts\python.exe retraining\training.py --model-output data/models/pushup_2d.joblib
```

The training pipeline follows the notebook-style sklearn workflow:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`
- `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, ...)`

If a model already exists at the target path, the training script creates a backup with the `_old` suffix first.

### 3. Evaluate on Filtered Videos

The retraining test runner:
- reads `data/processed/mediapipe_2d_app_pipeline_features.csv`
- keeps only videos where at least one repetition was detected
- reruns those videos through the live pipeline
- checks whether correct videos stay correct and incorrect videos are detected as incorrect

Run:

```powershell
.\venv\Scripts\python.exe -m pytest retraining\test_runner.py -s
```

## Notes on Metrics

- Metrics reported by `retraining/training.py` are computed on the **filtered repetition-level feature dataset**.
- These metrics are useful for model comparison, but they are not identical to full end-to-end app performance on raw videos or live webcam input.
- End-to-end behavior also depends on detection quality, repetition detection, and the runtime pipeline.

## Notes

- Python 3.10 is required for compatibility with `streamlit-webrtc`.
- Make sure your webcam is enabled.
- Stand far enough from the camera for reliable pose detection.
- The original model-development work is preserved in [posture_checker_offline](./posture_checker_offline).
