# Push-up Posture Checker

## Demo

Comparison of two pose estimation approaches:

* **MediaPipe** (fast, local)
* **MMPose** (GPU-based)

[Watch demo video](https://github.com/Hayo87/AI-Gym/releases/download/v1.0/annotated_comparison_side_by_side.mp4)

---

## Overview

This project implements a complete pipeline to analyze push-up posture using computer vision and machine learning.

The system:

* extracts body pose from videos
* engineers meaningful biomechanical features
* classifies push-up quality (correct / incorrect)
* provides visual and textual feedback

---

## Pipeline

### 1. Pose Estimation

Two approaches are used:

* **MediaPipe**

  * fast and lightweight
  * runs locally in real time
  * provides 2D landmarks + approximate 3D world coordinates

* **MMPose**

  * supports 2D and 3D keypoints
  * requires GPU (Google Colab)
  * heavier but flexible

Outputs:

* MediaPipe → `data/interim/landmarks_json/`
* MMPose → `data/interim/keypoints_json/`

---

### 2. Feature Engineering

Transforms raw pose data into structured features:

* joint angles (elbow, hip, body alignment)
* posture consistency
* movement dynamics

See report for full feature description.

Outputs:

* `data/processed/*.csv`

---

### 3. Posture Classification

Three approaches are compared:

* Rule-based
* Logistic Regression (baseline)
* MLP (neural network)

Tasks:

* classify push-ups as correct / incorrect
* detect posture issues

---

### 4. Video Visualization

* skeleton overlay on video
* repetition counting
* real-time feedback:

  * correctness
  * posture suggestions

Outputs:

* annotated videos in `outputs/`

---

## Trained Models

The project includes trained machine learning pipelines based on MediaPipe features:

* MLP (2D features)
  → `models/mlp_pipeline_mediapipe_2d.joblib`

* MLP (3D features)
  → `models/mlp_pipeline_mediapipe_3d.joblib`

Each pipeline includes:

* preprocessing
* feature scaling
* trained classifier

Example usage:

```python
import joblib

model = joblib.load("models/mlp_pipeline_mediapipe_2d.joblib")
```

---

## Integration with Real-Time App

These models are designed to be used in a real-time pipeline:

* pose detection → MediaPipe
* feature extraction → custom pipeline
* inference → trained model (`joblib`)

Note: the real-time Streamlit application is developed separately.
The final system uses a hybrid approach:
- MLP models for posture classification
- rule-based logic for repetition counting and feedback generation

This ensures both robustness and interpretability in real-time.

---

## Key Insights

* MediaPipe provides more stable and interpretable features
* MMPose features are noisier and require more processing
* Rule-based methods remain strong baselines
* ML models can benefit from 3D representations, but this depends on pose estimation quality.
* 3D does not always outperform 2D — depends on pose estimation quality
* MediaPipe offers the best trade-off between performance, speed, and usability

---

## Project Structure

```text
posture_checker_offline/
│
├── data/
│   ├── raw/
│   │   └── videos/
│   │       ├── correct/
│   │       ├── incorrect/
│   │       └── demo/
│   │
│   ├── interim/
│   │   ├── landmarks_json/
│   │   └── keypoints_json/
│   │
│   └── processed/
│       ├── *.csv
│
├── notebooks/
│   ├── 01_pose_estimation_mediapipe.ipynb
│   ├── 01_pose_estimation_mmpose_colab.ipynb
│   ├── 02_feature_engineering_mediapipe.ipynb
│   ├── 02_feature_engineering_mmpose.ipynb
│   ├── 03_mediaPipe_posture_checker_development.ipynb
│   ├── 03_MmPose_posture_checker_development.ipynb
│   ├── 04_comparison_mediapipe_vs_mmpose_and_2d_vs_3d.ipynb
│   ├── 05_video_inference_and_overlay_mediapipe.ipynb
│   ├── 05a_pose_estimation_2D_mmpose_colab.ipynb
│   ├── 05a_pose_estimation_3D_mmpose_colab.ipynb
│   ├── 05b_video_overlay_mmpose.ipynb
│
├── models/
│   ├── mlp_pipeline_mediapipe_2d.joblib
│   └── mlp_pipeline_mediapipe_3d.joblib
│
├── outputs/
│   ├── annotated_pushup.mp4
│   ├── annotated_pushup_mmpose.mp4
│   └── annotated_comparison_side_by_side.mp4
│
└── README.md
```

---

## Usage

Run the notebooks in order:

1. Pose estimation
   → `01_*`

2. Feature engineering
   → `02_*`

3. Model development
   → `03_*`

4. Comparison and evaluation
   → `04_*`

5. Inference and visualization
   → `05_*`

---

## Dataset

This project uses the following dataset:

Riccio, R. (2023). *Real-Time Exercise Recognition Dataset*
https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset

License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)

The dataset is used for educational and non-commercial purposes only.

---

## Notes

* MMPose notebooks require **Google Colab (GPU)**
* MediaPipe runs locally and is significantly faster
* Large datasets are not included in the repository
* The dataset is subject to a non-commercial license

---

## Author

Project developed as part of the **Deep Neural Engineering** course.
