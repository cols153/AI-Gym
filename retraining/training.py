from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = ["elbow_mean", "elbow_std", "body_mean", "hip_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a MediaPipe 2D MLP from the app-pipeline feature CSV, "
            "matching the notebook training setup."
        )
    )
    parser.add_argument(
        "--features-input",
        type=Path,
        default=ROOT / "data/processed/mediapipe_2d_app_pipeline_features.csv",
        help="Feature CSV created by create_features.py.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=ROOT / "data/models/pushup_2d_app_pipeline.joblib",
        help="Where to save the trained sklearn pipeline.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split and MLP training.",
    )
    return parser.parse_args()


def load_training_dataframe(features_input: Path) -> pd.DataFrame:
    if not features_input.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_input}")

    df = pd.read_csv(features_input)

    missing = [column for column in FEATURE_COLUMNS + ["label"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=FEATURE_COLUMNS).copy()

    if df.empty:
        raise ValueError("No trainable rows remain after filtering empty feature rows.")

    if df["label"].nunique() < 2:
        raise ValueError("Training data must contain at least two label classes.")

    return df


def build_model(random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=16,
            learning_rate_init=1e-3,
            max_iter=1000,
            random_state=random_state,
        )),
    ])


def train_model(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[Pipeline, dict]:
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classes": list(model.classes_),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=model.classes_,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=model.classes_,
        ),
    }
    return model, metrics


def backup_existing_model(model_output: Path) -> Path | None:
    if not model_output.exists():
        return None

    backup_path = model_output.with_name(f"{model_output.stem}_old{model_output.suffix}")
    model_output.replace(backup_path)
    return backup_path


def main() -> None:
    args = parse_args()
    df = load_training_dataframe(args.features_input)
    model, metrics = train_model(
        df=df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    backup_path = backup_existing_model(args.model_output)
    joblib.dump(model, args.model_output)

    print(f"Loaded features from: {args.features_input}")
    print(f"Filtered training rows: {len(df)}")
    print("Label counts:")
    print(df["label"].value_counts().to_string())
    print()
    print("Feature columns used:")
    print(", ".join(FEATURE_COLUMNS))
    print()
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Classes:")
    print(metrics["classes"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print()
    print("Classification report:")
    print(metrics["classification_report"])
    if backup_path is not None:
        print(f"Backed up previous model to: {backup_path}")
    print(f"Saved model to: {args.model_output}")


if __name__ == "__main__":
    main()
