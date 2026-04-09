import pandas as pd


def transform(sequence_df):
    sequence_df.drop(columns=["phase"], errors="ignore", inplace=True)

    numeric_cols = sequence_df.select_dtypes(include="number").columns
    sequence_df.loc[:, numeric_cols] = (
        sequence_df[numeric_cols]
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )

    X = pd.DataFrame([{
        "elbow_mean": sequence_df["elbow_angle"].mean(),
        "elbow_std": sequence_df["elbow_angle"].std(),
        "body_mean": sequence_df["body_angle"].mean(),
        "hip_mean": sequence_df["hip_angle"].mean(),
    }])

    return X