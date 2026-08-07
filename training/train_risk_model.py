import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.ml.feature_engineering import build_feature_vector


def train(
    dataset_path: str = "dataset_synthetic.csv",
    window_size: int = 12,
    output_dir: str = "app/ml/model_registry",
) -> None:
    df = pd.read_csv(dataset_path)

    features_list = []
    labels = []

    for patient_id in range(0, len(df), 100):
        chunk = df.iloc[patient_id : patient_id + 100]
        if len(chunk) < window_size:
            continue

        hr = chunk["heartRate"].values
        o2 = chunk["oxygen"].values
        act = chunk["activity"].values
        label = chunk["label"].iloc[0]

        feat = build_feature_vector(hr, o2, act, window_size)
        features_list.append(feat)
        labels.append(label)

    x = np.array(features_list)
    y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"ROC AUC:  {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / "risk_model_v1.joblib")

    metadata = {
        "version": "risk_model_v1",
        "trained_at": pd.Timestamp.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
        },
        "features": [
            "hr_mean",
            "hr_std",
            "hr_slope",
            "o2_mean",
            "o2_std",
            "o2_slope",
            "act_mean",
            "act_std",
            "act_slope",
        ],
        "window_size": window_size,
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
        },
        "dataset_hash": "",
        "notes": "Trained on synthetic dataset",
    }

    with open(out / "metadata_v1.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {out / 'risk_model_v1.joblib'}")
    print(f"Metadata saved to {out / 'metadata_v1.json'}")


if __name__ == "__main__":
    train()
