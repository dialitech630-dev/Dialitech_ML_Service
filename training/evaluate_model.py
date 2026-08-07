import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from app.ml.feature_engineering import build_feature_vector


def evaluate(
    model_path: str = "app/ml/model_registry/risk_model_v1.joblib",
    dataset_path: str = "dataset_synthetic.csv",
    window_size: int = 12,
) -> None:
    model = joblib.load(model_path)
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

    X = np.array(features_list)
    y = np.array(labels)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("Evaluation Results:")
    print(f"ROC AUC: {roc_auc_score(y, y_proba):.4f}")
    print()
    print(classification_report(y, y_pred))


if __name__ == "__main__":
    evaluate()
