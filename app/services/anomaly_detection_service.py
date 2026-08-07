import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetectionService:
    def __init__(self) -> None:
        self._model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
        )
        self._is_fitted = False

    def fit(self, training_data: np.ndarray) -> None:
        self._model.fit(training_data)
        self._is_fitted = True

    def detect(self, readings_matrix: np.ndarray) -> dict[str, object]:
        if not self._is_fitted:
            self.fit(readings_matrix)

        predictions = self._model.predict(readings_matrix)
        scores = self._model.decision_function(readings_matrix)

        anomaly_indices = np.where(predictions == -1)[0].tolist()
        anomaly_score = float(1 - np.min(scores)) if len(scores) > 0 else 0.0
        anomaly_score = max(0.0, min(1.0, anomaly_score))

        return {
            "anomalyDetected": len(anomaly_indices) > 0,
            "anomalyScore": round(anomaly_score, 3),
            "affectedReadingsIndexes": anomaly_indices,
        }
