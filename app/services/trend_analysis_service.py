import numpy as np

from app.ml.feature_engineering import compute_slope


class TrendAnalysisService:
    def analyze(self, values: np.ndarray) -> dict[str, object]:
        if len(values) < 2:
            return {
                "direction": "STABLE",
                "slope": 0.0,
                "confidence": 0.0,
            }

        slope = compute_slope(values)
        normalized_slope = slope / (np.mean(values) + 1e-10)
        confidence = min(abs(normalized_slope) * 10, 1.0)

        if abs(slope) < 0.05:
            direction = "STABLE"
        elif slope > 0:
            direction = "ASCENDING"
        else:
            direction = "DESCENDING"

        return {
            "direction": direction,
            "slope": round(float(slope), 3),
            "confidence": round(float(confidence), 3),
        }
