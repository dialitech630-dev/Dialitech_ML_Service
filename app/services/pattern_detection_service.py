import numpy as np


class PatternDetectionService:
    def detect(self, values: np.ndarray) -> dict[str, object]:
        if len(values) < 4:
            return {"patternsFound": False, "patterns": []}

        patterns = []

        if self._detect_oscillation(values):
            patterns.append(
                {
                    "type": "PERIODIC_OSCILLATION",
                    "description": "Repetitive oscillatory pattern detected",
                    "confidence": 0.71,
                }
            )

        if self._detect_high_variability(values):
            patterns.append(
                {
                    "type": "HIGH_VARIABILITY",
                    "description": "Unusually high variability in readings",
                    "confidence": 0.65,
                }
            )

        return {
            "patternsFound": len(patterns) > 0,
            "patterns": patterns,
        }

    def _detect_oscillation(self, values: np.ndarray) -> bool:
        diffs = np.diff(values)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        oscillation_ratio = sign_changes / (len(diffs) - 1) if len(diffs) > 1 else 0
        return oscillation_ratio > 0.5

    def _detect_high_variability(self, values: np.ndarray) -> bool:
        if len(values) < 4:
            return False
        cv = np.std(values) / (np.mean(values) + 1e-10)
        return bool(cv > 0.3)
