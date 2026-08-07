from typing import Protocol

import numpy as np


class RiskPredictor(Protocol):
    def predict(
        self, features: np.ndarray, patient_id: str
    ) -> dict[str, object]: ...
