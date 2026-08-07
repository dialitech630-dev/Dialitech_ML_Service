from typing import Protocol

import numpy as np


class AnomalyDetector(Protocol):
    def detect(self, readings_matrix: np.ndarray) -> dict[str, object]: ...
