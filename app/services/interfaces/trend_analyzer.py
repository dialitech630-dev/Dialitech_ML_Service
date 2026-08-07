from typing import Protocol

import numpy as np


class TrendAnalyzer(Protocol):
    def analyze(self, values: np.ndarray) -> dict[str, object]: ...
