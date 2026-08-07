from typing import Protocol

import numpy as np


class PatternDetector(Protocol):
    def detect(self, values: np.ndarray) -> dict[str, object]: ...
