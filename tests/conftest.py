import numpy as np
import pytest


@pytest.fixture
def sample_readings() -> dict:
    rng = np.random.default_rng(42)
    n = 20
    return {
        "heartRates": rng.uniform(60, 100, n).tolist(),
        "oxygens": rng.uniform(94, 99, n).tolist(),
        "activities": rng.uniform(20, 80, n).tolist(),
    }


@pytest.fixture
def minimal_readings() -> dict:
    return {
        "heartRates": [70.0, 72.0, 75.0, 78.0],
        "oxygens": [97.0, 96.5, 96.0, 95.5],
        "activities": [40.0, 35.0, 30.0, 25.0],
    }
