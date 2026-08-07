import numpy as np


def compute_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        raise ValueError(f"Need at least {window} values, got {len(values)}")
    result = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def compute_rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        raise ValueError(f"Need at least {window} values, got {len(values)}")
    result = np.array(
        [np.std(values[i : i + window]) for i in range(len(values) - window + 1)]
    )
    return result


def compute_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        raise ValueError(f"Need at least 2 values, got {len(values)}")
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    return float(slope)


def compute_range(values: np.ndarray) -> float:
    if len(values) < 2:
        raise ValueError(f"Need at least 2 values, got {len(values)}")
    return float(np.max(values) - np.min(values))


def build_feature_vector(
    heart_rates: np.ndarray,
    oxygens: np.ndarray,
    activities: np.ndarray,
    window: int,
) -> np.ndarray:
    if not (len(heart_rates) == len(oxygens) == len(activities)):
        raise ValueError("All input arrays must have the same length")

    if len(heart_rates) < window:
        raise ValueError(f"Need at least {window} readings, got {len(heart_rates)}")

    features = np.array(
        [
            np.mean(heart_rates[-window:]),
            np.std(heart_rates[-window:]),
            compute_slope(heart_rates[-window:]),
            np.mean(oxygens[-window:]),
            np.std(oxygens[-window:]),
            compute_slope(oxygens[-window:]),
            np.mean(activities[-window:]),
            np.std(activities[-window:]),
            compute_slope(activities[-window:]),
        ]
    )

    return features
