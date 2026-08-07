import numpy as np
import pytest

from app.ml.feature_engineering import (
    build_feature_vector,
    compute_range,
    compute_rolling_mean,
    compute_rolling_std,
    compute_slope,
)


class TestComputeRollingMean:
    def test_basic(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_rolling_mean(values, 3)
        assert len(result) == 3
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(4.0)

    def test_insufficient_values(self) -> None:
        with pytest.raises(ValueError, match="Need at least"):
            compute_rolling_mean(np.array([1.0, 2.0]), 3)


class TestComputeRollingStd:
    def test_constant_values(self) -> None:
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = compute_rolling_std(values, 3)
        assert all(s == pytest.approx(0.0) for s in result)

    def test_insufficient_values(self) -> None:
        with pytest.raises(ValueError):
            compute_rolling_std(np.array([1.0]), 3)


class TestComputeSlope:
    def test_ascending(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        slope = compute_slope(values)
        assert slope == pytest.approx(1.0)

    def test_descending(self) -> None:
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        slope = compute_slope(values)
        assert slope == pytest.approx(-1.0)

    def test_single_value(self) -> None:
        with pytest.raises(ValueError):
            compute_slope(np.array([1.0]))


class TestComputeRange:
    def test_basic(self) -> None:
        assert compute_range(np.array([1.0, 5.0, 3.0])) == pytest.approx(4.0)

    def test_single_value(self) -> None:
        with pytest.raises(ValueError):
            compute_range(np.array([1.0]))


class TestBuildFeatureVector:
    def test_valid_input(self) -> None:
        rng = np.random.default_rng(42)
        hr = rng.uniform(60, 100, 12)
        o2 = rng.uniform(94, 99, 12)
        act = rng.uniform(20, 80, 12)
        features = build_feature_vector(hr, o2, act, 12)
        assert features.shape == (9,)

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            build_feature_vector(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0]),
                2,
            )

    def test_insufficient_window(self) -> None:
        with pytest.raises(ValueError, match="Need at least"):
            build_feature_vector(
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                12,
            )
