import numpy as np
import pytest

from app.ml.model_loader import ModelLoader, get_model_loader

pytestmark = pytest.mark.unit


class TestModelLoader:
    def setup_method(self) -> None:
        self.loader = ModelLoader()

    def test_initial_state_not_loaded(self) -> None:
        assert self.loader.is_loaded is False
        assert self.loader.version == ""
        assert self.loader.metadata == {}

    def test_load_existing_model(self) -> None:
        self.loader.load("risk_model_v1")
        assert self.loader.is_loaded is True
        assert self.loader.version == "risk_model_v1"
        assert isinstance(self.loader.metadata, dict)
        assert "model_type" in self.loader.metadata

    def test_load_nonexistent_model(self) -> None:
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            self.loader.load("nonexistent_model")

    def test_predict_proba_before_load(self) -> None:
        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        with pytest.raises(RuntimeError, match="Model not loaded"):
            self.loader.predict_proba(features)

    def test_predict_proba_after_load(self) -> None:
        self.loader.load("risk_model_v1")
        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        score = self.loader.predict_proba(features)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_proba_returns_probability_of_class_1(self) -> None:
        self.loader.load("risk_model_v1")
        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        score = self.loader.predict_proba(features)
        assert 0.0 <= score <= 1.0


class TestGetModelLoader:
    def test_returns_same_instance(self) -> None:
        loader1 = get_model_loader()
        loader2 = get_model_loader()
        assert loader1 is loader2

    def test_singleton_loads_model(self) -> None:
        loader = get_model_loader()
        loader.load("risk_model_v1")
        assert loader.is_loaded is True
