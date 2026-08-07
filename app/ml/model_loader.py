import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_REGISTRY_DIR = Path(__file__).parent / "model_registry"


class ModelLoader:
    def __init__(self) -> None:
        self._model: Any = None
        self._metadata: dict[str, Any] = {}
        self._version: str = ""

    def load(self, version: str) -> None:
        model_path = MODEL_REGISTRY_DIR / f"{version}.joblib"
        metadata_path = MODEL_REGISTRY_DIR / "metadata_v1.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        self._model = joblib.load(model_path)
        self._version = version

        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

        logger.info("Loaded model version %s", version)

    def predict_proba(self, features: np.ndarray) -> float:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        proba = self._model.predict_proba(features.reshape(1, -1))[0]
        return float(proba[1])

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._version

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


_model_loader: ModelLoader | None = None


def get_model_loader() -> ModelLoader:
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
