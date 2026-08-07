# Prompt de Implementación — Dialitech ML Service

> **Instrucción para opencode**: Implementa el microservicio ML completo
> siguiendo **exactamente** la estructura de carpetas, clases, contratos y
> decisiones de diseño que se detallan a continuación. No inventes atajos,
> no mezcles capas, no omitas archivos. Cada archivo debe existir con el
> contenido aquí especificado.

---

## Contexto del Proyecto

Dialitech tiene una API .NET (`API.Dialitech`) que recibe datos de
wearables de pacientes en diálisis, evalúa reglas fijas, y persiste en
MongoDB. Este microservicio ML es un servicio **independiente** (Python +
FastAPI) que se integra con la API .NET existente sin modificarla
estructuralmente.

- **El ML Service NUNCA escribe en MongoDB**. Todo pasa por la API .NET.
- **El ML Service vive en su propio repo, contenedor, y ciclo de despliegue**.
- **Un solo endpoint de análisis** (`POST /api/v1/analyze`) agrupa las 4
  capacidades (patrones, tendencias, riesgo, anomalías).
- **El pipeline de entrenamiento** (`training/`) NO se importa desde `app/`.

---

## Estructura de Carpetas Obligatoria

```
dialitech-ml-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── routes/
│   │       │   ├── __init__.py
│   │       │   ├── analyze.py
│   │       │   ├── health.py
│   │       │   └── model_info.py
│   │       └── dependencies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── security.py
│   │   └── exceptions.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── reading.py
│   │   ├── analyze_request.py
│   │   ├── risk.py
│   │   ├── trend.py
│   │   ├── pattern.py
│   │   └── anomaly.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── interfaces/
│   │   │   ├── __init__.py
│   │   │   ├── risk_predictor.py
│   │   │   ├── trend_analyzer.py
│   │   │   ├── pattern_detector.py
│   │   │   └── anomaly_detector.py
│   │   ├── risk_prediction_service.py
│   │   ├── trend_analysis_service.py
│   │   ├── pattern_detection_service.py
│   │   ├── anomaly_detection_service.py
│   │   └── clinical_analysis_orchestrator.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── model_loader.py
│   │   └── model_registry/
│   │       ├── risk_model_v1.joblib
│   │       └── metadata_v1.json
│   └── utils/
│       ├── __init__.py
│       └── time_windows.py
├── training/
│   ├── generate_dataset.py
│   ├── train_risk_model.py
│   ├── evaluate_model.py
│   └── notebooks/
│       └── exploratory_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_feature_engineering.py
│   │   ├── test_trend_analysis_service.py
│   │   ├── test_pattern_detection_service.py
│   │   ├── test_anomaly_detection_service.py
│   │   ├── test_risk_prediction_service.py
│   │   └── test_clinical_analysis_orchestrator.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_analyze_endpoint.py
│   │   └── test_health_endpoint.py
│   ├── fixtures/
│   │   └── test_model.joblib
│   └── conftest.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── render.yaml
└── README.md
```

---

## Archivos a Implementar (en orden de dependencia)

### 1. `pyproject.toml`

```toml
[project]
name = "dialitech-ml-service"
version = "1.0.0"
description = "Microservicio de Machine Learning para análisis clínico de pacientes en diálisis"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
    "ruff>=0.3.0",
    "mypy>=1.9.0",
    "black>=24.0.0",
    "isort>=5.13.0",
    "bandit>=1.7.0",
    "pip-audit>=2.7.0",
    "hypothesis>=6.100.0",
]
training = [
    "pandas>=2.2.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "jupyter>=1.0.0",
]

[tool.ruff]
target-version = "py311"
line-length = 88
select = ["E", "F", "I", "N", "UP", "B", "A", "SIM", "TCH"]
ignore = ["E501"]

[tool.ruff.isort]
known-first-party = ["app"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.black]
target-version = ["py311"]
line-length = 88

[tool.isort]
profile = "black"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.bandit]
exclude_dirs = ["tests", "training"]
```

---

### 2. `requirements.txt`

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.6.0
pydantic-settings>=2.2.0
scikit-learn>=1.4.0
joblib>=1.3.0
numpy>=1.26.0
```

---

### 3. `requirements-dev.txt`

```
-r requirements.txt
pytest>=8.0.0
pytest-cov>=5.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
ruff>=0.3.0
mypy>=1.9.0
black>=24.0.0
isort>=5.13.0
bandit>=1.7.0
pip-audit>=2.7.0
hypothesis>=6.100.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
```

---

### 4. `.env.example`

```
API_KEY=your-secret-api-key-here
MODEL_VERSION=risk_model_v1
LOG_LEVEL=INFO
WINDOW_SIZE_DEFAULT=12
RISK_THRESHOLD_HIGH=0.7
RISK_THRESHOLD_MEDIUM=0.4
```

---

### 5. `app/__init__.py`

Archivo vacío.

---

### 6. `app/core/__init__.py`

Archivo vacío.

---

### 7. `app/core/config.py`

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración del servicio. Lee de variables de entorno."""

    API_KEY: str
    MODEL_VERSION: str = "risk_model_v1"
    LOG_LEVEL: str = "INFO"
    WINDOW_SIZE_DEFAULT: int = 12
    RISK_THRESHOLD_HIGH: float = 0.7
    RISK_THRESHOLD_MEDIUM: float = 0.4

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
```

---

### 8. `app/core/logging.py`

```python
import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
```

---

### 9. `app/core/security.py`

```python
from fastapi import Depends, Header, HTTPException


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    from app.core.config import settings

    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
```

---

### 10. `app/core/exceptions.py`

```python
from fastapi import Request
from fastapi.responses import JSONResponse


class InsufficientReadingsError(Exception):
    def __init__(self, required: int, actual: int) -> None:
        self.required = required
        self.actual = actual
        super().__init__(
            f"Insufficient readings: {required} required, {actual} provided"
        )


class ModelNotLoadedError(Exception):
    def __init__(self) -> None:
        super().__init__("ML model is not loaded")


class AnalysisError(Exception):
    def __init__(self, detail: str = "Analysis failed") -> None:
        self.detail = detail
        super().__init__(detail)


async def insufficient_readings_handler(
    request: Request, exc: InsufficientReadingsError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "detail": f"Requires at least {exc.required} readings, got {exc.actual}"
        },
    )


async def model_not_loaded_handler(
    request: Request, exc: ModelNotLoadedError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "ML model is not available"},
    )


async def analysis_error_handler(
    request: Request, exc: AnalysisError
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal analysis error"},
    )


EXCEPTION_HANDLERS = {
    InsufficientReadingsError: insufficient_readings_handler,
    ModelNotLoadedError: model_not_loaded_handler,
    AnalysisError: analysis_error_handler,
}
```

---

### 11. `app/schemas/__init__.py`

Archivo vacío.

---

### 12. `app/schemas/reading.py`

```python
from datetime import datetime

from pydantic import BaseModel


class HealthReading(BaseModel):
    heartRate: float
    oxygen: float
    activity: float
    timestamp: datetime
```

---

### 13. `app/schemas/analyze_request.py`

```python
from pydantic import BaseModel, Field

from app.schemas.reading import HealthReading


class AnalyzeRequest(BaseModel):
    patientId: str
    windowSize: int = Field(default=12, ge=1)
    readings: list[HealthReading]
```

---

### 14. `app/schemas/risk.py`

```python
from pydantic import BaseModel


class RiskPrediction(BaseModel):
    riskScore: float
    riskLevel: str  # "LOW", "MEDIUM", "HIGH"
    recommendation: str
```

---

### 15. `app/schemas/trend.py`

```python
from pydantic import BaseModel


class TrendInfo(BaseModel):
    direction: str  # "ASCENDING", "DESCENDING", "STABLE"
    slope: float
    confidence: float


class TrendAnalysis(BaseModel):
    heartRate: TrendInfo
    oxygen: TrendInfo
    activity: TrendInfo
```

---

### 16. `app/schemas/pattern.py`

```python
from pydantic import BaseModel


class PatternInfo(BaseModel):
    type: str
    description: str
    confidence: float


class PatternDetection(BaseModel):
    patternsFound: bool
    patterns: list[PatternInfo]
```

---

### 17. `app/schemas/anomaly.py`

```python
from pydantic import BaseModel


class AnomalyDetection(BaseModel):
    anomalyDetected: bool
    anomalyScore: float
    affectedReadingsIndexes: list[int]
```

---

### 18. `app/schemas/__init__.py` (actualizado)

```python
from app.schemas.anomaly import AnomalyDetection
from app.schemas.analyze_request import AnalyzeRequest
from app.schemas.pattern import PatternDetection
from app.schemas.risk import RiskPrediction
from app.schemas.trend import TrendAnalysis

__all__ = [
    "AnalyzeRequest",
    "RiskPrediction",
    "TrendAnalysis",
    "PatternDetection",
    "AnomalyDetection",
]
```

---

### 19. `app/services/interfaces/__init__.py`

Archivo vacío.

---

### 20. `app/services/interfaces/risk_predictor.py`

```python
from typing import Protocol

import numpy as np


class RiskPredictor(Protocol):
    def predict(
        self, features: np.ndarray, patient_id: str
    ) -> dict[str, object]: ...
```

---

### 21. `app/services/interfaces/trend_analyzer.py`

```python
from typing import Protocol

import numpy as np


class TrendAnalyzer(Protocol):
    def analyze(self, values: np.ndarray) -> dict[str, object]: ...
```

---

### 22. `app/services/interfaces/pattern_detector.py`

```python
from typing import Protocol

import numpy as np


class PatternDetector(Protocol):
    def detect(self, values: np.ndarray) -> dict[str, object]: ...
```

---

### 23. `app/services/interfaces/anomaly_detector.py`

```python
from typing import Protocol

import numpy as np


class AnomalyDetector(Protocol):
    def detect(self, readings_matrix: np.ndarray) -> dict[str, object]: ...
```

---

### 24. `app/ml/__init__.py`

Archivo vacío.

---

### 25. `app/ml/feature_engineering.py`

```python
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
    """Construye el vector de features para el modelo de riesgo.

    Features (9 total):
    - hr_mean, hr_std, hr_slope
    - o2_mean, o2_std, o2_slope
    - act_mean, act_std, act_slope
    """
    if not (len(heart_rates) == len(oxygens) == len(activities)):
        raise ValueError("All input arrays must have the same length")

    if len(heart_rates) < window:
        raise ValueError(f"Need at least {window} readings, got {len(heart_rates)}")

    features = np.array([
        np.mean(heart_rates[-window:]),
        np.std(heart_rates[-window:]),
        compute_slope(heart_rates[-window:]),
        np.mean(oxygens[-window:]),
        np.std(oxygens[-window:]),
        compute_slope(oxygens[-window:]),
        np.mean(activities[-window:]),
        np.std(activities[-window:]),
        compute_slope(activities[-window:]),
    ])

    return features
```

---

### 26. `app/ml/model_loader.py`

```python
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
```

---

### 27. `app/ml/model_registry/metadata_v1.json`

```json
{
    "version": "risk_model_v1",
    "trained_at": "2026-08-06",
    "model_type": "RandomForestClassifier",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    },
    "features": [
        "hr_mean", "hr_std", "hr_slope",
        "o2_mean", "o2_std", "o2_slope",
        "act_mean", "act_std", "act_slope"
    ],
    "window_size": 12,
    "metrics": {
        "roc_auc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0
    },
    "dataset_hash": "",
    "notes": "Placeholder — replace after first training run"
}
```

---

### 28. `app/utils/__init__.py`

Archivo vacío.

---

### 29. `app/utils/time_windows.py`

```python
from datetime import datetime


def get_recent_window(
    timestamps: list[datetime], window_size: int
) -> list[int]:
    """Retorna índices de las últimas `window_size` lecturas."""
    if len(timestamps) < window_size:
        raise ValueError(
            f"Need at least {window_size} timestamps, got {len(timestamps)}"
        )
    return list(range(len(timestamps) - window_size, len(timestamps)))
```

---

### 30. `app/services/__init__.py`

Archivo vacío.

---

### 31. `app/services/risk_prediction_service.py`

```python
import logging

import numpy as np

from app.core.config import settings
from app.ml.model_loader import get_model_loader

logger = logging.getLogger(__name__)


class RiskPredictionService:
    def predict(self, features: np.ndarray, patient_id: str) -> dict[str, object]:
        loader = get_model_loader()

        if not loader.is_loaded:
            from app.core.exceptions import ModelNotLoadedError

            raise ModelNotLoadedError()

        score = loader.predict_proba(features)

        if score >= settings.RISK_THRESHOLD_HIGH:
            level = "HIGH"
            recommendation = (
                "High risk detected. Consider immediate clinical review."
            )
        elif score >= settings.RISK_THRESHOLD_MEDIUM:
            level = "MEDIUM"
            recommendation = (
                "Moderate risk. Schedule follow-up within 24 hours."
            )
        else:
            level = "LOW"
            recommendation = "Risk within acceptable range."

        logger.info(
            "Risk prediction for patient %s: score=%.3f, level=%s",
            patient_id,
            score,
            level,
        )

        return {
            "riskScore": round(score, 3),
            "riskLevel": level,
            "recommendation": recommendation,
        }
```

---

### 32. `app/services/trend_analysis_service.py`

```python
import numpy as np

from app.ml.feature_engineering import compute_slope


class TrendAnalysisService:
    def analyze(self, values: np.ndarray) -> dict[str, object]:
        if len(values) < 2:
            return {
                "direction": "STABLE",
                "slope": 0.0,
                "confidence": 0.0,
            }

        slope = compute_slope(values)
        normalized_slope = slope / (np.mean(values) + 1e-10)
        confidence = min(abs(normalized_slope) * 10, 1.0)

        if abs(slope) < 0.05:
            direction = "STABLE"
        elif slope > 0:
            direction = "ASCENDING"
        else:
            direction = "DESCENDING"

        return {
            "direction": direction,
            "slope": round(float(slope), 3),
            "confidence": round(float(confidence), 3),
        }
```

---

### 33. `app/services/pattern_detection_service.py`

```python
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
        return cv > 0.3
```

---

### 34. `app/services/anomaly_detection_service.py`

```python
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetectionService:
    def __init__(self) -> None:
        self._model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
        )
        self._is_fitted = False

    def fit(self, training_data: np.ndarray) -> None:
        self._model.fit(training_data)
        self._is_fitted = True

    def detect(self, readings_matrix: np.ndarray) -> dict[str, object]:
        if not self._is_fitted:
            self.fit(readings_matrix)

        predictions = self._model.predict(readings_matrix)
        scores = self._model.decision_function(readings_matrix)

        anomaly_indices = np.where(predictions == -1)[0].tolist()
        anomaly_score = float(1 - np.min(scores)) if len(scores) > 0 else 0.0
        anomaly_score = max(0.0, min(1.0, anomaly_score))

        return {
            "anomalyDetected": len(anomaly_indices) > 0,
            "anomalyScore": round(anomaly_score, 3),
            "affectedReadingsIndexes": anomaly_indices,
        }
```

---

### 35. `app/services/clinical_analysis_orchestrator.py`

```python
import asyncio
import logging

import numpy as np

from app.ml.feature_engineering import build_feature_vector
from app.services.anomaly_detection_service import AnomalyDetectionService
from app.services.interfaces.anomaly_detector import AnomalyDetector
from app.services.interfaces.pattern_detector import PatternDetector
from app.services.interfaces.risk_predictor import RiskPredictor
from app.services.interfaces.trend_analyzer import TrendAnalyzer
from app.services.pattern_detection_service import PatternDetectionService
from app.services.risk_prediction_service import RiskPredictionService
from app.services.trend_analysis_service import TrendAnalysisService

logger = logging.getLogger(__name__)


class ClinicalAnalysisOrchestrator:
    def __init__(
        self,
        risk_predictor: RiskPredictor | None = None,
        trend_analyzer: TrendAnalyzer | None = None,
        pattern_detector: PatternDetector | None = None,
        anomaly_detector: AnomalyDetector | None = None,
    ) -> None:
        self._risk_predictor = risk_predictor or RiskPredictionService()
        self._trend_analyzer = trend_analyzer or TrendAnalysisService()
        self._pattern_detector = pattern_detector or PatternDetectionService()
        self._anomaly_detector = anomaly_detector or AnomalyDetectionService()

    async def analyze(
        self,
        heart_rates: np.ndarray,
        oxygens: np.ndarray,
        activities: np.ndarray,
        patient_id: str,
        window_size: int,
    ) -> dict[str, object]:
        features = build_feature_vector(heart_rates, oxygens, activities, window_size)

        readings_matrix = np.column_stack([heart_rates, oxygens, activities])

        risk_result, hr_trend, o2_trend, act_trend, pattern_result, anomaly_result = (
            await asyncio.gather(
                asyncio.to_thread(
                    self._risk_predictor.predict, features, patient_id
                ),
                asyncio.to_thread(self._trend_analyzer.analyze, heart_rates),
                asyncio.to_thread(self._trend_analyzer.analyze, oxygens),
                asyncio.to_thread(self._trend_analyzer.analyze, activities),
                asyncio.to_thread(self._pattern_detector.detect, heart_rates),
                asyncio.to_thread(
                    self._anomaly_detector.detect, readings_matrix
                ),
                return_exceptions=True,
            )
        )

        results: dict[str, object] = {}

        for name, result in [
            ("risk", risk_result),
            ("hr_trend", hr_trend),
            ("o2_trend", o2_trend),
            ("act_trend", act_trend),
            ("pattern", pattern_result),
            ("anomaly", anomaly_result),
        ]:
            if isinstance(result, Exception):
                logger.error("Capability %s failed: %s", name, result)
            else:
                results[name] = result

        return {
            "riskPrediction": results.get("risk", {
                "riskScore": 0.0,
                "riskLevel": "LOW",
                "recommendation": "Analysis unavailable",
            }),
            "trendAnalysis": {
                "heartRate": results.get("hr_trend", {
                    "direction": "STABLE",
                    "slope": 0.0,
                    "confidence": 0.0,
                }),
                "oxygen": results.get("o2_trend", {
                    "direction": "STABLE",
                    "slope": 0.0,
                    "confidence": 0.0,
                }),
                "activity": results.get("act_trend", {
                    "direction": "STABLE",
                    "slope": 0.0,
                    "confidence": 0.0,
                }),
            },
            "patternDetection": results.get("pattern", {
                "patternsFound": False,
                "patterns": [],
            }),
            "anomalyDetection": results.get("anomaly", {
                "anomalyDetected": False,
                "anomalyScore": 0.0,
                "affectedReadingsIndexes": [],
            }),
        }
```

---

### 36. `app/api/__init__.py`

Archivo vacío.

---

### 37. `app/api/v1/__init__.py`

Archivo vacío.

---

### 38. `app/api/v1/dependencies.py`

```python
from functools import lru_cache

from app.services.clinical_analysis_orchestrator import ClinicalAnalysisOrchestrator


@lru_cache
def get_orchestrator() -> ClinicalAnalysisOrchestrator:
    return ClinicalAnalysisOrchestrator()
```

---

### 39. `app/api/v1/routes/__init__.py`

Archivo vacío.

---

### 40. `app/api/v1/routes/analyze.py`

```python
import logging

import numpy as np
from fastapi import APIRouter, Depends

from app.core.exceptions import InsufficientReadingsError
from app.core.security import verify_api_key
from app.ml.model_loader import get_model_loader
from app.schemas.analyze_request import AnalyzeRequest
from app.api.v1.dependencies import get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze")
async def analyze(
    request: AnalyzeRequest,
    _api_key: str = Depends(verify_api_key),
    orchestrator=Depends(get_orchestrator),
) -> dict:
    loader = get_model_loader()
    window_size = request.windowSize or 12

    if len(request.readings) < window_size:
        raise InsufficientReadingsError(
            required=window_size, actual=len(request.readings)
        )

    heart_rates = np.array([r.heartRate for r in request.readings])
    oxygens = np.array([r.oxygen for r in request.readings])
    activities = np.array([r.activity for r in request.readings])

    result = await orchestrator.analyze(
        heart_rates=heart_rates,
        oxygens=oxygens,
        activities=activities,
        patient_id=request.patientId,
        window_size=window_size,
    )

    return {
        "patientId": request.patientId,
        "modelVersion": loader.version,
        **result,
    }
```

---

### 41. `app/api/v1/routes/health.py`

```python
from fastapi import APIRouter

from app.ml.model_loader import get_model_loader

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness_check() -> dict:
    loader = get_model_loader()
    if not loader.is_loaded:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "Model not loaded"},
        )
    return {"status": "ready", "modelVersion": loader.version}
```

---

### 42. `app/api/v1/routes/model_info.py`

```python
from fastapi import APIRouter, Depends

from app.core.security import verify_api_key
from app.ml.model_loader import get_model_loader

router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model-info")
async def model_info(
    _api_key: str = Depends(verify_api_key),
) -> dict:
    loader = get_model_loader()
    return {
        "version": loader.version,
        "isLoaded": loader.is_loaded,
        "metadata": loader.metadata,
    }
```

---

### 43. `app/main.py`

```python
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.core.config import settings
from app.core.exceptions import EXCEPTION_HANDLERS
from app.core.logging import setup_logging
from app.ml.model_loader import get_model_loader


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    logger = logging.getLogger(__name__)

    loader = get_model_loader()
    try:
        loader.load(settings.MODEL_VERSION)
        logger.info("Model loaded successfully: %s", settings.MODEL_VERSION)
    except FileNotFoundError:
        logger.warning(
            "Model file not found for version %s. Service will start without model.",
            settings.MODEL_VERSION,
        )

    yield

    logger.info("Shutting down ML service")


app = FastAPI(
    title="Dialitech ML Service",
    description="Microservicio de análisis clínico ML para pacientes en diálisis",
    version="1.0.0",
    lifespan=lifespan,
)

for exc_type, handler in EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exc_type, handler)

from app.api.v1.routes.analyze import router as analyze_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.model_info import router as model_info_router

app.include_router(analyze_router)
app.include_router(health_router)
app.include_router(model_info_router)
```

---

### 44. `training/__init__.py`

Archivo vacío (o no crearlo si se prefiere que training no sea paquete).

---

### 45. `training/generate_dataset.py`

```python
"""Generación de dataset sintético para entrenamiento del modelo de riesgo."""

import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    n_patients: int = 50,
    readings_per_patient: int = 100,
    window_size: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for patient_id in range(n_patients):
        base_hr = rng.uniform(60, 100)
        base_o2 = rng.uniform(94, 99)
        base_activity = rng.uniform(20, 80)

        risk_label = rng.choice([0, 1], p=[0.7, 0.3])

        hr_trend = rng.uniform(-2, 2) if risk_label == 0 else rng.uniform(1, 5)
        o2_trend = rng.uniform(-0.5, 0.5) if risk_label == 0 else rng.uniform(-2, -0.5)
        act_trend = rng.uniform(-1, 1) if risk_label == 0 else rng.uniform(-5, -1)

        for i in range(readings_per_patient):
            hr = base_hr + hr_trend * i / readings_per_patient + rng.normal(0, 2)
            o2 = base_o2 + o2_trend * i / readings_per_patient + rng.normal(0, 0.5)
            act = max(
                0,
                base_activity
                + act_trend * i / readings_per_patient
                + rng.normal(0, 5),
            )
            rows.append(
                {
                    "heartRate": round(hr, 1),
                    "oxygen": round(max(80, min(100, o2)), 1),
                    "activity": round(act, 1),
                    "label": risk_label,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    df.to_csv("dataset_synthetic.csv", index=False)
    print(f"Generated {len(df)} rows")
```

---

### 46. `training/train_risk_model.py`

```python
"""Entrenamiento del modelo de predicción de riesgo (Random Forest)."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.ml.feature_engineering import build_feature_vector


def train(
    dataset_path: str = "dataset_synthetic.csv",
    window_size: int = 12,
    output_dir: str = "app/ml/model_registry",
) -> None:
    df = pd.read_csv(dataset_path)
    rng = np.random.default_rng(42)

    features_list = []
    labels = []

    for patient_id in range(0, len(df), 100):
        chunk = df.iloc[patient_id : patient_id + 100]
        if len(chunk) < window_size:
            continue

        hr = chunk["heartRate"].values
        o2 = chunk["oxygen"].values
        act = chunk["activity"].values
        label = chunk["label"].iloc[0]

        feat = build_feature_vector(hr, o2, act, window_size)
        features_list.append(feat)
        labels.append(label)

    X = np.array(features_list)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"ROC AUC:  {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / "risk_model_v1.joblib")

    metadata = {
        "version": "risk_model_v1",
        "trained_at": pd.Timestamp.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
        },
        "features": [
            "hr_mean", "hr_std", "hr_slope",
            "o2_mean", "o2_std", "o2_slope",
            "act_mean", "act_std", "act_slope",
        ],
        "window_size": window_size,
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
        },
        "dataset_hash": "",
        "notes": "Trained on synthetic dataset",
    }

    with open(out / "metadata_v1.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {out / 'risk_model_v1.joblib'}")
    print(f"Metadata saved to {out / 'metadata_v1.json'}")


if __name__ == "__main__":
    train()
```

---

### 47. `training/evaluate_model.py`

```python
"""Evaluación del modelo entrenado."""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from app.ml.feature_engineering import build_feature_vector


def evaluate(
    model_path: str = "app/ml/model_registry/risk_model_v1.joblib",
    dataset_path: str = "dataset_synthetic.csv",
    window_size: int = 12,
) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(dataset_path)

    features_list = []
    labels = []

    for patient_id in range(0, len(df), 100):
        chunk = df.iloc[patient_id : patient_id + 100]
        if len(chunk) < window_size:
            continue

        hr = chunk["heartRate"].values
        o2 = chunk["oxygen"].values
        act = chunk["activity"].values
        label = chunk["label"].iloc[0]

        feat = build_feature_vector(hr, o2, act, window_size)
        features_list.append(feat)
        labels.append(label)

    X = np.array(features_list)
    y = np.array(labels)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("Evaluation Results:")
    print(f"ROC AUC: {roc_auc_score(y, y_proba):.4f}")
    print()
    print(classification_report(y, y_pred))


if __name__ == "__main__":
    evaluate()
```

---

### 48. `tests/__init__.py`

Archivo vacío.

---

### 49. `tests/unit/__init__.py`

Archivo vacío.

---

### 50. `tests/conftest.py`

```python
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
```

---

### 51. `tests/unit/test_feature_engineering.py`

```python
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
```

---

### 52. `tests/unit/test_trend_analysis_service.py`

```python
import numpy as np

from app.services.trend_analysis_service import TrendAnalysisService


class TestTrendAnalysisService:
    def setup_method(self) -> None:
        self.service = TrendAnalysisService()

    def test_ascending_trend(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.service.analyze(values)
        assert result["direction"] == "ASCENDING"
        assert result["slope"] > 0

    def test_descending_trend(self) -> None:
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = self.service.analyze(values)
        assert result["direction"] == "DESCENDING"
        assert result["slope"] < 0

    def test_stable_trend(self) -> None:
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = self.service.analyze(values)
        assert result["direction"] == "STABLE"

    def test_single_value(self) -> None:
        result = self.service.analyze(np.array([5.0]))
        assert result["direction"] == "STABLE"
```

---

### 53. `tests/unit/test_pattern_detection_service.py`

```python
import numpy as np

from app.services.pattern_detection_service import PatternDetectionService


class TestPatternDetectionService:
    def setup_method(self) -> None:
        self.service = PatternDetectionService()

    def test_oscillation_detected(self) -> None:
        values = np.array([10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0])
        result = self.service.detect(values)
        assert result["patternsFound"] is True

    def test_stable_no_pattern(self) -> None:
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = self.service.detect(values)
        assert result["patternsFound"] is False

    def test_too_few_values(self) -> None:
        result = self.service.detect(np.array([1.0, 2.0]))
        assert result["patternsFound"] is False
```

---

### 54. `tests/unit/test_anomaly_detection_service.py`

```python
import numpy as np

from app.services.anomaly_detection_service import AnomalyDetectionService


class TestAnomalyDetectionService:
    def setup_method(self) -> None:
        self.service = AnomalyDetectionService()

    def test_fit_and_detect(self) -> None:
        rng = np.random.default_rng(42)
        training_data = rng.uniform(0, 100, (100, 3))
        self.service.fit(training_data)

        normal = np.array([[70.0, 97.0, 40.0]])
        result = self.service.detect(normal)
        assert "anomalyDetected" in result
        assert "anomalyScore" in result
        assert "affectedReadingsIndexes" in result

    def test_unfitted_model_fits_on_first_call(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, (20, 3))
        result = self.service.detect(data)
        assert isinstance(result["anomalyDetected"], bool)
```

---

### 55. `tests/unit/test_risk_prediction_service.py`

```python
import numpy as np

from app.services.risk_prediction_service import RiskPredictionService


class TestRiskPredictionService:
    def setup_method(self) -> None:
        self.service = RiskPredictionService()

    def test_predict_returns_expected_keys(self) -> None:
        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert "riskScore" in result
        assert "riskLevel" in result
        assert "recommendation" in result
        assert result["riskLevel"] in ("LOW", "MEDIUM", "HIGH")
```

---

### 56. `tests/unit/test_clinical_analysis_orchestrator.py`

```python
import numpy as np

from app.services.clinical_analysis_orchestrator import ClinicalAnalysisOrchestrator
import pytest


class TestClinicalAnalysisOrchestrator:
    def setup_method(self) -> None:
        self.orchestrator = ClinicalAnalysisOrchestrator()

    @pytest.mark.asyncio
    async def test_analyze_returns_all_sections(self) -> None:
        rng = np.random.default_rng(42)
        hr = rng.uniform(60, 100, 12)
        o2 = rng.uniform(94, 99, 12)
        act = rng.uniform(20, 80, 12)

        result = await self.orchestrator.analyze(
            heart_rates=hr,
            oxygens=o2,
            activities=act,
            patient_id="test-patient",
            window_size=12,
        )

        assert "riskPrediction" in result
        assert "trendAnalysis" in result
        assert "patternDetection" in result
        assert "anomalyDetection" in result
```

---

### 57. `tests/integration/__init__.py`

Archivo vacío.

---

### 58. `tests/integration/test_analyze_endpoint.py`

```python
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestAnalyzeEndpoint:
    def test_valid_request(self, client: TestClient) -> None:
        rng = np.random.default_rng(42)
        readings = []
        for _ in range(15):
            readings.append({
                "heartRate": round(float(rng.uniform(60, 100)), 1),
                "oxygen": round(float(rng.uniform(94, 99)), 1),
                "activity": round(float(rng.uniform(20, 80)), 1),
                "timestamp": "2026-08-06T10:00:00Z",
            })

        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "integration-test",
                "windowSize": 12,
                "readings": readings,
            },
            headers={"X-API-Key": "REPLACED_API_KEY"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["patientId"] == "integration-test"
        assert "riskPrediction" in data
        assert "trendAnalysis" in data
        assert "patternDetection" in data
        assert "anomalyDetection" in data

    def test_insufficient_readings(self, client: TestClient) -> None:
        readings = [
            {
                "heartRate": 70.0,
                "oxygen": 97.0,
                "activity": 40.0,
                "timestamp": "2026-08-06T10:00:00Z",
            }
        ]

        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": readings,
            },
            headers={"X-API-Key": "REPLACED_API_KEY"},
        )

        assert response.status_code == 400

    def test_missing_api_key(self, client: TestClient) -> None:
        rng = np.random.default_rng(42)
        readings = [
            {
                "heartRate": round(float(rng.uniform(60, 100)), 1),
                "oxygen": round(float(rng.uniform(94, 99)), 1),
                "activity": round(float(rng.uniform(20, 80)), 1),
                "timestamp": "2026-08-06T10:00:00Z",
            }
            for _ in range(15)
        ]

        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": readings,
            },
        )

        assert response.status_code == 422
```

---

### 59. `tests/integration/test_health_endpoint.py`

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_liveness(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code in (200, 503)
```

---

### 60. `Dockerfile`

```dockerfile
FROM python:3.11-slim AS base

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 61. `docker-compose.yml`

```yaml
services:
  ml-service:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

### 62. `render.yaml`

```yaml
services:
  - type: web
    name: dialitech-ml-service
    runtime: docker
    plan: free
    healthCheckPath: /health
    envVars:
      - key: API_KEY
        sync: false
      - key: MODEL_VERSION
        value: risk_model_v1
      - key: LOG_LEVEL
        value: INFO
      - key: RISK_THRESHOLD_HIGH
        value: "0.7"
      - key: RISK_THRESHOLD_MEDIUM
        value: "0.4"
```

---

### 63. `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Format check
        run: black --check .

      - name: Import order check
        run: isort --check-only .

      - name: Lint
        run: ruff check .

      - name: Type check
        run: mypy app/

      - name: Security scan
        run: bandit -r app/

      - name: Dependency audit
        run: pip-audit

      - name: Tests with coverage
        run: pytest --cov=app --cov-report=xml --cov-report=term-missing

      - name: Build Docker image
        run: docker build -t dialitech-ml-service .
```

---

### 64. `tests/fixtures/test_model.joblib`

> **Nota**: Este archivo se genera ejecutando `training/train_risk_model.py`
> después de generar el dataset con `training/generate_dataset.py`. Opcionalmente,
> se puede crear un modelo pequeño y determinista específico para tests que se
> versiona en el repo. Por ahora, los tests unitarios no necesitan este archivo
> (mockean las dependencias), y los tests de integración pueden fallar gracefully
> si el modelo no está cargado.

---

## Instrucciones Finales

1. **Crea todos los archivos** en el orden presentado.
2. **Todos los `__init__.py`** deben existir aunque estén vacíos.
3. **No agregues comentarios** en el código a menos que se pida explícitamente.
4. **Sigue las convenciones de Python**: type hints en todo, snake_case para
   funciones/variables, PascalCase para clases.
5. **Los tests unitarios no deben necesitar el modelo real** — usan las Protocol
   para hacer mock de las dependencias.
6. **Los tests de integración** usan `TestClient` de FastAPI.
7. **Después de implementar**, ejecuta:
   ```bash
   black --check .
   isort --check-only .
   ruff check .
   mypy app/
   pytest --cov=app
   ```
8. Si algún test o linter falla, corrige antes de declarar完成.
