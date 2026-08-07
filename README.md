# Dialitech ML Service

Microservicio independiente de análisis clínico con Machine Learning para pacientes en diálisis. Propone predicción de riesgo, análisis de tendencias, detección de patrones y detección de anomalías.

## Arquitectura

```
┌──────────────┐      ┌────────────────────┐      ┌──────────────────┐
│   Wearable   │─────▶│  API .NET (Render) │─────▶│  ML Service      │
│  (firmware)  │      │  /api/v1/health/*  │      │  /api/v1/analyze │
└──────────────┘      └────────────────────┘      └──────────────────┘
                            │    │                        │
                            │    └── Alertas              │
                            │         (MongoDB)           │
                            └── Lecturas ─────────────────┘
                                (MongoDB)
```

- **Stateless**: nunca escribe directamente en la base de datos.
- **Modular**: `app/` para inferencia, `training/` para entrenamiento (nunca se importa en producción).
- **Feature sharing**: `feature_engineering.py` se usa tanto en entrenamiento como en inferencia.

## Qué analiza

| Capacidad | Descripción |
|-----------|-------------|
| **Risk Prediction** | Score 0-1 de riesgo clínico (LOW / MEDIUM / HIGH) usando RandomForest |
| **Trend Analysis** | Dirección (ASCENDING / DESCENDING / STABLE), pendiente y confianza para heart rate, oxygen y activity |
| **Pattern Detection** | Oscilaciones periódicas, alta variabilidad, tendencia monotónica |
| **Anomaly Detection** | Z-score por métrica, detección de outliers |

## Endpoints

Todos los endpoints protegidos requieren header `X-API-Key`.

### `POST /api/v1/analyze`

Análisis clínico completo.

```json
{
  "patientId": "patient_001",
  "windowSize": 12,
  "readings": [
    {
      "heartRate": 75.0,
      "oxygen": 97.0,
      "activity": 45.0,
      "timestamp": "2025-01-15T10:30:00Z"
    }
  ]
}
```

**Response:**

```json
{
  "patientId": "patient_001",
  "modelVersion": "risk_model_v1",
  "riskPrediction": {
    "riskScore": 0.23,
    "riskLevel": "LOW",
    "riskFactors": [],
    "recommendation": "No immediate action required"
  },
  "trendAnalysis": {
    "heartRate": { "direction": "STABLE", "slope": 0.01, "confidence": 0.95 },
    "oxygen": { "direction": "STABLE", "slope": -0.002, "confidence": 0.98 },
    "activity": { "direction": "ASCENDING", "slope": 0.5, "confidence": 0.72 }
  },
  "patternDetection": {
    "patternsFound": false,
    "patterns": [],
    "insights": ["No unusual patterns detected"]
  },
  "anomalyDetection": {
    "anomalyDetected": false,
    "anomalyScore": 0.15,
    "threshold": 2.0,
    "affectedMetrics": [],
    "insights": ["All metrics within normal range"]
  }
}
```

### `GET /health`

Health check sin autenticación. Retorna `{"status": "healthy"}`.

### `GET /health/ready`

Readiness check. Retorna 503 si el modelo no está cargado.

### `GET /api/v1/model-info`

Información del modelo cargado (requiere `X-API-Key`).

## Cómo correrlo

### Requisitos

- Python 3.11+
- pip

### Instalación

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Variables de entorno

Crea un archivo `.env`:

```env
API_KEY=REPLACED_API_KEY
MODEL_VERSION=risk_model_v1
LOG_LEVEL=INFO
WINDOW_SIZE_DEFAULT=12
RISK_THRESHOLD_HIGH=0.7
RISK_THRESHOLD_MEDIUM=0.4
```

### Entrenar el modelo

```bash
# Generar dataset sintético
python training/generate_dataset.py

# Entrenar modelo
PYTHONPATH=. python training/train_risk_model.py

# Evaluar modelo
PYTHONPATH=. python training/evaluate_model.py
```

### Iniciar el servidor

```bash
# Opción 1: script helper
serve.bat          # Windows

# Opción 2: directo
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: `http://localhost:8000/docs`

## Docker

```bash
# Build
docker build -t dialitech-ml-service .

# Run
docker run -p 8000:8000 -e API_KEY=REPLACED_API_KEY dialitech-ml-service
```

## Despliegue en Railway

1. Conecta el repo en Railway
2. Railway detecta el Dockerfile automáticamente
3. Agrega las variables de entorno en el dashboard:
   - `API_KEY` — clave compartida con la API .NET
   - `MODEL_VERSION` — `risk_model_v1`
4. Railway asigna el puerto vía `$PORT` (el Dockerfile lo maneja)

## Estructura del proyecto

```
dialitech-ml-service/
├── app/                          # Código de producción
│   ├── api/v1/routes/            # Endpoints (analyze, health, model_info)
│   ├── core/                     # Config, exceptions, logging, security
│   ├── ml/                       # Model loader, feature engineering, model registry
│   ├── schemas/                  # Pydantic models (request/response)
│   └── services/                 # Business logic (risk, trends, patterns, anomalies, orchestrator)
├── training/                     # Scripts offline (nunca se importa en producción)
│   ├── generate_dataset.py
│   ├── train_risk_model.py
│   └── evaluate_model.py
├── tests/                        # Tests unitarios y de integración
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Tecnologías

- **FastAPI** — framework web
- **scikit-learn** — RandomForestClassifier
- **pydantic** — validación de datos
- **uvicorn** — ASGI server
- **joblib** — serialización del modelo
- **Docker** — contenedorización

## License

Propietario — Dialitech.
