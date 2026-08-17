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
API_KEY=your-secret-api-key-here
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
docker run -p 8000:8000 -e API_KEY=your-secret-api-key-here dialitech-ml-service
```

## Despliegue

El servicio está preparado para desplegarse en **Railway** o **Render**:

### Railway
1. Conecta el repo en Railway
2. Railway detecta el Dockerfile automáticamente
3. Agrega las variables de entorno en el dashboard:
   - `API_KEY` — clave compartida con la API .NET
   - `MODEL_VERSION` — `risk_model_v1`
4. Railway asigna el puerto vía `$PORT` (el Dockerfile lo maneja)

### Render
El archivo `render.yaml` configura el despliegue automático:
- Docker runtime en plan free
- Health check en `/health`
- Variables de entorno predefinidas

## Estructura del proyecto

```
dialitech-ml-service/
├── app/                          # Código de producción
│   ├── api/v1/routes/            # Endpoints (analyze, health, model_info)
│   ├── core/                     # Config, exceptions, logging, security, rate_limit, security_headers
│   ├── ml/                       # Model loader, feature engineering, model registry
│   ├── schemas/                  # Pydantic models (request/response)
│   ├── services/                 # Business logic (risk, trends, patterns, anomalies, orchestrator)
│   └── utils/                    # Utilidades (time_windows)
├── training/                     # Scripts offline (nunca se importa en producción)
│   ├── generate_dataset.py
│   ├── train_risk_model.py
│   └── evaluate_model.py
├── tests/                        # Tests unitarios y de integración
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── .github/workflows/ci.yml      # Pipeline CI (lint, type-check, SAST, tests, coverage)
├── .github/CODEOWNERS            # Routing de reviewers para archivos críticos
├── .github/dependabot.yml        # Actualizaciones automáticas de dependencias
├── .pre-commit-config.yaml       # Hooks locales (lint, format, bandit)
├── .gitleaksignore               # Falsos positivos de Gitleaks
├── .semgrep.yml                  # Reglas SAST (Semgrep)
├── Dockerfile
├── docker-compose.yml
├── render.yaml                   # Configuración Render
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Tecnologías

- **FastAPI** — framework web
- **scikit-learn** — RandomForestClassifier
- **pydantic** — validación de datos
- **uvicorn** — ASGI server
- **joblib** — serialización del modelo
- **Docker** — contenedorización

## Tests

```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Ejecutar todos los tests
pytest

# Solo tests unitarios
pytest tests/unit/

# Solo tests de integración
pytest tests/integration/

# Con cobertura
pytest --cov=app --cov-report=term-missing
```

Cobertura actual: **95%**

## Seguridad

- **Autenticación**: API Key via header `X-API-Key` (comparación constant-time con `hmac.compare_digest`). La clave **no tiene valor por defecto** y debe inyectarse vía entorno (`API_KEY` obligatoria).
- **Rate limiting**: 60 req/min por IP (middleware en memoria)
- **Security headers**: X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, Permissions-Policy, Referrer-Policy
- **CORS**: Restringido a origen específico (`https://dialitech.netlify.app`)
- **Validación de entrada**: Límites en `windowSize` (1-1000) y `readings` (1-10000)
- **Contenedor**: Usuario no-root (UID 1000)

### DevSecOps (CI en GitHub Actions)

| Herramienta | Función | Reporte |
|-------------|---------|---------|
| Gitleaks | Detección de secretos (full history) | CI step |
| pre-commit | Lint, formato, imports, bandit (local = CI) | CI step |
| mypy (strict) | Type checking (config en `pyproject.toml`) | CI step |
| Bandit | SAST de código Python | CI step |
| Semgrep | SAST + reglas custom (`.semgrep.yml` + `p/ci`) | CI step |
| pip-audit | Auditoría de vulnerabilidades en dependencias | CI step |
| Trivy | Escaneo de vulnerabilidades de la imagen Docker | Security tab (SARIF) |
| anchore/sbom-action | Generación de SBOM | Artifact (30 días) |
| Dependabot | Actualizaciones automáticas de dependencias | PR automático |
| pytest + pytest-cov | Tests unitarios + integración, umbral 70% | CI step |

**Branch protection** (configurada en GitHub Settings → Branches):
- PR requerido para merge a `main`
- 1 approval mínimo
- Status check `build` requerido
- Stale approvals deshabilitados
- Force push bloqueado

> **Nota sobre numpy**: CI y Docker usan Python 3.11 (`numpy<2.3`). En entornos locales con Python 3.14 se instala `numpy>=2.5` automáticamente vía markers de entorno.

## License

Propietario — Dialitech.
