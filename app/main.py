import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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
