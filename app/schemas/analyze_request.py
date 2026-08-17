from pydantic import BaseModel, Field

from app.schemas.reading import HealthReading


class AnalyzeRequest(BaseModel):
    patientId: str = Field(..., min_length=1, max_length=128)
    windowSize: int = Field(default=12, ge=1, le=1000)
    readings: list[HealthReading] = Field(..., min_length=1, max_length=10000)
