from pydantic import BaseModel, Field

from app.schemas.reading import HealthReading


class AnalyzeRequest(BaseModel):
    patientId: str
    windowSize: int = Field(default=12, ge=1)
    readings: list[HealthReading]
