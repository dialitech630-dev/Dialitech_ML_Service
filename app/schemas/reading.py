from datetime import datetime

from pydantic import BaseModel


class HealthReading(BaseModel):
    heartRate: float
    oxygen: float
    activity: float
    timestamp: datetime
