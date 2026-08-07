from pydantic import BaseModel


class AnomalyDetection(BaseModel):
    anomalyDetected: bool
    anomalyScore: float
    affectedReadingsIndexes: list[int]
