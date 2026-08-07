from pydantic import BaseModel


class TrendInfo(BaseModel):
    direction: str
    slope: float
    confidence: float


class TrendAnalysis(BaseModel):
    heartRate: TrendInfo
    oxygen: TrendInfo
    activity: TrendInfo
