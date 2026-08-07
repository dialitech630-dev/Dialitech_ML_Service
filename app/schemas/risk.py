from pydantic import BaseModel


class RiskPrediction(BaseModel):
    riskScore: float
    riskLevel: str
    recommendation: str
