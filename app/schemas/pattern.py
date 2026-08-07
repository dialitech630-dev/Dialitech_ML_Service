from pydantic import BaseModel


class PatternInfo(BaseModel):
    type: str
    description: str
    confidence: float


class PatternDetection(BaseModel):
    patternsFound: bool
    patterns: list[PatternInfo]
