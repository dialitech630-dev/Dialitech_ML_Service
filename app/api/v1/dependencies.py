from functools import lru_cache

from app.services.clinical_analysis_orchestrator import ClinicalAnalysisOrchestrator


@lru_cache
def get_orchestrator() -> ClinicalAnalysisOrchestrator:
    return ClinicalAnalysisOrchestrator()
