from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_KEY: str = "REPLACED_API_KEY"
    MODEL_VERSION: str = "risk_model_v1"
    LOG_LEVEL: str = "INFO"
    WINDOW_SIZE_DEFAULT: int = 12
    RISK_THRESHOLD_HIGH: float = 0.7
    RISK_THRESHOLD_MEDIUM: float = 0.4

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
