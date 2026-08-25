from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
    )

    anthropic_api_key: str
    voyage_api_key: str
    max_tokens: int = 1024

settings = Settings()