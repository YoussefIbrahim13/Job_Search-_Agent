import logging
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="qwen/qwen3-32b")
    tavily_api_key: str = Field(default="")
    tavily_max_results: int = Field(default=5)
    max_agent_iterations: int = Field(default=2)
    temp_upload_dir: str = Field(default="temp_cvs")
    max_cv_chars: int = Field(default=2000)
    cors_origins: str = Field(default="*")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()