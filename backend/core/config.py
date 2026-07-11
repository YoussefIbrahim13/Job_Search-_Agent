import logging
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

            

    # ollama_base_url: str = Field(default="http://localhost:11434")
    # ollama_model: str = Field(default="llama3.1:8b")
   
   
   
   
   
    tavily_api_key: str = Field(default="")
    tavily_max_results: int = Field(default=5)
    # Relative crawl-date window handed to Tavily's `time_range` param.
    # One of: "day" | "week" | "month" | "year". Belt-and-suspenders with the
    # absolute `start_date` filter (RECENCY_WINDOW_DAYS in tools.py). Default
    # "month" == "past month"; set to "week" for stricter "hiring now" recency.
    tavily_time_range: str = Field(default="month")
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