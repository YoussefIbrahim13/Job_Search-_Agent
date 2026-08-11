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

    # Results requested per query. This is the top of the funnel, and the funnel
    # is narrow: a real ".net Cairo" search returned 10 candidates, of which 5
    # were category pages and 5 were closed postings — zero survivors, purely
    # for lack of raw material.
    #
    # Tavily bills per SEARCH, not per result (advanced depth = 2 credits for up
    # to 20 results), so widening this is essentially free. The binding cost is
    # the character budget in tools.MAX_RESULT_CHARS, which decides how many of
    # these actually reach the model.
    tavily_max_results: int = Field(default=15)
    # Relative crawl-date window handed to Tavily's `time_range` param.
    # One of: "day" | "week" | "month" | "year". Belt-and-suspenders with the
    # absolute `start_date` filter (RECENCY_WINDOW_DAYS in tools.py). Default
    # "month" == "past month"; set to "week" for stricter "hiring now" recency.
    tavily_time_range: str = Field(default="month")
    # Must exceed the number of planned queries (2: full-time + internship) with
    # headroom, or a single transient Groq 429 consumes the entire budget and
    # the run falls back to ToolMessage recovery. It also gates the
    # coerce_internship branch, which requires iterations < max - 1.
    max_agent_iterations: int = Field(default=4)
    # NOTE: a `temp_upload_dir` setting used to live here, defaulting to
    # "temp_cvs". Nothing ever read it — uploads are parsed entirely in memory
    # and never touch disk. It was removed rather than left in place, because a
    # config knob that looks like it controls where CVs are written, while the
    # UI promises CVs are never stored, is a footgun in both directions.
    max_cv_chars: int = Field(default=2000)

    # Comma-separated list of allowed browser origins, e.g.
    # "http://localhost:8888,https://recruitbot.example.com".
    # Deliberately NOT "*": the API is served same-origin with the SPA, and a
    # wildcard is invalid in combination with allow_credentials (browsers reject
    # the response outright), which breaks the cookie auth this is a prereq for.
    cors_origins: str = Field(default="http://localhost:8888,http://127.0.0.1:8888")

    # NOTE: DEBUG emits CV-derived content (detected title, skills, filename)
    # and full model output. Keep at INFO or above in any shared environment.
    log_level: str = Field(default="INFO")

    @property
    def cors_origin_list(self) -> list[str]:
        """`cors_origins` split into the list CORSMiddleware expects."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()