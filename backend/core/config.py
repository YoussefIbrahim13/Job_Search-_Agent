import logging
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

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

    # ── Phase 2: structured job sources ─────────────────────────────────────
    #
    # JSearch (RapidAPI) is the primary structured provider. It aggregates
    # Google for Jobs, which is the only realistic route to Egypt/MENA coverage
    # with real posted dates and numeric salaries — the two fields the Tavily
    # snippet path could never produce.
    #
    # Empty key is not an error: the adapter reports itself unsupported and the
    # registry falls through to the free providers. The app must keep working
    # for someone who has not signed up for RapidAPI.
    jsearch_api_key: str = Field(default="")
    jsearch_api_host: str = Field(default="jsearch.p.rapidapi.com")

    # Free tier is low hundreds of requests per MONTH, so a per-day ceiling is
    # what stops one enthusiastic afternoon from consuming the month. Enforced
    # by the registry's quota accounting (2.4), before the provider call.
    jsearch_daily_quota: int = Field(default=20)

    # Seconds to wait on a provider HTTP call. Deliberately short: the registry
    # fans out across providers concurrently, so a slow provider should drop
    # out and let the others answer rather than holding the whole search.
    source_http_timeout: float = Field(default=10.0)

    # Total seconds one provider may take, across however many HTTP calls it
    # makes. Distinct from source_http_timeout because an adapter may issue
    # several requests (Arbeitnow walks pages), so a per-request timeout does
    # not bound a provider's total contribution to search latency.
    source_provider_timeout: float = Field(default=25.0)

    # Consecutive failures before a provider's circuit breaker opens, and how
    # long it stays open. The point is to stop paying latency for a provider
    # that is currently broken — every search would otherwise wait the full
    # timeout to rediscover the same outage.
    source_breaker_threshold: int = Field(default=3)
    source_breaker_cooldown: float = Field(default=300.0)

    # Cooldown after a quota refusal. Much longer than the failure cooldown:
    # a 429 against a monthly cap will still be a 429 in five minutes, and
    # retrying only burns what budget remains.
    source_quota_cooldown: float = Field(default=3600.0)

    # ── Phase 2.5: pipeline selection ───────────────────────────────────────
    #
    # False routes searches through the LangGraph agent (the prototype path).
    # True routes them through the structured pipeline: registry fan-out ->
    # dedup -> deterministic score -> semantic pass -> threshold.
    #
    # Defaults to False on purpose. The agent is the current known-good
    # baseline; the replacement should be switched on deliberately, compared on
    # real searches, and only then made the default. Both paths return the same
    # response shape, so the frontend cannot tell them apart except via the
    # `pipeline` field.
    use_structured_pipeline: bool = Field(default=False)

    # Minimum final score (0-100) for a job to be returned.
    #
    # Note the interaction with neutral scoring: a job that states nothing at
    # all lands near 50, because "unknown" is deliberately not "bad". A
    # threshold below that admits every uninformative listing. Tune against the
    # Phase 5 eval set rather than by intuition.
    ranking_score_threshold: float = Field(default=40.0)

    # ── Phase 2.3: LLM semantic ranking pass ────────────────────────────────
    #
    # The last 10 of 100 ranking points. Everything else is a fact comparison
    # done in code; the model is asked only whether two differently-worded
    # roles are the same kind of role.
    #
    # Disabling it costs those 10 points and nothing else — the deterministic
    # scorer stands alone, which is the whole reason for the split.
    semantic_pass_enabled: bool = Field(default=True)

    # How many top-ranked candidates the semantic pass looks at. Cost scales
    # linearly with this, and it is applied after deterministic ranking, so the
    # jobs below the cut are ones already judged poor matches on facts.
    semantic_max_jobs: int = Field(default=15)

    # Description characters sent per job. Enough to judge the role, short
    # enough that fifteen of them fit comfortably in one request.
    semantic_description_chars: int = Field(default=600)

    # NOTE: DEBUG emits CV-derived content (detected title, skills, filename)
    # and full model output. Keep at INFO or above in any shared environment.
    log_level: str = Field(default="INFO")

    @property
    def cors_origin_list(self) -> list[str]:
        """`cors_origins` split into the list CORSMiddleware expects."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()