import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.core.config import get_settings
from .api.routes import router


# Force UTF-8 on the log stream so emoji/Unicode in log messages (🤖, ✅, →,
# Arabic snippet text, etc.) don't raise UnicodeEncodeError under the Windows
# console's legacy cp1252 codec. reconfigure() exists on Py3.7+ TextIO streams;
# guard it so a non-standard stdout (pytest capture, etc.) can't break startup.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

logging.basicConfig(
    level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("🤖 Recruitment AI Agent System — Starting Up")

    

    
    logger.info("  LLM model:    %s", settings.groq_model)
    
    
    if not settings.tavily_api_key:
        logger.error("TAVILY_API_KEY is not set!")
        sys.exit(1)
        
    if not settings.groq_api_key:
        logger.error("GROQ_API_KEY is not set!")
        sys.exit(1)

    logger.info("✅ Configuration validated. Agent ready.")
    yield
    logger.info("🛑 Shutting Down")





    # logger.info("  LLM model:    %s", settings.ollama_model)
    # import requests
    # try:
    #     requests.get(settings.ollama_base_url, timeout=2)
    #     logger.info("✅ Ollama is reachable.")
    # except Exception:
    #     logger.warning(f"⚠️ Cannot connect to Ollama at {settings.ollama_base_url}. Make sure it's running!")

    # if not settings.tavily_api_key:
    #     logger.error("TAVILY_API_KEY is not set!")
    #     sys.exit(1)

    # logger.info("✅ Configuration validated. Agent ready.")
    # yield
    # logger.info("🛑 Shutting Down")




app = FastAPI(title="Recruitment AI Agent API", lifespan=lifespan)

# The SPA is served same-origin from the StaticFiles mount below, so in the
# default deployment CORS never engages at all. These origins exist for split
# dev setups (e.g. a Vite dev server on another port). allow_origins must stay
# an explicit list rather than "*": the wildcard is invalid alongside
# allow_credentials and browsers reject such responses outright.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router)

if Path("static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})