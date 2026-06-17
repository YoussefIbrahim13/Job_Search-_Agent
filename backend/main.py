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


logging.basicConfig(
    level=logging.INFO,
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

app = FastAPI(title="Recruitment AI Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if Path("static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})