"""
backend/agents/routes.py
========================
FastAPI route handlers for the Recruitment AI Agent.

All heavy agent logic lives in recruitment_agent.py.
This file is HTTP glue only: request validation, file handling, and response shaping.

Key improvements vs original
-----------------------------
  - No circular import: this file imports FROM recruitment_agent, never the reverse.
  - graph.invoke() offloaded to asyncio.to_thread() so the async event loop is
    never blocked by a slow Ollama call.
  - asyncio.wait_for() wraps every agent call with a configurable timeout.
  - CV upload enforces a 10 MB hard limit BEFORE reading the full file into memory,
    eliminating the OOM risk from unbounded uploads.
  - PyPDF2 (deprecated) replaced by backend.parsers.cv_parser.parse_cv_bytes,
    which uses PyMuPDF → pdfminer fallback and handles multi-column layouts,
    tables, and Unicode correctly.
  - detected_title from the parser is forwarded to run_cv_analysis so the
    title-hint feature actually works.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.agents.recruitment_agent import run_cv_analysis, run_targeted_search
from backend.parsers.cv_parser import parse_cv_bytes

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Hard upper bound on CV file size.  Enforced before any buffering so a large
# upload cannot exhaust worker memory before we have a chance to reject it.
MAX_CV_BYTES: int = 10 * 1024 * 1024   # 10 MB

# Maximum wall-clock seconds to wait for the agent graph to complete.
# Tune this to match your Ollama model's typical response latency.
# A 5-minute window comfortably covers a 5-iteration search on most hardware.
AGENT_TIMEOUT_SECONDS: int = 300


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    job_title: str = Field(..., min_length=2, max_length=200)
    location:  str = Field(..., min_length=2, max_length=200)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/api/targeted-search")
async def targeted_search_endpoint(request: SearchRequest):
    """
    Search for real job listings matching a job title + location.

    The agent graph (graph.invoke) is synchronous and CPU/IO-bound.
    We run it in a thread pool via asyncio.to_thread() so it never blocks
    the FastAPI event loop, and wrap it with asyncio.wait_for() so a hung
    Ollama call cannot starve other in-flight requests indefinitely.
    """
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_targeted_search,
                request.job_title,
                request.location,
            ),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Targeted search timed out after %ds for job_title=%r location=%r",
            AGENT_TIMEOUT_SECONDS,
            request.job_title,
            request.location,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"The search agent timed out after {AGENT_TIMEOUT_SECONDS} seconds. "
                "Please try again."
            ),
        )
    except Exception:
        logger.exception("Unexpected error in targeted_search_endpoint")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while searching for jobs.",
        )

    return result


@router.post("/api/analyze-cv")
async def analyze_cv_endpoint(
    cv: UploadFile = File(...),
    preferred_location: str = Form(""),
):
    """
    Accept a CV file (PDF / DOCX / TXT), extract text via cv_parser, and
    invoke the recruitment agent to find matching jobs.

    Processing pipeline
    -------------------
    1. File-size guard  — read up to MAX_CV_BYTES + 1 bytes; reject if over limit.
    2. Parse            — cv_parser.parse_cv_bytes() handles PDF (PyMuPDF / pdfminer
                          fallback), DOCX (python-docx), and plain text.
    3. Sanity check     — reject obviously empty / image-only CVs early.
    4. Agent call       — run_cv_analysis() in a thread, wrapped with a timeout.
    5. Shape response   — attach lightweight profile metadata for the frontend.
    """

    # ── 1. File-size guard ─────────────────────────────────────────────────
    #
    # UploadFile.read() streams the file, so we request MAX_CV_BYTES + 1 bytes.
    # If we get more than MAX_CV_BYTES back the file is too large and we reject
    # it immediately — before the rest of the bytes are even read from the
    # network socket.  This prevents OOM on huge uploads.
    file_bytes = await cv.read(MAX_CV_BYTES + 1)
    if len(file_bytes) > MAX_CV_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"CV file must be ≤ {MAX_CV_BYTES // (1024 * 1024)} MB. "
                "Please compress or trim your CV and try again."
            ),
        )

    # ── 2. Parse the CV ────────────────────────────────────────────────────
    #
    # parse_cv_bytes() selects the right extractor based on file extension:
    #   .pdf  → PyMuPDF (fitz) with pdfminer fallback
    #   .docx → python-docx (also reads table cells)
    #   .txt  → UTF-8 / latin-1 decode
    # It also normalises whitespace, truncates to settings.max_cv_chars, and
    # infers the candidate's likely job title from the top of the document.
    try:
        profile = parse_cv_bytes(file_bytes, cv.filename or "upload")
    except ValueError as exc:
        # Unsupported file extension or similar user error.
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # Required parsing library (PyMuPDF / pdfminer / python-docx) missing.
        logger.exception("CV parser runtime error for file %r", cv.filename)
        raise HTTPException(status_code=500, detail=str(exc))

    # ── 3. Sanity check ────────────────────────────────────────────────────
    #
    # is_empty() returns True when fewer than 50 characters of text were
    # extracted — typically an image-only PDF or a completely blank file.
    if profile.is_empty():
        raise HTTPException(
            status_code=422,
            detail=(
                profile.parse_error
                or (
                    "The CV appears to be empty or image-only. "
                    "Please upload a text-based PDF, DOCX, or TXT file."
                )
            ),
        )

    # ── 4. Run the agent ───────────────────────────────────────────────────
    #
    # profile.raw_text  — cleaned, truncated CV text
    # profile.detected_title — e.g. "Software Engineer" inferred from CV header
    #   (was ignored in the original routes.py, causing the title-hint feature
    #   to never fire; now forwarded correctly)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_cv_analysis,
                profile.raw_text,
                profile.detected_title,
            ),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "CV analysis timed out after %ds for file %r",
            AGENT_TIMEOUT_SECONDS,
            cv.filename,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"The analysis agent timed out after {AGENT_TIMEOUT_SECONDS} seconds. "
                "Please try again."
            ),
        )
    except Exception:
        logger.exception("Unexpected error in analyze_cv_endpoint")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while analysing the CV.",
        )

    # ── 5. Attach lightweight profile metadata ─────────────────────────────
    #
    # The frontend expects a "profile" key alongside the jobs array.
    # We populate skills from the first matched job (best available proxy when
    # the agent hasn't returned a dedicated skills field).
    result["profile"] = {
        "detected_title":   profile.detected_title,
        "word_count":       profile.word_count,
        "experience_level": "Professional",
        "skills": (
            result["jobs"][0].get("required_skills", [])
            if result.get("jobs")
            else []
        ),
        "summary": (
            "CV analysed successfully. Matching jobs based on extracted skills."
        ),
    }

    if preferred_location:
        result["location"] = preferred_location

    return result