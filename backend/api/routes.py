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
  - Server-side magic-bytes validation: the declared filename extension is
    cross-checked against the actual leading bytes of the uploaded buffer
    BEFORE the bytes ever reach parse_cv_bytes. cv.content_type is never
    trusted, since it's client-supplied and trivially spoofed. A file
    extension is necessary but not sufficient — a renamed executable still
    has to lie about its own binary signature to get through.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

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

# Generic message returned for ANY magic-byte mismatch. Deliberately vague —
# it does not tell an attacker which check tripped (wrong signature vs.
# detected executable vs. failed text heuristic), since that distinction is
# only useful to someone probing the validator.
SECURITY_VIOLATION_DETAIL: str = (
    "Security Violation: File content integrity mismatch. Nice try!"
)


# ---------------------------------------------------------------------------
# Server-side magic-bytes / file-signature validation
# ---------------------------------------------------------------------------
#
# Why this exists: the frontend's extension check (and FastAPI's UploadFile
# accept hints) only constrain a well-behaved browser. Nothing stops a raw
# HTTP client from POSTing a file named "resume.pdf" whose actual bytes are
# an executable, a script, or garbage. cv.content_type is just as easy to
# spoof — it's a header the client sets, not something FastAPI verifies
# against the body. The only trustworthy signal is the file's own leading
# bytes, which is what every real "file" utility (libmagic, Windows'
# Explorer type detection, etc.) relies on.
#
# This check runs strictly BEFORE parse_cv_bytes(), so a forged file is
# rejected before it reaches any parsing library — closing off the more
# exotic attack surface of feeding crafted bytes to PyMuPDF/python-docx
# under a false pretext.

# Known dangerous executable/binary signatures. If a buffer's leading bytes
# match any of these, it is rejected outright regardless of what extension
# was declared — there is no legitimate CV format that begins this way.
_EXECUTABLE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"MZ", "Windows PE/EXE/DLL"),
    (b"\x7fELF", "Linux ELF binary"),
    (b"\xca\xfe\xba\xbe", "Mach-O / Java class (fat binary)"),
    (b"\xfe\xed\xfa\xce", "Mach-O binary (32-bit)"),
    (b"\xfe\xed\xfa\xcf", "Mach-O binary (64-bit)"),
    (b"\xcf\xfa\xed\xfe", "Mach-O binary (little-endian)"),
    (b"#!", "Shebang script"),
)

_PDF_SIGNATURE: bytes = b"%PDF"
_ZIP_SIGNATURE: bytes = b"PK\x03\x04"  # .docx is OOXML — a ZIP container

# Extensions that parse_cv_bytes routes to the ZIP-based DOCX extractor.
# Legacy binary .doc (pre-2007, OLE2 Compound File Format) is NOT zip-based
# and uses a different signature entirely (D0 CF 11 E0 ...). python-docx
# cannot open that legacy format anyway, so requiring the ZIP signature for
# both extensions doesn't reject anything that would have worked downstream —
# it just rejects it earlier, with a clearer reason, instead of failing later
# inside the parser with a confusing exception.
_DOCX_LIKE_EXTENSIONS: frozenset[str] = frozenset({".docx", ".doc"})
_TEXT_LIKE_EXTENSIONS: frozenset[str] = frozenset({".txt", ".text", ".md"})


def _looks_like_text(data: bytes, sample_size: int = 8192) -> bool:
    """
    Heuristically determine whether a byte buffer is plausible plain text,
    as opposed to binary content wearing a .txt/.md extension.

    A NUL byte anywhere in the sample is an immediate disqualifier — no
    legitimate plain-text CV contains one, but they're extremely common in
    arbitrary binaries. Beyond that, we measure what fraction of bytes fall
    outside the "text" range (printable ASCII, common whitespace/control
    characters, and the high half of Latin-1/UTF-8 continuation bytes) and
    reject if that fraction is too high. The threshold is deliberately
    permissive enough to admit UTF-8 CVs containing non-Latin scripts
    (Arabic, accented characters, etc.), which are common and legitimate.
    """
    if not data:
        return True  # an empty file isn't binary; emptiness is handled elsewhere

    sample = data[:sample_size]
    if b"\x00" in sample:
        return False

    # Tab, LF, FF, CR, ESC, plus printable 0x20-0xFF (excluding DEL) are
    # treated as "text-compatible" bytes. This intentionally allows raw
    # UTF-8 continuation bytes (0x80-0xFF) since multi-byte UTF-8 sequences
    # use that range heavily and we don't want to reject legitimate
    # non-English CV content.
    text_byte_values = frozenset({0x07, 0x08, 0x09, 0x0A, 0x0C, 0x0D, 0x1B}) | (
        set(range(0x20, 0x100)) - {0x7F}
    )
    binary_byte_count = sum(1 for b in sample if b not in text_byte_values)
    binary_ratio = binary_byte_count / len(sample)

    return binary_ratio < 0.30


def validate_magic_bytes(file_bytes: bytes, filename: str) -> None:
    """
    Cross-check the actual leading bytes of an uploaded buffer against the
    file type implied by its declared extension. Raises HTTPException(400)
    on any mismatch or detected executable signature.

    This function trusts neither cv.content_type (client-supplied, trivially
    spoofed) nor the filename extension alone (also client-supplied) — it
    uses the extension only to select *which* signature to check for, and
    the actual bytes are what determine pass/fail.

    Parameters
    ----------
    file_bytes  Raw bytes already read from the upload (post size-guard).
    filename    Original filename, used only to select the expected
                signature family; never trusted as proof of file type.

    Raises
    ------
    HTTPException(400)  On any signature mismatch or detected binary/executable
                         content, using the generic SECURITY_VIOLATION_DETAIL
                         message so the failure reason isn't leaked to a caller
                         probing the validator.
    """
    if not file_bytes:
        # An empty upload isn't a forgery attempt; let the downstream
        # is_empty() check in the parsing stage produce the user-facing
        # "CV appears to be empty" message instead of a security error.
        return

    # Reject any known executable/binary signature unconditionally, no
    # matter what extension was declared. There is no CV format — PDF,
    # DOCX, or plain text — that legitimately begins with any of these.
    for signature, description in _EXECUTABLE_SIGNATURES:
        if file_bytes.startswith(signature):
            logger.warning(
                "Rejected upload '%s': detected %s signature disguised with a "
                "document extension.",
                filename,
                description,
            )
            raise HTTPException(status_code=400, detail=SECURITY_VIOLATION_DETAIL)

    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        if not file_bytes.startswith(_PDF_SIGNATURE):
            logger.warning(
                "Rejected upload '%s': declared .pdf but leading bytes are %r, "
                "not %r.",
                filename,
                file_bytes[:8],
                _PDF_SIGNATURE,
            )
            raise HTTPException(status_code=400, detail=SECURITY_VIOLATION_DETAIL)
        return

    if suffix in _DOCX_LIKE_EXTENSIONS:
        if not file_bytes.startswith(_ZIP_SIGNATURE):
            logger.warning(
                "Rejected upload '%s': declared %s but leading bytes are %r, "
                "not the expected ZIP/OOXML signature %r.",
                filename,
                suffix,
                file_bytes[:8],
                _ZIP_SIGNATURE,
            )
            raise HTTPException(status_code=400, detail=SECURITY_VIOLATION_DETAIL)
        return

    if suffix in _TEXT_LIKE_EXTENSIONS:
        if not _looks_like_text(file_bytes):
            logger.warning(
                "Rejected upload '%s': declared as text but content failed the "
                "plain-text heuristic (binary content detected).",
                filename,
            )
            raise HTTPException(status_code=400, detail=SECURITY_VIOLATION_DETAIL)
        return

    # Unknown/unsupported extension: let parse_cv_bytes raise its own
    # descriptive ValueError ("Unsupported file type '.xyz'") rather than
    # duplicating that logic or producing a misleading security message for
    # what is really just an unsupported-format error.


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
    1. File-size guard      — read up to MAX_CV_BYTES + 1 bytes; reject if over limit.
    2. Magic-bytes guard    — verify the buffer's actual leading bytes match the
                              file type implied by its extension; reject executables,
                              binary content disguised as text, and any other
                              signature mismatch BEFORE the bytes reach a parser.
    3. Parse                — cv_parser.parse_cv_bytes() handles PDF (PyMuPDF / pdfminer
                              fallback), DOCX (python-docx), and plain text.
    4. Sanity check         — reject obviously empty / image-only CVs early.
    5. Agent call           — run_cv_analysis() in a thread, wrapped with a timeout.
    6. Shape response       — attach lightweight profile metadata for the frontend.
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

    # ── 2. Magic-bytes guard ───────────────────────────────────────────────
    #
    # cv.content_type is set by the client and is not verified against the
    # body by FastAPI/Starlette, so it is never consulted here. Validation
    # is based exclusively on the actual bytes we just read. This runs
    # before parse_cv_bytes() so a forged upload never reaches PyMuPDF,
    # python-docx, or any other parsing library.
    validate_magic_bytes(file_bytes, cv.filename or "upload")

    # ── 3. Parse the CV ────────────────────────────────────────────────────
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

    # ── 4. Sanity check ────────────────────────────────────────────────────
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

    # ── 5. Run the agent ───────────────────────────────────────────────────
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

    # ── 6. Attach lightweight profile metadata ─────────────────────────────
    #
    # The frontend expects a "profile" key alongside the jobs array.
    # We populate skills from the first matched job (best available proxy when
    # the agent hasn't returned a dedicated skills field).
    result["profile"] = {
        "detected_title":   profile.detected_title,
        "word_count":       profile.word_count,
        "experience_level": "Professional",
       "skills": profile.skills,
        "summary": (
            "CV analysed successfully. Matching jobs based on extracted skills."
        ),
    }

    if preferred_location:
        result["location"] = preferred_location

    return result