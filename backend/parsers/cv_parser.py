"""
backend/parsers/cv_parser.py
============================
CV / résumé file parser.

Supports three formats:
    • PDF   — via PyMuPDF (fitz); falls back to pdfminer if fitz is not installed.
    • DOCX  — via python-docx.
    • TXT   — raw UTF-8 / latin-1 decode.

The module intentionally does NOT call any LLM here.  It only extracts raw text
and builds a lightweight ``CandidateProfile`` dataclass that the agent layer can
then reason about.  Keeping parsing and inference separate makes unit testing and
future parser swaps trivial.

Exported symbols
----------------
    CandidateProfile   — dataclass with the parsed information
    parse_cv_bytes     — main entry point; accepts (bytes, filename) → CandidateProfile
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CandidateProfile:
    """
    Structured representation of a parsed CV.

    Fields
    ------
    raw_text        Raw extracted text (truncated to config.max_cv_chars).
    file_name       Original file name (used for logging / error messages).
    detected_title  Best-guess job title inferred from the first non-blank lines.
    word_count      Number of words in raw_text (after truncation).
    parse_error     Non-empty if parsing produced a recoverable warning.
    """

    raw_text: str = ""
    file_name: str = "unknown"
    detected_title: str = ""
    word_count: int = 0
    parse_error: str = ""

    # Convenience helpers --------------------------------------------------

    def is_empty(self) -> bool:
        """Return True when no meaningful text was extracted."""
        return len(self.raw_text.strip()) < 50

    def summary_for_agent(self) -> str:
        """
        Return a compact, single-string summary suitable for injection into
        the LLM prompt.  The text is already truncated; this just wraps it
        with a minimal header so the model understands the context.
        """
        title_hint = f"Detected title: {self.detected_title}\n\n" if self.detected_title else ""
        return (
            f"=== CV CONTENT ({self.word_count} words) ===\n"
            f"{title_hint}"
            f"{self.raw_text}\n"
            f"=== END CV CONTENT ==="
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace and remove control characters."""
    # Replace non-breaking spaces and other Unicode whitespace with a regular space
    text = re.sub(r"[\u00a0\u2000-\u200f\u2028\u2029\ufeff]", " ", text)
    # Collapse 3+ consecutive newlines to two (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace from every line
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _truncate(text: str, max_chars: int) -> str:
    """Hard-truncate text and append a marker if content was cut."""
    if len(text) <= max_chars:
        return text
    logger.warning(
        "CV text truncated from %d → %d characters to protect context window.",
        len(text),
        max_chars,
    )
    return text[:max_chars] + "\n\n[... CV TRUNCATED FOR CONTEXT SAFETY ...]"


def _infer_title(text: str) -> str:
    """
    Naively try to extract a job title from the top of the CV.

    Strategy: look at the first 10 non-blank lines; if one of them is short
    (5–60 chars) and comes after a name-like line, use it as the title.
    Returns an empty string when detection fails (the agent will handle it).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:15]
    # Heuristic: short lines that are not email/phone/URL are candidate titles
    url_re = re.compile(r"(http|www\.|@|\+\d)", re.IGNORECASE)
    for line in lines[1:6]:  # skip the very first line (likely the name)
        if 5 <= len(line) <= 70 and not url_re.search(line):
            # Simple title-casing check: most job titles are ≥ 2 words OR
            # contain known keywords
            if re.search(
                r"(engineer|developer|analyst|manager|architect|scientist|"
                r"designer|consultant|lead|intern|specialist|officer|director)",
                line,
                re.IGNORECASE,
            ):
                return line
    return ""


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------

def _extract_pdf(data: bytes) -> str:
    """Extract text from a PDF using PyMuPDF (preferred) or pdfminer."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages_text)
    except ImportError:
        logger.debug("PyMuPDF not available, falling back to pdfminer.")

    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams

        output = io.StringIO()
        extract_text_to_fp(
            io.BytesIO(data),
            output,
            laparams=LAParams(),
            output_type="text",
            codec="utf-8",
        )
        return output.getvalue()
    except ImportError:
        raise RuntimeError(
            "No PDF library found.  Install either 'PyMuPDF' or 'pdfminer.six':\n"
            "    pip install PyMuPDF\n"
            "  or\n"
            "    pip install pdfminer.six"
        )


def _extract_docx(data: bytes) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        from docx import Document  # python-docx
    except ImportError:
        raise RuntimeError(
            "python-docx is required for .docx parsing.\n"
            "Install it with:  pip install python-docx"
        )

    doc = Document(io.BytesIO(data))
    paragraphs = [para.text for para in doc.paragraphs]
    # Also grab text from tables (common in modern CVs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.append(cell.text)
    return "\n".join(paragraphs)


def _extract_txt(data: bytes) -> str:
    """Decode plain-text bytes, trying UTF-8 then latin-1."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_cv_bytes(file_bytes: bytes, file_name: str) -> CandidateProfile:
    """
    Parse a CV from raw bytes.

    Parameters
    ----------
    file_bytes  Raw binary content of the uploaded file.
    file_name   Original file name including extension (e.g. ``"john_doe_cv.pdf"``).

    Returns
    -------
    CandidateProfile
        Always returns a profile object; ``parse_error`` is non-empty on soft
        failures (the agent can still attempt to work with partial text).

    Raises
    ------
    ValueError   When the file extension is not supported.
    RuntimeError When a required parsing library is missing.
    """
    settings = get_settings()
    suffix = Path(file_name).suffix.lower()

    logger.info("Parsing CV file: '%s' (%d bytes)", file_name, len(file_bytes))

    parse_error = ""
    raw_text = ""

    try:
        if suffix == ".pdf":
            raw_text = _extract_pdf(file_bytes)
        elif suffix in (".docx", ".doc"):
            raw_text = _extract_docx(file_bytes)
        elif suffix in (".txt", ".text", ".md"):
            raw_text = _extract_txt(file_bytes)
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'.  "
                "Please upload a PDF, DOCX, or TXT file."
            )
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:  # noqa: BLE001
        # Soft failure: record the error but return a partial profile so the
        # agent layer can surface a useful error message to the user.
        parse_error = f"Parsing error: {exc}"
        logger.exception("Unexpected error while parsing '%s'.", file_name)

    # Normalise and truncate -------------------------------------------------
    raw_text = _clean_text(raw_text)
    raw_text = _truncate(raw_text, settings.max_cv_chars)
    word_count = len(raw_text.split())
    detected_title = _infer_title(raw_text)

    profile = CandidateProfile(
        raw_text=raw_text,
        file_name=file_name,
        detected_title=detected_title,
        word_count=word_count,
        parse_error=parse_error,
    )

    if profile.is_empty() and not parse_error:
        profile.parse_error = "CV appears to be empty or contained no extractable text."
        logger.warning("Empty CV after parsing '%s'.", file_name)
    else:
        logger.info(
            "CV parsed successfully: %d words, title hint: '%s'",
            word_count,
            detected_title or "(none detected)",
        )

    return profile