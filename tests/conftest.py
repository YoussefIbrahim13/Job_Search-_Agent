"""
Shared pytest configuration.

The modules under test read settings at import time, so populate the required
keys with dummies before anything imports `backend.*`. Without this, importing
`backend.agents.tools` in a clean checkout picks up whatever happens to be in
the developer's real `.env`, making test outcomes machine-dependent.
"""
import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("LOG_LEVEL", "WARNING")
