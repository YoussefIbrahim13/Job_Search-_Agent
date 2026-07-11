<div align="center">

<br />

```
██████╗ ███████╗ ██████╗██████╗ ██╗   ██╗██╗████████╗██████╗  ██████╗ ████████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██║   ██║██║╚══██╔══╝██╔══██╗██╔═══██╗╚══██╔══╝
██████╔╝█████╗  ██║     ██████╔╝██║   ██║██║   ██║   ██████╔╝██║   ██║   ██║   
██╔══██╗██╔══╝  ██║     ██╔══██╗██║   ██║██║   ██║   ██╔══██╗██║   ██║   ██║   
██║  ██║███████╗╚██████╗██║  ██║╚██████╔╝██║   ██║   ██████╔╝╚██████╔╝   ██║   
╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝   ╚═╝   ╚═════╝  ╚═════╝    ╚═╝   
```

### AI-Powered Job Search Agent

*A production-grade, multi-layer intelligence pipeline for finding real, open jobs — not zombie postings.*

<br />

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Tavily](https://img.shields.io/badge/Tavily-Job_Search-4A90E2?style=for-the-badge)](https://tavily.com/)
[![License](https://img.shields.io/badge/License-MIT-22FF88?style=for-the-badge)](LICENSE)

</div>

---

## ✦ What Is This?

**RecruitBot** is an autonomous recruitment AI agent that finds *real, currently open* job postings tailored to a candidate's skill stack — from a live job search or a raw CV upload. It doesn't just scrape; it reasons, filters, scores, and validates every listing through a 5-layer quality pipeline before surfacing results.

Built on **LangGraph**, powered by **Groq's LLaMA 3.3 70B**, and backed by **Tavily** for structured web search across 11 approved job boards, it handles everything from query construction to live staleness detection — including Arabic posting-age strings.

---

## ✦ Live Preview

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║          SYSTEM ONLINE — RECRUITMENT AI AGENT               ║
║                                                              ║
║  ┌─ INPUT MODE ─────────────────────────────────────────┐   ║
║  │  [ Upload CV ]          [ Search by Title + Location ] │   ║
║  └─────────────────────────────────────────────────────-─┘   ║
║                                                              ║
║  ┌─ AGENT PIPELINE ─────────────────────────────────────┐   ║
║  │                                                        │   ║
║  │  CV PARSE → SKILL EXTRACT → QUERY BUILD → SEARCH      │   ║
║  │      ↓             ↓             ↓           ↓        │   ║
║  │  [PyMuPDF]   [Regex Match]  [LangGraph]  [Tavily]     │   ║
║  │                                           ↓            │   ║
║  │           PATH GATE → STALENESS → LIVE PROBE           │   ║
║  │                              ↓                         │   ║
║  │                     VALIDATED LISTINGS                  │   ║
║  └────────────────────────────────────────────────────────┘   ║
║                                                              ║
║  ┌─ JOB CARD ────────────────────────────────────────────┐   ║
║  │  Senior Python Developer · Cairo                       │   ║
║  │  Match: 87/100 · Wuzzuf · Posted: 3 days ago           │   ║
║  │  Skills: Python, Django, PostgreSQL, REST, Docker       │   ║
║  └────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════╝
```

> 🚀 **[Live Demo Placeholder]** — Replace this with a screenshot or GIF of RecruitBot in action. Hosted demo URL: `https://your-deployment-url.com`

</div>

---

## ✦ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Backend Framework** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-2E303E?style=flat-square) |
| **Agent Orchestration** | ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-Core-1C3C3C?style=flat-square&logo=langchain&logoColor=white) |
| **LLM Provider** | ![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat-square) *(Ollama/local supported)* |
| **Web Search** | ![Tavily](https://img.shields.io/badge/Tavily_AI-Search-4A90E2?style=flat-square) |
| **CV Parsing** | ![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF-CC0000?style=flat-square) ![pdfminer](https://img.shields.io/badge/pdfminer.six-Fallback-777?style=flat-square) ![python-docx](https://img.shields.io/badge/python--docx-DOCX-2B579A?style=flat-square) |
| **Configuration** | ![Pydantic](https://img.shields.io/badge/Pydantic-Settings-E92063?style=flat-square&logo=pydantic&logoColor=white) ![dotenv](https://img.shields.io/badge/dotenv-Secrets-ECD53F?style=flat-square) |
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white) ![Vanilla JS](https://img.shields.io/badge/Vanilla_JS-F7DF1E?style=flat-square&logo=javascript&logoColor=black) |
| **Language** | ![Python](https://img.shields.io/badge/Python_3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white) |

</div>

---

## ✦ Architecture Deep Dive

```
┌────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                    │
│          RecruitBot UI (static/index.html)  ←→  FastAPI Backend        │
│               /api/analyze-cv      /api/targeted-search                 │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                    CV PROCESSING PIPELINE                               │
│                                                                         │
│  UploadFile → [Size Guard 10MB] → [Magic-Bytes Validation]             │
│                     │                      │                            │
│               *.pdf → PyMuPDF     *.docx → python-docx                 │
│             (column-aware)        (tables included)                     │
│                     │                                                   │
│              _clean_text() → _truncate() → _infer_title()              │
│              _harvest_skills()  ← deterministic, pre-compiled regex    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                   LANGGRAPH AGENT GRAPH                                 │
│                                                                         │
│  START → [llm_node (Groq)] → _route()                                  │
│                │                  │                                     │
│         tool_calls?         no tool call?                               │
│                │                  │                                     │
│         [tool_node]     [graceful_exit_node]                            │
│                │            (FIX R fallback)                            │
│                │                  │                                     │
│          [llm_node] ←─────────────┘                                    │
│          [coerce_internship_node] if intern query missing               │
│                                                                         │
│  LLM fires 2 PARALLEL Tavily queries per turn:                         │
│    ① full-time/senior  →  site:wuzzuf.net OR site:bayt.com            │
│    ② internship/trainee →  site:linkedin.com/jobs                      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                  5-LAYER QUALITY FILTER (tools.py)                      │
│                                                                         │
│  Layer 1 — Domain Blacklist     → zombie aggregators blocked           │
│  Layer 2 — Content Pollution    → forums, Q&A, blogs removed           │
│  Layer 3 — Path Gate (FIX N)    → non-listing subpaths rejected        │
│  Layer 4 — Staleness Scan (FIX O) → age badges, closed declarations   │
│  Layer 5 — Live Shallow Probe   → 8KB head fetch for Wuzzuf/LinkedIn  │
└────────────────────────────────────────────────────────────────────────┘
```

### The Agent State Machine

The LangGraph graph compiles a deterministic state machine with four nodes:

| Node | Role |
|---|---|
| `llm_node` | Drives Groq's LLaMA 3.3 70B with tool bindings; issues parallel Tavily calls |
| `tool_node` | Executes `tavily_job_search` + `scrape_job_page` tools |
| `coerce_internship_node` | Injects a follow-up message if the internship query was skipped |
| `graceful_exit_node` | Extracts + validates JSON; falls back to ToolMessage recovery (FIX R) |

---

## ✦ Feature Highlights

### 🔍 Dual-Mode Search
- **CV Upload** — Accepts PDF, DOCX, or TXT. Skills are extracted deterministically via a pre-compiled regex vocabulary of 100+ tech terms before being injected into the LLM prompt. No hallucinated skills.
- **Targeted Search** — Direct job title + location query. The agent constructs valid Tavily queries from inline tech signals in the title.

### 🧠 Intelligent Query Construction
The system prompt enforces strict query grammar: `<TECH> <ROLE> <LOCATION> <modifier> site:<BOARD>`. A sanitiser strips natural-language prefixes ("please find me...", "search for...") before the query reaches Tavily, and a stray-token stripper prevents Go-language searches from being accidentally injected into `.NET` queries.

### 🛡️ 5-Layer Anti-Zombie Pipeline
Every result from Tavily passes through:
1. **Domain blacklist** — Known CSR zombie aggregators (GulfTalent, NaukriGulf, etc.)
2. **Content pollution filter** — Quora, Reddit, Medium, Stack Overflow, and 30+ more
3. **Positive-Assertion Path Gate** — Per-board canonical URL regex validation (e.g. Wuzzuf `/jobs/p/<slug>`, LinkedIn `/jobs/view/<id>`)
4. **Snippet staleness scan** — Detects `"Posted 5 years ago"`, `"منذ 4 سنوات"`, and closed/filled declarations in both English and Arabic
5. **Live shallow probe** — Fetches the first 8KB of the raw HTML for Wuzzuf and LinkedIn listings, checks for closure badges before sidebar noise corrupts the signal

### 🔒 Upload Security
- **File size guard** — Rejects uploads over 10 MB before the body is fully read
- **Magic-bytes validation** — Cross-checks file content against declared extension; rejects executables (ELF, PE, Mach-O, shebang scripts) regardless of filename
- **Binary text detection** — NUL-byte sniffing and binary ratio heuristic for `.txt`/`.md` uploads

### 📊 Match Scoring (0–100)
Each job listing receives a differentiated score:

| Component | Max Points | Criteria |
|---|---|---|
| `TITLE_MATCH` | 50 | Exact match → 50, same family → 35, adjacent → 20, intern → 10 |
| `LOCATION_MATCH` | 30 | Exact → 30, same country → 15, remote → 5 |
| `INFO_QUALITY` | 20 | +5 each: real salary, explicit experience, ≥3 skills, confirmed URL |

### 🌐 11 Approved Job Boards

| Category | Boards |
|---|---|
| Global | LinkedIn Jobs, Indeed, Glassdoor |
| MENA / Egypt / Gulf | Wuzzuf, Bayt, Akhtaboot |
| Remote-focused | WeWorkRemotely, RemoteOK, Himalayas |
| Tech-specialist | Wellfound (AngelList), Dice |

---

## ✦ Getting Started

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/) (LLaMA 3.3 70B)
- A free [Tavily API key](https://app.tavily.com/) (1,000 searches/month on free tier)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YoussefIbrahim13/Job_Search-_Agent.git
cd Job_Search-_Agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the `.env` template and fill in your keys:

```bash
# .env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx

# Model selection (default: LLaMA 3.3 70B on Groq)
GROQ_MODEL=llama-3.3-70b-versatile

# Tuning knobs
MAX_AGENT_ITERATIONS=3    # Recommended ceiling — don't exceed without reason
TAVILY_MAX_RESULTS=8      # Results fetched per Tavily query
MAX_CV_CHARS=2000         # CV text sent to the LLM (context safety)
```

> **Optional — Local LLM via Ollama:** Comment out the `GROQ_*` keys and uncomment the `OLLAMA_*` block in `.env`. Ensure `ollama serve` is running and pull your model (`ollama pull llama3.1:8b`).

### Running the Server

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser. The static frontend is served automatically by FastAPI.

---

## ✦ API Reference

### `POST /api/targeted-search`

Search for jobs by title and location.

```json
// Request body
{
  "job_title": "Python Backend Developer",
  "location": "Cairo"
}
```

```json
// Response
{
  "job_title": "Python Backend Developer",
  "location": "Cairo",
  "total_found": 4,
  "agent_summary": "Searched Wuzzuf/Bayt for full-time and LinkedIn for internships in Cairo. Found 4 validated listings.",
  "search_queries_used": [
    "Python Django Backend Developer Cairo jobs site:wuzzuf.net OR site:bayt.com",
    "Python Django Backend Developer Cairo internship site:linkedin.com/jobs"
  ],
  "jobs": [
    {
      "company_name": "ACME Tech",
      "job_title": "Senior Python Developer",
      "match_score": 87,
      "location": "Cairo, Egypt",
      "experience_needed": "3-5 years",
      "salary_range": "Not specified",
      "required_skills": ["Python", "Django", "PostgreSQL", "REST", "Docker"],
      "match_reason": "Exact title match for Python Backend Developer in Cairo.",
      "source": "wuzzuf.net",
      "application_link": "https://wuzzuf.net/jobs/p/..."
    }
  ]
}
```

### `POST /api/analyze-cv`

Upload a CV for automated skill extraction and job matching.

```bash
curl -X POST http://localhost:8000/api/analyze-cv \
  -F "cv=@my_resume.pdf" \
  -F "preferred_location=Cairo"
```

The response includes a `profile` object alongside the standard jobs array:

```json
{
  "profile": {
    "detected_title": "Software Engineer",
    "word_count": 312,
    "experience_level": "Professional",
    "skills": ["C#", ".NET Core", "ASP.NET", "SQL Server", "Docker", "Redis"]
  },
  "jobs": [...]
}
```

---

## ✦ Project Structure

```
Job_Search-_Agent/
│
├── backend/
│   ├── main.py                    # FastAPI app factory, lifespan, CORS
│   │
│   ├── core/
│   │   └── config.py              # Pydantic-settings with lru_cache
│   │
│   ├── api/
│   │   └── routes.py              # HTTP endpoints; magic-bytes validation
│   │
│   ├── agents/
│   │   ├── recruitment_agent.py   # LangGraph graph, system prompt, scoring
│   │   ├── tools.py               # Tavily tool + 5-layer filter pipeline
│   │   └── diagnose_live_probe.py # Diagnostic utility for live probe tuning
│   │
│   └── parsers/
│       └── cv_parser.py           # Multi-format CV parser; spatial block sort
│
├── static/
│   └── index.html                 # Cyberpunk-themed single-file frontend
│
├── requirements.txt
└── .env                           # API keys (not committed)
```

---

## ✦ Design Decisions & Engineering Notes

**Why LangGraph over a simple chain?**
The job-search workflow is inherently stateful: the agent must fire two parallel tool calls, detect whether the internship query ran, inject a coercion message if it didn't, and handle graceful degradation when the iteration cap fires mid-tool-call. A state graph with explicit routing logic is far more auditable than a prompt-jailbreak approach.

**Why a deterministic skill harvester in the parser?**
LLMs hallucinate skills. Pre-extracting a canonical list from the CV text and injecting it verbatim into the prompt (rather than asking the LLM to "read the CV and identify skills") eliminates the main source of phantom-skill-based bad queries.

**Why 8KB for the Wuzzuf/LinkedIn live probe?**
The closure badge and `posted-X-ago` string both render in the first few hundred bytes of the page's `<main>` content — well before any `<aside>`, `<footer>`, or sidebar widget that would introduce false positives. 8KB reliably captures both while staying safely short of company-bio bleed on Wuzzuf and repost-badge noise on LinkedIn.

**Why is `MAX_AGENT_ITERATIONS=3` the recommended ceiling?**
Groq's API enforces per-minute token limits. At 3 iterations (1 LLM call + 2 tool executions + 1 finalisation call), the agent stays comfortably within free-tier rate limits while covering both the full-time and internship query paths.

---

## ✦ Roadmap

- [ ] **Cover Letter Generator** — Use the extracted CV + job description to draft a personalised cover letter
- [ ] **Application Tracker** — Persist job cards to a lightweight SQLite store with status tags
- [ ] **Streaming Responses** — SSE-based streaming of job cards as they clear the filter pipeline
- [ ] **Multi-language CV Support** — Arabic CV parsing and Arabic job-title query construction
- [ ] **Webhook / Scheduler** — Run daily searches and push new listings to email or Slack

---

## ✦ Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you'd like to change.

```bash
# Fork → clone → branch
git checkout -b feature/my-improvement

# Make changes, then
git commit -m "feat: describe your change"
git push origin feature/my-improvement
# Open a PR
```

---

## ✦ License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

Built with 🔮 by [YoussefIbrahim13](https://github.com/YoussefIbrahim13)

*Find the job. Skip the noise.*

</div>
