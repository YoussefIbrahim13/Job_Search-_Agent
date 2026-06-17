# 🤖 Recruitment AI Agent System

A production-ready, **100% free** AI agent that finds real job opportunities using:
- **Groq** (`llama3-70b-8192`) — Free LLM inference
- **Tavily** — Free live web search (1,000 searches/month)
- **LangGraph** — Agentic ReAct loop with tool orchestration
- **LangChain-Groq** — Groq integration

---

## 📁 Project Structure

```
recruitment_agent/
├── main.py           # CLI entry point (Mode 1: CV, Mode 2: Job Search)
├── agent.py          # LangGraph ReAct agent, system prompt, output parsing
├── cv_parser.py      # PDF/DOCX CV text extraction + Groq structured parsing
├── tools.py          # Tavily search, web scraper, LinkedIn, company research
├── config.py         # Settings management (.env loader)
├── requirements.txt  # Python dependencies
├── .env.example      # API key template → copy to .env
└── README.md         # This file
```

---

## ⚙️ Architecture

```
                    ┌────────────────────────────────────────────┐
                    │           CLI (main.py)                     │
                    └────────────┬──────────────┬────────────────┘
                                 │              │
                    ┌────────────▼──┐    ┌──────▼──────────────┐
                    │  MODE 1: CV   │    │  MODE 2: JOB SEARCH  │
                    │  cv_parser.py │    │  (title + location)  │
                    └────────────┬──┘    └──────┬───────────────┘
                                 │              │
                    ┌────────────▼──────────────▼────────────────┐
                    │              agent.py                        │
                    │                                              │
                    │  ┌──────────────────────────────────────┐   │
                    │  │        LangGraph ReAct Loop           │   │
                    │  │                                      │   │
                    │  │  [SystemPrompt + Task]               │   │
                    │  │        ↓                             │   │
                    │  │  [Groq LLM – llama3-70b]  ◄─────┐   │   │
                    │  │        ↓ (tool_calls)           │   │   │
                    │  │  [Tool Node] ──────────────────►─┘   │   │
                    │  │        ↓ (final answer)              │   │
                    │  │  [Output Parser → JobListing[]]      │   │
                    │  └──────────────────────────────────────┘   │
                    └────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────────┐
                    │     TOOLS (tools.py)                       │
                    │                                            │
                    │  ┌─────────────────────────────────────┐  │
                    │  │ TavilyJobSearchTool                  │  │
                    │  │   → Search job boards + career pages │  │
                    │  ├─────────────────────────────────────┤  │
                    │  │ LinkedInJobSearchTool                │  │
                    │  │   → LinkedIn-specific job search    │  │
                    │  ├─────────────────────────────────────┤  │
                    │  │ WebPageScraperTool                   │  │
                    │  │   → requests + BeautifulSoup         │  │
                    │  ├─────────────────────────────────────┤  │
                    │  │ CompanyResearchTool                  │  │
                    │  │   → Company info + career page URL   │  │
                    │  └─────────────────────────────────────┘  │
                    └────────────────────────────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────┐
                    │          Structured Output                  │
                    │                                            │
                    │  ┌──────────────────────────────────────┐ │
                    │  │  Rich CLI Table / JSON / Markdown     │ │
                    │  │                                      │ │
                    │  │  Company | Title | Location | Skills │ │
                    │  │  Salary  | Type  | Match    | Link   │ │
                    │  └──────────────────────────────────────┘ │
                    └────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd recruitment_agent
pip install -r requirements.txt
```

### 2. Get free API keys

| Service | URL | Free Tier |
|---------|-----|-----------|
| **Groq** | https://console.groq.com/keys | Free — very fast inference |
| **Tavily** | https://app.tavily.com | Free — 1,000 searches/month |

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
nano .env
```

### 4. Run

```bash
# Interactive menu
python main.py

# Mode 1: Analyse a CV
python main.py --mode cv --cv ./resume.pdf

# Mode 1: CV with JSON output
python main.py --mode cv --cv ./cv.docx --output json

# Mode 2: Search by job title
python main.py --mode search --title "Python Backend Engineer" --location "Berlin"

# Mode 2: Remote jobs with specific skills
python main.py --mode search --title "Data Scientist" --location "Remote" --skills "Python,SQL,PyTorch"

# Mode 2: Markdown output (for piping to a file)
python main.py --mode search --title "DevOps Engineer" --location "London" --output markdown > jobs.md
```

---

## 📊 Output Example

```
╭──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ 🎯  Found 8 Job Opportunities                                                                    │
├───┬──────────────┬────────────────────────────┬──────────────────┬─────────────────────────────┤
│ # │ Company      │ Job Title                  │ Location         │ Required Skills             │
├───┼──────────────┼────────────────────────────┼──────────────────┼─────────────────────────────┤
│ 1 │ Stripe       │ Senior Backend Engineer    │ London, UK       │ Python, Go, Kafka, gRPC     │
│ 2 │ Revolut      │ Python Engineer            │ Remote (EU)      │ Python, FastAPI, PostgreSQL  │
│ 3 │ Monzo        │ Backend Engineer           │ London / Remote  │ Go, Kubernetes, Microservice │
╰───┴──────────────┴────────────────────────────┴──────────────────┴─────────────────────────────╯
```

---

## 🗂️ File Reference

### `config.py`
Loads `.env` variables. Exits with a clear error if required keys are missing.
Exposes a `settings` singleton with typed attributes.

### `cv_parser.py`
- `extract_raw_text(path)` → Handles PDF (pdfplumber + pypdf fallback) and DOCX
- `parse_cv(path)` → Returns `CVProfile` dataclass with all fields
- Uses Groq to extract structured JSON from raw CV text (zero-shot)

### `tools.py`
Four `BaseTool` subclasses registered with LangGraph:
1. `TavilyJobSearchTool` — Multi-domain job board search
2. `LinkedInJobSearchTool` — LinkedIn-restricted search
3. `WebPageScraperTool` — requests + BS4, rate-limited, noise-filtered
4. `CompanyResearchTool` — Career page + open roles research

### `agent.py`
- `build_agent_graph()` → Compiles the LangGraph `StateGraph`
- `run_agent(task)` → Executes the loop, returns `(list[JobListing], raw_str)`
- `parse_job_listings(output)` → Extracts `<RECRUITMENT_RESULTS>` JSON
- `build_cv_task_prompt(profile)` → Task string for Mode 1
- `build_search_task_prompt(title, location)` → Task string for Mode 2

### `main.py`
- `mode_cv()` → Parse CV → build task → run agent → render
- `mode_search()` → Build task → run agent → render
- `render_table/json/markdown()` → Three output formats via `rich`

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `TAVILY_API_KEY` | *(required)* | Your Tavily API key |
| `GROQ_MODEL` | `llama3-70b-8192` | Groq model name |
| `LLM_TEMPERATURE` | `0.1` | Creativity (0=deterministic) |
| `LLM_MAX_TOKENS` | `4096` | Max output tokens |
| `TAVILY_MAX_RESULTS` | `8` | Results per search |
| `TAVILY_SEARCH_DEPTH` | `advanced` | `basic` or `advanced` |
| `MAX_AGENT_ITERATIONS` | `10` | LangGraph recursion limit |
| `OUTPUT_FORMAT` | `table` | `table`, `json`, or `markdown` |

---

## 🔄 How the Agent Loop Works

```
1. User provides task (CV profile or job search request)
   ↓
2. LangGraph sends [SystemPrompt + Task] to Groq
   ↓
3. Groq (llama3-70b) reasons and decides which tool to call
   ↓
4. Tool executes (Tavily search / web scrape / LinkedIn / company research)
   ↓
5. Tool result fed back to Groq
   ↓
6. Steps 3–5 repeat until Groq decides it has enough data
   ↓
7. Groq outputs final answer with <RECRUITMENT_RESULTS> JSON block
   ↓
8. Parser extracts JobListing objects
   ↓
9. CLI renders as table / JSON / Markdown
```

---

## 💡 Tips & Troubleshooting

**Rate limits:** Groq free tier has RPM limits. If you hit them, the `tenacity` retry decorator will back off and retry automatically.

**Scanned PDFs:** If your CV is a scanned image, pdfplumber will return empty text. Convert to a text-layer PDF first using Adobe Acrobat or an online OCR tool.

**No results?** Try broader job titles in Mode 2 (e.g. "Engineer" instead of "Senior Principal Platform Engineer").

**Tavily quota:** The free tier gives 1,000 searches/month. Each agent run typically uses 3–6 searches.

**Switching models:** Edit `GROQ_MODEL` in `.env`. `llama-3.1-70b-versatile` offers a larger context window (128K) which is useful for very detailed CVs.

# Job_Search-_Agent