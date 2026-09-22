# AI Research Agent — 100 to 500 Page Mode

Streamlit + CrewAI + Groq + DuckDuckGo long-form research application.

## Features
- User enters any topic.
- Target length selector from 100 to 500 pages.
- Multiple web research rounds.
- Chapter-by-chapter generation.
- Groq `openai/gpt-oss-120b`.
- DuckDuckGo via `ddgs`.
- Markdown download.
- Source URL collection.
- Word/page estimates.

## Important
A physical page is formatting-dependent. This project plans around 500 words/page. Actual pages vary by font, margins, spacing, tables and export format.

A 500-page report can require many LLM calls and a long runtime. API limits, Streamlit limits, search availability and model output limits still apply.

## Streamlit Secrets
Add:
GROQ_API_KEY = "your_key"

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Streamlit Cloud
Use Python 3.12 from `runtime.txt` and add `GROQ_API_KEY` under App Settings -> Secrets.


## FINAL ImportError Fix

The original Streamlit startup error came from the import chain:
`app.py -> research_agent.py -> crewai -> search_tool.py -> crewai.tools.BaseTool`.

`search_tool.py` now uses a plain Python wrapper because the application calls `_run()` directly. `research_agent.py` also imports CrewAI lazily only when research actually starts.

The requirements file uses:
`crewai[litellm]==1.15.22`

Python runtime:
`3.12`

After uploading these files to GitHub, commit/push and use Streamlit Cloud:
**Manage app -> Reboot app**.

Required Streamlit Secret:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```
