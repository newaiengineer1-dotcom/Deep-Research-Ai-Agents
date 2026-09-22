import os
import re

from crewai import Agent, Task, Crew, LLM


# ============================================================
# GOOGLE GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Add GEMINI_API_KEY to Streamlit Cloud Secrets."
    )


# IMPORTANT:
# Do NOT use:
#   gpt-oss-120b
#   openai/gpt-oss-120b
#   google/gemini...
#
# CrewAI Gemini format:
MODEL_NAME = "gemini/gemini-2.5-flash"


llm = LLM(
    model=MODEL_NAME,
    api_key=GEMINI_API_KEY,
    temperature=0.2,
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(text: str) -> str:
    """Clean generated text."""

    if not text:
        return ""

    text = str(text)

    # Remove excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]{3,}", " ", text)

    return text.strip()


def validate_page_limit(max_pages: int) -> int:
    """Keep report pages between 5 and 500."""

    try:
        max_pages = int(max_pages)
    except Exception:
        max_pages = 10

    return max(5, min(500, max_pages))


def target_words(max_pages: int) -> int:
    """
    Approximate 500 words per report page.
    """

    return max_pages * 500


# ============================================================
# OUTLINE GENERATION
# ============================================================

def build_outline(
    topic: str,
    evidence: str,
    max_pages: int,
) -> str:

    pages = validate_page_limit(max_pages)
    words = target_words(pages)

    agent = Agent(
        role="Senior Research Analyst",
        goal=(
            "Create a comprehensive and logically structured "
            "research report outline."
        ),
        backstory=(
            "You are an experienced research analyst specializing "
            "in technology, energy, business, markets and engineering."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Create a detailed research outline for:

TOPIC:
{topic}

TARGET REPORT:
Approximately {pages} pages
Approximately {words:,} words

ADDITIONAL EVIDENCE:
{evidence if evidence else "No additional evidence was provided."}

Create a professional report structure containing:

1. Executive Summary
2. Introduction
3. Background
4. Market / Industry Overview
5. Current Situation
6. Key Technologies / Developments
7. Regional Analysis
8. Market Trends
9. Challenges and Risks
10. Opportunities
11. Technical / Economic Analysis where applicable
12. Future Outlook
13. Conclusions
14. References / Sources

Adapt the structure to the actual research topic.

Do not write the full report yet.
Only produce the detailed outline.

The outline should contain enough sections and subsections
to support a report of approximately {pages} pages.
""",
        expected_output="A detailed professional research report outline.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()

    return clean_text(str(result))


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    topic: str,
    outline: str,
    evidence: str,
    max_pages: int,
) -> str:

    pages = validate_page_limit(max_pages)
    words = target_words(pages)

    agent = Agent(
        role="Senior Research Report Writer",
        goal=(
            "Produce a comprehensive, factual, professional and "
            "well-structured research report."
        ),
        backstory=(
            "You are a senior research consultant experienced in "
            "technology, renewable energy, BESS, engineering, "
            "business and market research."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Write a professional research report.

==================================================
RESEARCH TOPIC
==================================================

{topic}

==================================================
TARGET LENGTH
==================================================

Maximum requested pages: {pages}

Approximate target:
{words:,} words

==================================================
REPORT OUTLINE
==================================================

{outline}

==================================================
ADDITIONAL EVIDENCE
==================================================

{evidence if evidence else "No additional evidence was supplied."}

==================================================
REPORT REQUIREMENTS
==================================================

Create a detailed professional report.

Requirements:

- Start with a clear title.
- Include an Executive Summary.
- Use Markdown headings.
- Use numbered sections where appropriate.
- Use tables where useful.
- Explain technical concepts clearly.
- Include quantitative information where supported.
- Clearly distinguish facts from estimates.
- Do not invent statistics.
- Do not invent sources.
- Do not fabricate citations.
- If information is unavailable, state that it is unavailable.
- Avoid repetitive content.
- Maintain professional research-report language.
- Make the report useful for business and technical decision-making.
- Include a conclusion.
- Include a references/source section when source information
  is actually available.

IMPORTANT:
The report should be approximately {words:,} words or less.
Do not intentionally exceed the requested maximum length.

Return ONLY the final report.
""",
        expected_output=(
            "A complete professional research report in Markdown format."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()

    return clean_text(str(result))


# ============================================================
# PAGE LIMIT ENFORCEMENT
# ============================================================

def enforce_page_limit(
    report: str,
    max_pages: int,
) -> str:

    pages = validate_page_limit(max_pages)

    # Approximate 500 words per page
    maximum_words = pages * 500

    words = report.split()

    if len(words) <= maximum_words:
        return report

    truncated = " ".join(words[:maximum_words])

    return (
        truncated
        + "\n\n---\n\n"
        + f"*Report truncated to approximately {pages} pages "
          "according to the selected maximum page limit.*"
    )


# ============================================================
# MAIN FUNCTION USED BY APP.PY
# ============================================================

def generate_long_research_report(
    topic: str,
    evidence: str = "",
    max_pages: int = 10,
) -> str:

    max_pages = validate_page_limit(max_pages)

    if not topic or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    # Step 1 — Build research structure
    outline = build_outline(
        topic=topic.strip(),
        evidence=evidence.strip(),
        max_pages=max_pages,
    )

    # Step 2 — Generate report
    report = generate_report(
        topic=topic.strip(),
        outline=outline,
        evidence=evidence.strip(),
        max_pages=max_pages,
    )

    # Step 3 — Enforce requested page limit
    report = enforce_page_limit(
        report=report,
        max_pages=max_pages,
    )

    return report
