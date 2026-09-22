import os
import re
from crewai import Agent, Task, Crew, LLM


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Add GEMINI_API_KEY to Streamlit Cloud Secrets."
    )

MODEL_NAME = "gemini/gemini-3.6-flash"

llm = LLM(
    model=MODEL_NAME,
    api_key=GEMINI_API_KEY,
    temperature=0.2,
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = str(text)

    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def validate_page_limit(max_pages):
    try:
        max_pages = int(max_pages)
    except Exception:
        max_pages = 10

    return max(5, min(500, max_pages))


def target_words(max_pages):
    # Professional reports normally vary considerably,
    # so this is an approximate planning target.
    return max_pages * 450


# ============================================================
# OUTLINE
# ============================================================

def build_outline(topic, evidence, max_pages):

    max_pages = validate_page_limit(max_pages)
    words = target_words(max_pages)

    agent = Agent(
        role="Senior Research Director",
        goal=(
            "Design a professional, comprehensive and logically "
            "structured research report."
        ),
        backstory=(
            "You are a senior research consultant experienced in "
            "technology, renewable energy, BESS, engineering, "
            "markets, finance and strategic research."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Create a detailed professional outline for the following research report.

TOPIC:
{topic}

TARGET:
Approximately {max_pages} pages
Approximately {words:,} words

ADDITIONAL EVIDENCE:
{evidence if evidence else "No additional evidence supplied."}

The outline must be suitable for a professional consulting,
engineering, market intelligence or investment report.

Include appropriate sections such as:

Executive Summary
Key Findings
Introduction
Background
Objectives
Scope
Methodology
Market / Industry Overview
Technology Landscape
Market Analysis
Regional Analysis
Competitive Landscape
Economic / Financial Analysis
Risks and Challenges
Opportunities
Scenario Analysis
Future Outlook
Conclusions
References
Appendices

Adapt the structure to the actual research topic.

Do not write the report.
Only create the detailed hierarchical outline.
""",
        expected_output="Detailed professional hierarchical report outline.",
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

def generate_report(topic, outline, evidence, max_pages):

    max_pages = validate_page_limit(max_pages)
    words = target_words(max_pages)

    agent = Agent(
        role="Senior Research Report Author",
        goal=(
            "Produce an accurate, professional and publication-ready "
            "research report."
        ),
        backstory=(
            "You are an experienced research analyst and technical "
            "consultant. You produce professional reports used by "
            "engineers, executives, investors and project developers."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Write the final professional research report.

============================================================
TOPIC
============================================================

{topic}

============================================================
TARGET LENGTH
============================================================

Maximum requested length: {max_pages} pages
Approximate target: {words:,} words

============================================================
OUTLINE
============================================================

{outline}

============================================================
ADDITIONAL EVIDENCE
============================================================

{evidence if evidence else "No additional evidence supplied."}

============================================================
MANDATORY REPORT FORMAT
============================================================

Return the report in clean Markdown.

Use this hierarchy:

# REPORT TITLE

## Executive Summary

## Key Findings

## 1. Introduction

### 1.1 Background

### 1.2 Objectives

### 1.3 Scope

### 1.4 Methodology

## 2. ...

### 2.1 ...

### 2.2 ...

Continue the numbering according to the actual report.

============================================================
PROFESSIONAL REPORT REQUIREMENTS
============================================================

1. Use clear numbered sections.

2. Use subsections where useful.

3. Include an Executive Summary.

4. Include Key Findings.

5. Include professional tables when quantitative
   information is available.

6. Every table must use this format:

Table X-X: Table Title

| Column | Column | Column |
|---|---|---|
| Data | Data | Data |

7. Immediately after a table include:

Source: [source if genuinely available]

8. Clearly distinguish:
   - Actual data
   - Estimates
   - Forecasts
   - Calculations
   - Assumptions

9. Do not invent statistics.

10. Do not invent citations.

11. Do not fabricate URLs.

12. If reliable source information is unavailable,
    state that clearly.

13. Include quantitative analysis where appropriate.

14. Include regional comparisons where relevant.

15. Include risks and limitations.

16. Include future outlook.

17. Include a conclusion.

18. Include References only for sources actually available
    from the provided evidence/research.

19. Include Appendices when appropriate.

20. Do not use conversational language.

21. Do not mention that you are an AI.

22. Do not include a preamble before the report.

============================================================
TABLE DESIGN
============================================================

Use tables for:

- Market size
- Capacity
- Revenue
- Growth
- Technology comparison
- Regional comparison
- Cost comparison
- Financial assumptions
- Risk matrix
- Scenario analysis
- Project parameters

============================================================
QUALITY
============================================================

The report should read like a professional consulting,
engineering or market intelligence report.

Avoid unnecessary repetition.

Use concise paragraphs.

Use bullet lists when useful.

Use professional terminology.

Return ONLY the final report.
""",
        expected_output="Complete publication-ready professional research report in Markdown.",
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
# PAGE LIMIT
# ============================================================

def enforce_page_limit(report, max_pages):

    max_pages = validate_page_limit(max_pages)

    maximum_words = max_pages * 500

    words = report.split()

    if len(words) <= maximum_words:
        return report

    return (
        " ".join(words[:maximum_words])
        + "\n\n---\n\n"
        f"*Report limited to approximately {max_pages} pages.*"
    )


# ============================================================
# MAIN API
# ============================================================

def generate_long_research_report(
    topic,
    evidence="",
    max_pages=10,
):

    max_pages = validate_page_limit(max_pages)

    if not topic or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    outline = build_outline(
        topic=topic.strip(),
        evidence=evidence.strip(),
        max_pages=max_pages,
    )

    report = generate_report(
        topic=topic.strip(),
        outline=outline,
        evidence=evidence.strip(),
        max_pages=max_pages,
    )

    report = enforce_page_limit(
        report,
        max_pages,
    )

    return report
