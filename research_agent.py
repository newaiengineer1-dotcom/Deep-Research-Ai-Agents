import os

from crewai import Agent, Task, Crew, LLM


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Add GROQ_API_KEY to Streamlit Secrets."
    )


# ============================================================
# IMPORTANT
# ============================================================
# Current Groq model ID:
#
# openai/gpt-oss-120b
#
# DO NOT use:
#
# gpt-oss-120b
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# CREWAI LLM
# ============================================================

llm = LLM(
    model=MODEL_NAME,
    provider="groq",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2,
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    return str(text).replace("\x00", "").strip()


# ============================================================
# PAGE LIMIT
# ============================================================

def validate_page_limit(max_pages):

    try:
        max_pages = int(max_pages)
    except (TypeError, ValueError):
        max_pages = 10

    return max(5, min(max_pages, 500))


def target_words(max_pages):
    return max_pages * 500


# ============================================================
# OUTLINE
# ============================================================

def build_outline(topic, evidence, max_pages):

    words = target_words(max_pages)

    agent = Agent(
        role="Senior Research Strategist",
        goal=(
            "Create a detailed and logically structured "
            "research outline."
        ),
        backstory=(
            "Experienced research strategist specializing "
            "in renewable energy, solar PV, BESS, technology "
            "and business research."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
Create a detailed professional research outline.

RESEARCH TOPIC:
{topic}

RESEARCH EVIDENCE:
{evidence if evidence else "No additional evidence provided."}

REQUESTED REPORT SIZE:
{max_pages} pages

APPROXIMATE WORD COUNT:
{words} words

Create a logical structure containing relevant sections such as:

1. Executive Summary
2. Introduction
3. Background
4. Market Overview
5. Technology Analysis
6. Technical Analysis
7. Economic / Business Analysis
8. Trends
9. Opportunities
10. Risks and Challenges
11. Regional Analysis
12. Future Outlook
13. Key Findings
14. Conclusion
15. References

Avoid repetition and filler.

Return ONLY the outline.
""",
        expected_output="A detailed professional research outline.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process="sequential",
        verbose=False,
    )

    result = crew.kickoff()

    return clean_text(
        getattr(result, "raw", str(result))
    )


# ============================================================
# FINAL REPORT
# ============================================================

def generate_report(
    topic,
    outline,
    evidence,
    max_pages,
):

    words = target_words(max_pages)

    agent = Agent(
        role="Senior Research Report Writer",
        goal=(
            "Produce a comprehensive factual professional "
            "research report."
        ),
        backstory=(
            "Expert long-form research writer specializing "
            "in renewable energy, solar PV, BESS, markets "
            "and technical research."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
Write the final research report.

TOPIC:
{topic}

OUTLINE:
{outline}

RESEARCH EVIDENCE:
{evidence if evidence else "No additional evidence provided."}

============================================================
REPORT LENGTH
============================================================

Maximum pages:
{max_pages}

Target words:
{words}

Approximately 500 words = 1 page.

============================================================
REPORT REQUIREMENTS
============================================================

Include relevant:

- Title
- Executive Summary
- Introduction
- Background
- Market Analysis
- Technical Analysis
- Technology Analysis
- Business/Economic Analysis
- Trends
- Opportunities
- Risks
- Regional Analysis
- Future Outlook
- Key Findings
- Conclusion
- References

Rules:

- Use Markdown.
- Use professional language.
- Use tables where useful.
- Do not fabricate statistics.
- Do not fabricate sources.
- Do not fabricate citations.
- Identify estimates and uncertainty.
- Avoid unnecessary repetition.
- Do not add filler.

The final report should approximately match
the user's selected {max_pages}-page limit.

Return ONLY the complete report.
""",
        expected_output=(
            "A complete professional research report "
            "approximately matching the requested length."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process="sequential",
        verbose=False,
    )

    result = crew.kickoff()

    return clean_text(
        getattr(result, "raw", str(result))
    )


# ============================================================
# PAGE LIMIT PROTECTION
# ============================================================

def enforce_page_limit(report, max_pages):

    maximum_words = target_words(max_pages)

    words = report.split()

    if len(words) <= maximum_words:
        return report

    limited = " ".join(
        words[:maximum_words]
    )

    return (
        limited
        + "\n\n---\n\n"
        + f"Report limited to approximately {max_pages} pages."
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_long_research_report(
    topic,
    evidence="",
    max_pages=10,
):

    max_pages = validate_page_limit(max_pages)

    if not topic or not topic.strip():
        raise ValueError(
            "Research topic cannot be empty."
        )

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

    return enforce_page_limit(
        report,
        max_pages,
    )
