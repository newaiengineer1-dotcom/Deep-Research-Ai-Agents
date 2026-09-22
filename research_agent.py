import os

from crewai import Agent, Task, Crew, LLM


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Add GROQ_API_KEY to Streamlit Secrets."
    )


# ============================================================
# IMPORTANT
# ============================================================
# DO NOT CHANGE THIS TO:
# gpt-oss-120b
#
# The correct Groq model ID is:
# openai/gpt-oss-120b
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ============================================================
# CREATE LLM
# ============================================================

llm = LLM(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    temperature=0.2,
)


# ============================================================
# TEXT CLEANER
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    return str(text).replace("\x00", "").strip()


# ============================================================
# PAGE VALIDATION
# ============================================================

def validate_page_limit(max_pages):

    try:
        max_pages = int(max_pages)

    except (TypeError, ValueError):

        max_pages = 10

    # User is allowed to select 5–500 pages
    max_pages = max(5, min(max_pages, 500))

    return max_pages


# ============================================================
# WORD ESTIMATION
# ============================================================

def calculate_target_words(max_pages):

    # Approximate 500 words per page
    return max_pages * 500


# ============================================================
# BUILD OUTLINE
# ============================================================

def build_outline(
    topic,
    evidence,
    max_pages,
):

    target_words = calculate_target_words(max_pages)

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    agent = Agent(

        role="Senior Research Strategist",

        goal=(
            "Create a comprehensive and logically structured "
            "research outline."
        ),

        backstory=(
            "You are an experienced research strategist "
            "specializing in renewable energy, solar PV, BESS, "
            "technology, business and market research."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )

    # --------------------------------------------------------
    # TASK
    # --------------------------------------------------------

    task = Task(

        description=f"""

Create a detailed professional research outline.

============================================================
RESEARCH TOPIC
============================================================

{topic}


============================================================
AVAILABLE RESEARCH EVIDENCE
============================================================

{evidence if evidence else "No additional evidence supplied."}


============================================================
REQUESTED REPORT LENGTH
============================================================

Maximum pages:
{max_pages}

Approximate words:
{target_words}


============================================================
REQUIRED STRUCTURE
============================================================

Create a logical structure containing:

1. Executive Summary
2. Introduction
3. Background and Context
4. Market Overview
5. Technology Analysis
6. Technical Analysis
7. Business / Economic Analysis
8. Market Trends
9. Opportunities
10. Challenges and Risks
11. Regional / Global Analysis
12. Future Outlook
13. Key Findings
14. Conclusion
15. References


Only include sections that are relevant to the research topic.


============================================================
QUALITY REQUIREMENTS
============================================================

- Use a professional research structure.
- Create useful subsections.
- Avoid unnecessary repetition.
- Do not create filler sections.
- Allocate enough sections for approximately
  {max_pages} pages.
- Target approximately {target_words} words.
- Do not intentionally exceed the requested length.

Return ONLY the outline.

""",

        expected_output=(
            "A detailed professional research outline "
            "with sections and subsections."
        ),

        # IMPORTANT CREWAI REQUIREMENT
        agent=agent,
    )

    # --------------------------------------------------------
    # CREW
    # --------------------------------------------------------

    crew = Crew(

        agents=[agent],

        tasks=[task],

        process="sequential",

        verbose=False,
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    result = crew.kickoff()

    return clean_text(
        getattr(
            result,
            "raw",
            str(result),
        )
    )


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    topic,
    outline,
    evidence,
    max_pages,
):

    target_words = calculate_target_words(max_pages)

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    agent = Agent(

        role="Senior Research Report Writer",

        goal=(
            "Produce a comprehensive, factual and professional "
            "research report."
        ),

        backstory=(
            "You are an expert long-form research writer "
            "specializing in renewable energy, solar PV, BESS, "
            "technology, markets and business research."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )

    # --------------------------------------------------------
    # TASK
    # --------------------------------------------------------

    task = Task(

        description=f"""

Write the final professional research report.

============================================================
RESEARCH TOPIC
============================================================

{topic}


============================================================
RESEARCH OUTLINE
============================================================

{outline}


============================================================
RESEARCH EVIDENCE
============================================================

{evidence if evidence else "No additional evidence supplied."}


============================================================
REPORT LENGTH
============================================================

Maximum requested pages:

{max_pages}

Approximate target words:

{target_words}

Use approximately 500 words per page.


============================================================
REPORT CONTENT
============================================================

Include relevant sections such as:

# Title

## Executive Summary

## Introduction

## Background and Context

## Market Overview

## Technology Analysis

## Technical Analysis

## Business / Economic Analysis

## Key Trends

## Opportunities

## Risks and Challenges

## Regional / Global Analysis

## Future Outlook

## Key Findings

## Conclusion

## References


============================================================
QUALITY RULES
============================================================

1. Use professional language.

2. Use Markdown headings.

3. Use tables where useful.

4. Do not fabricate statistics.

5. Do not fabricate sources.

6. Do not fabricate citations.

7. Clearly identify estimates.

8. Clearly identify uncertainty.

9. Avoid unnecessary repetition.

10. Do not add filler simply to increase length.

11. Keep the report approximately within
    the selected page limit.

12. Prioritize useful research content.


============================================================
FINAL LENGTH
============================================================

The user selected:

{max_pages} pages

Target:

{target_words} words.

Do not intentionally exceed the requested page limit.

Return the complete report.

""",

        expected_output=(
            "A complete professional research report "
            "approximately matching the requested length."
        ),

        # IMPORTANT CREWAI REQUIREMENT
        agent=agent,
    )

    # --------------------------------------------------------
    # CREW
    # --------------------------------------------------------

    crew = Crew(

        agents=[agent],

        tasks=[task],

        process="sequential",

        verbose=False,
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    result = crew.kickoff()

    return clean_text(
        getattr(
            result,
            "raw",
            str(result),
        )
    )


# ============================================================
# PAGE LIMIT PROTECTION
# ============================================================

def enforce_page_limit(
    report,
    max_pages,
):

    max_words = calculate_target_words(max_pages)

    words = report.split()

    if len(words) <= max_words:

        return report

    limited_report = " ".join(
        words[:max_words]
    )

    limited_report += (

        "\n\n---\n\n"

        f"**Report Length Notice:** "
        f"The generated report was limited to approximately "
        f"{max_pages} pages."
    )

    return limited_report


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_long_research_report(
    topic,
    evidence="",
    max_pages=10,
):

    # --------------------------------------------------------
    # Validate pages
    # --------------------------------------------------------

    max_pages = validate_page_limit(
        max_pages
    )

    # --------------------------------------------------------
    # Validate topic
    # --------------------------------------------------------

    if not topic or not topic.strip():

        raise ValueError(
            "Research topic cannot be empty."
        )

    # --------------------------------------------------------
    # Build outline
    # --------------------------------------------------------

    outline = build_outline(

        topic=topic.strip(),

        evidence=evidence.strip()
        if evidence
        else "",

        max_pages=max_pages,
    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    report = generate_report(

        topic=topic.strip(),

        outline=outline,

        evidence=evidence.strip()
        if evidence
        else "",

        max_pages=max_pages,
    )

    # --------------------------------------------------------
    # Apply page limit
    # --------------------------------------------------------

    report = enforce_page_limit(

        report=report,

        max_pages=max_pages,
    )

    return report
