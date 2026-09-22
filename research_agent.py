import os

from crewai import Agent, Task, Crew, LLM


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Please add GROQ_API_KEY to Streamlit Secrets."
    )


# ============================================================
# IMPORTANT:
# USE THE FULL GROQ MODEL ID
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
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    return str(text).replace("\x00", "").strip()


# ============================================================
# PAGE / WORD CALCULATION
# ============================================================

def calculate_target_words(max_pages):

    """
    Estimate approximately 500 words per page.

    Examples:

    5 pages   = 2,500 words
    10 pages  = 5,000 words
    50 pages  = 25,000 words
    100 pages = 50,000 words
    500 pages = 250,000 words
    """

    return int(max_pages) * 500


# ============================================================
# VALIDATE PAGE LIMIT
# ============================================================

def validate_page_limit(max_pages):

    try:
        max_pages = int(max_pages)

    except (TypeError, ValueError):

        max_pages = 10

    # Force allowed range: 5–500
    max_pages = max(
        5,
        min(
            max_pages,
            500,
        ),
    )

    return max_pages


# ============================================================
# BUILD RESEARCH OUTLINE
# ============================================================

def build_outline(
    topic,
    evidence,
    max_pages,
):

    target_words = calculate_target_words(max_pages)


    # --------------------------------------------------------
    # OUTLINE AGENT
    # --------------------------------------------------------

    agent = Agent(

        role="Senior Research Strategist",

        goal=(
            "Create a detailed, logical and professional "
            "research outline."
        ),

        backstory=(
            "You are a senior research strategist with expertise "
            "in renewable energy, solar PV, BESS, technology, "
            "business and market research."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )


    # --------------------------------------------------------
    # OUTLINE TASK
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

{evidence if evidence else "No additional evidence provided."}


============================================================
REPORT LENGTH
============================================================

Maximum requested pages:

{max_pages}

Approximate target words:

{target_words}


============================================================
OUTLINE REQUIREMENTS
============================================================

Create a professional structure containing:

1. Executive Summary

2. Introduction

3. Background and Context

4. Current Market / Industry Situation

5. Key Technology Developments

6. Technical Analysis where relevant

7. Economic / Business Analysis where relevant

8. Market Trends

9. Opportunities

10. Challenges and Risks

11. Regional / Global Analysis where relevant

12. Future Outlook

13. Key Findings

14. Conclusion

15. References / Sources


The outline must:

- Be logical.
- Avoid unnecessary repetition.
- Have appropriate subsections.
- Allocate sufficient sections for the requested report length.
- Support approximately {max_pages} pages.
- Support approximately {target_words} words.
- Prioritize useful information.
- Not add filler merely to increase page count.

Return ONLY the research outline.
""",

        expected_output=(
            "A detailed professional research outline "
            "containing sections and subsections."
        ),

        # ====================================================
        # CRITICAL CREWAI FIX
        # ====================================================

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
    # EXECUTE
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
# GENERATE FINAL REPORT
# ============================================================

def generate_report(
    topic,
    outline,
    evidence,
    max_pages,
):

    target_words = calculate_target_words(max_pages)


    # --------------------------------------------------------
    # REPORT AGENT
    # --------------------------------------------------------

    agent = Agent(

        role="Senior Research Report Writer",

        goal=(
            "Write a comprehensive, factual and professional "
            "research report using the supplied research evidence."
        ),

        backstory=(
            "You are an experienced long-form research writer "
            "specializing in renewable energy, solar PV, BESS, "
            "technology, markets and business analysis."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )


    # --------------------------------------------------------
    # REPORT TASK
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

{evidence if evidence else "No additional evidence provided."}


============================================================
REPORT LENGTH
============================================================

User-selected maximum:

{max_pages} pages

Approximate target:

{target_words} words


Use approximately 500 words per page.


============================================================
REPORT STRUCTURE
============================================================

Include:

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


Only include sections that are relevant to the topic.


============================================================
QUALITY REQUIREMENTS
============================================================

1. Use professional language.

2. Use Markdown headings.

3. Use tables when useful.

4. Avoid unnecessary repetition.

5. Do not create filler content.

6. Do not fabricate statistics.

7. Do not fabricate sources.

8. Do not fabricate citations.

9. Clearly distinguish facts from estimates.

10. Clearly identify uncertainty.

11. Use the supplied research evidence.

12. Maintain logical flow.

13. Make the report useful for professional research.


============================================================
PAGE LIMIT
============================================================

The user selected:

{max_pages} pages

Target approximately:

{target_words} words.

Keep the report approximately within this limit.

Do NOT intentionally exceed the selected page limit.

Return the complete research report.
""",

        expected_output=(
            "A complete professional research report "
            "approximately matching the requested length."
        ),

        # ====================================================
        # CRITICAL CREWAI FIX
        # ====================================================

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
    # EXECUTE
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
# FINAL PAGE-LIMIT PROTECTION
# ============================================================

def enforce_page_limit(
    report,
    max_pages,
):

    max_words = calculate_target_words(max_pages)

    words = report.split()


    # --------------------------------------------------------
    # Already within limit
    # --------------------------------------------------------

    if len(words) <= max_words:

        return report


    # --------------------------------------------------------
    # Truncate if necessary
    # --------------------------------------------------------

    truncated_words = words[:max_words]

    truncated_report = " ".join(
        truncated_words
    )


    truncated_report += (

        "\n\n---\n\n"

        f"**Report Length Notice:** "
        f"The report was limited to approximately "
        f"{max_pages} pages based on the user-selected "
        f"maximum length."
    )


    return truncated_report


# ============================================================
# MAIN FUNCTION USED BY APP.PY
# ============================================================

def generate_long_research_report(
    topic,
    evidence="",
    max_pages=10,
):

    # --------------------------------------------------------
    # Validate selected pages
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
    # Final page protection
    # --------------------------------------------------------

    report = enforce_page_limit(

        report=report,

        max_pages=max_pages,
    )


    return report
