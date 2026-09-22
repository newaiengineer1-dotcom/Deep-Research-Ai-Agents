import os
import re
from crewai import Agent, Task, Crew, LLM


# =========================================================
# CONFIGURATION
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured."
    )


# Correct Groq model ID
MODEL_NAME = "openai/gpt-oss-120b"


# =========================================================
# LLM
# =========================================================

llm = LLM(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2,
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    if not text:
        return ""

    text = str(text)

    text = text.replace("\x00", "")

    return text.strip()


# =========================================================
# PAGE ESTIMATION
# =========================================================

def estimate_pages(text):
    """
    Approximate page count.

    Assumption:
    ~500 words per page.

    This is an estimate because actual page count
    depends on font, margins, spacing, tables, etc.
    """

    words = len(text.split())

    return max(1, round(words / 500))


def estimate_words_for_pages(max_pages):
    """
    Convert requested pages into approximate word count.
    """

    return max_pages * 500


# =========================================================
# BUILD RESEARCH OUTLINE
# =========================================================

def build_outline(topic, evidence, max_pages):

    target_words = estimate_words_for_pages(max_pages)

    agent = Agent(
        role="Senior Research Strategist",
        goal=(
            "Create a comprehensive and logically structured "
            "research report outline."
        ),
        backstory=(
            "You are an experienced research strategist specializing "
            "in technical, business, energy and technology research."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
Create a detailed research outline for:

RESEARCH TOPIC:
{topic}

AVAILABLE RESEARCH EVIDENCE:
{evidence}

TARGET REPORT SIZE:
Approximately {max_pages} pages.

TARGET WORD COUNT:
Approximately {target_words} words.

Requirements:

1. Create a logical professional structure.
2. Include an introduction.
3. Include major research sections.
4. Include subsections where appropriate.
5. Include market/technical/business analysis where relevant.
6. Include evidence-based conclusions.
7. Avoid unnecessary repetition.
8. Allocate enough sections to support approximately
   {max_pages} pages.
9. The final report must not intentionally exceed the
   requested page limit.

Return ONLY the outline.
""",
        expected_output="A detailed structured research outline.",
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()

    return clean_text(
        getattr(result, "raw", str(result))
    )


# =========================================================
# GENERATE FINAL REPORT
# =========================================================

def generate_report(topic, outline, evidence, max_pages):

    target_words = estimate_words_for_pages(max_pages)

    agent = Agent(
        role="Senior Research Report Writer",
        goal=(
            "Produce a comprehensive, factual, well-structured "
            "professional research report."
        ),
        backstory=(
            "You are an expert long-form research writer who "
            "produces evidence-based technical and business reports."
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

RESEARCH OUTLINE:
{outline}

RESEARCH EVIDENCE:
{evidence}

REPORT LENGTH:

Maximum pages requested:
{max_pages}

Approximate maximum words:
{target_words}

IMPORTANT LENGTH RULE:

Keep the report approximately within the requested
{max_pages}-page limit.

Use approximately 500 words per page.

Therefore:

5 pages   ≈ 2,500 words
10 pages  ≈ 5,000 words
25 pages  ≈ 12,500 words
50 pages  ≈ 25,000 words
100 pages ≈ 50,000 words
250 pages ≈ 125,000 words
500 pages ≈ 250,000 words

REPORT REQUIREMENTS:

- Professional title
- Executive Summary
- Introduction
- Main research sections
- Subsections
- Technical/business analysis where relevant
- Tables where useful
- Key findings
- Risks and limitations
- Conclusion
- References/sources based only on available evidence

Do not invent sources.

Do not fabricate statistics.

Clearly identify uncertainty.

Do not repeat information merely to increase length.

Prioritize useful information over filler.

Use Markdown headings.

The final report should be approximately
{max_pages} pages or less.
""",
        expected_output=(
            "A complete professional research report "
            "within the requested length."
        ),
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()

    return clean_text(
        getattr(result, "raw", str(result))
    )


# =========================================================
# LIMIT REPORT BY WORD COUNT
# =========================================================

def enforce_page_limit(report, max_pages):

    max_words = estimate_words_for_pages(max_pages)

    words = report.split()

    if len(words) <= max_words:
        return report

    truncated_words = words[:max_words]

    truncated_report = " ".join(truncated_words)

    truncated_report += (
        "\n\n---\n\n"
        "*Report truncated to the selected maximum "
        f"length of approximately {max_pages} pages.*"
    )

    return truncated_report


# =========================================================
# MAIN FUNCTION
# =========================================================

def generate_long_research_report(
    topic,
    evidence=None,
    max_pages=25,
):

    # Safety limits
    max_pages = max(5, min(int(max_pages), 500))

    if evidence is None:
        evidence = ""

    # -----------------------------------------------------
    # Build outline
    # -----------------------------------------------------

    outline = build_outline(
        topic=topic,
        evidence=evidence,
        max_pages=max_pages,
    )

    # -----------------------------------------------------
    # Generate report
    # -----------------------------------------------------

    report = generate_report(
        topic=topic,
        outline=outline,
        evidence=evidence,
        max_pages=max_pages,
    )

    # -----------------------------------------------------
    # Enforce requested page limit
    # -----------------------------------------------------

    report = enforce_page_limit(
        report=report,
        max_pages=max_pages,
    )

    return report
