import os
from crewai import Agent, Task, Crew, LLM


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# ============================================================
# LLM
# ============================================================

llm = LLM(
    model=MODEL_NAME,="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2,
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    if not text:
        return ""

    return str(text).replace("\x00", "").strip()


def estimate_words_for_pages(max_pages):
    """
    Approximate 500 words per page.
    """

    return int(max_pages) * 500


# ============================================================
# BUILD OUTLINE
# ============================================================

def build_outline(topic, evidence, max_pages):

    target_words = estimate_words_for_pages(max_pages)

    agent = Agent(
        role="Senior Research Strategist",

        goal=(
            "Create a comprehensive, logical and professional "
            "research outline."
        ),

        backstory=(
            "You are an experienced research strategist specializing "
            "in renewable energy, technology, business and market research."
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
2. Include an Executive Summary.
3. Include an Introduction.
4. Include major research sections.
5. Include subsections where appropriate.
6. Include market analysis where relevant.
7. Include technical analysis where relevant.
8. Include business and economic analysis where relevant.
9. Include risks and challenges.
10. Include future outlook.
11. Include evidence-based conclusions.
12. Avoid unnecessary repetition.
13. Allocate enough sections to support approximately
    {max_pages} pages.
14. The final report must not intentionally exceed
    the requested page limit.

Return ONLY the outline.
""",

        expected_output=(
            "A detailed structured research outline "
            "with sections and subsections."
        ),

        # ====================================================
        # IMPORTANT FIX
        # ====================================================
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
# GENERATE FINAL REPORT
# ============================================================

def generate_report(
    topic,
    outline,
    evidence,
    max_pages,
):

    target_words = estimate_words_for_pages(max_pages)

    agent = Agent(

        role="Senior Research Report Writer",

        goal=(
            "Produce a comprehensive, factual and "
            "professional research report."
        ),

        backstory=(
            "You are an expert long-form research writer "
            "specializing in renewable energy, solar PV, "
            "BESS, technology and business research."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )

    task = Task(

        description=f"""
Write the final research report.

RESEARCH TOPIC:
{topic}

RESEARCH OUTLINE:
{outline}

RESEARCH EVIDENCE:
{evidence}

============================================================
REPORT LENGTH
============================================================

Maximum requested pages:
{max_pages}

Approximate target words:
{target_words}

Use approximately 500 words per page.

Examples:

5 pages   ≈ 2,500 words
10 pages  ≈ 5,000 words
25 pages  ≈ 12,500 words
50 pages  ≈ 25,000 words
100 pages ≈ 50,000 words
250 pages ≈ 125,000 words
500 pages ≈ 250,000 words

============================================================
REPORT REQUIREMENTS
============================================================

Include:

1. Title
2. Executive Summary
3. Introduction
4. Main research sections
5. Subsections
6. Market analysis
7. Technical analysis where relevant
8. Business/economic analysis where relevant
9. Key trends
10. Risks and challenges
11. Opportunities
12. Future outlook
13. Key findings
14. Conclusion
15. References/sources where available

Rules:

- Use Markdown headings.
- Use tables where useful.
- Do not invent sources.
- Do not fabricate statistics.
- Do not fabricate citations.
- Clearly identify uncertainty.
- Do not repeat information simply to increase length.
- Prioritize useful information over filler.
- Keep the report approximately within the requested
  {max_pages}-page limit.

Return the complete research report.
""",

        expected_output=(
            "A complete professional research report "
            "within the requested length."
        ),

        # ====================================================
        # IMPORTANT FIX
        # ====================================================
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
# PAGE LIMIT ENFORCEMENT
# ============================================================

def enforce_page_limit(report, max_pages):

    max_words = estimate_words_for_pages(max_pages)

    words = report.split()

    if len(words) <= max_words:
        return report

    truncated = " ".join(words[:max_words])

    truncated += (
        "\n\n---\n\n"
        f"*Report limited to approximately {max_pages} pages.*"
    )

    return truncated


# ============================================================
# MAIN RESEARCH FUNCTION
# ============================================================

def generate_long_research_report(
    topic,
    evidence=None,
    max_pages=25,
):

    # --------------------------------------------------------
    # Safety validation
    # --------------------------------------------------------

    max_pages = max(
        5,
        min(
            int(max_pages),
            500,
        ),
    )

    if evidence is None:
        evidence = ""

    # --------------------------------------------------------
    # STEP 1 — CREATE OUTLINE
    # --------------------------------------------------------

    outline = build_outline(
        topic=topic,
        evidence=evidence,
        max_pages=max_pages,
    )

    # --------------------------------------------------------
    # STEP 2 — WRITE REPORT
    # --------------------------------------------------------

    report = generate_report(
        topic=topic,
        outline=outline,
        evidence=evidence,
        max_pages=max_pages,
    )

    # --------------------------------------------------------
    # STEP 3 — ENFORCE PAGE LIMIT
    # --------------------------------------------------------

    report = enforce_page_limit(
        report=report,
        max_pages=max_pages,
    )

    return report
