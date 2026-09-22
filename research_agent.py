```python
import os

from crewai import Agent, Task, Crew, LLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ============================================================
# API KEY
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Add GROQ_API_KEY to your environment variables "
        "or Streamlit Secrets."
    )


# ============================================================
# LLM
# ============================================================

llm = LLM(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    temperature=0.2,
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    """
    Clean model output and remove null characters.
    """
    if not text:
        return ""

    return str(text).replace("\x00", "").strip()


def estimate_words_for_pages(max_pages):
    """
    Approximate 500 words per page.
    """
    return int(max_pages) * 500


# ============================================================
# CREWAI RESULT HELPER
# ============================================================

def get_result_text(result):
    """
    Safely extract text from a CrewAI kickoff result.
    """
    if result is None:
        return ""

    raw = getattr(result, "raw", None)

    if raw:
        return clean_text(raw)

    return clean_text(str(result))


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
10. Include opportunities where relevant.
11. Include future outlook.
12. Include evidence-based conclusions.
13. Avoid unnecessary repetition.
14. Allocate enough sections to support approximately
    {max_pages} pages.
15. The final report should remain within the requested
    approximate page limit.

Important:

- Do not invent research evidence.
- Do not invent statistics.
- Do not invent citations.
- Use only the supplied evidence where factual claims
  require supporting evidence.
- Clearly identify areas where evidence is unavailable.

Return ONLY the outline.
""",

        expected_output=(
            "A detailed structured research outline "
            "with sections and subsections."
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

    return get_result_text(result)


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
6. Market analysis where relevant
7. Technical analysis where relevant
8. Business/economic analysis where relevant
9. Key trends
10. Risks and challenges
11. Opportunities
12. Future outlook
13. Key findings
14. Conclusion
15. References/sources where available

============================================================
WRITING RULES
============================================================

- Use Markdown headings.
- Use tables where useful.
- Do not invent sources.
- Do not fabricate statistics.
- Do not fabricate citations.
- Clearly identify uncertainty.
- Do not repeat information simply to increase length.
- Prioritize useful information over filler.
- Follow the research outline.
- Use the supplied evidence as the factual foundation.
- If evidence is insufficient for a claim, say so.
- Do not present unsupported information as fact.
- Keep the report approximately within the requested
  {max_pages}-page limit.

Return the complete research report.
""",

        expected_output=(
            "A complete professional research report "
            "within the requested length."
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

    return get_result_text(result)


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
    # Validate topic
    # --------------------------------------------------------

    if not topic or not str(topic).strip():
        raise ValueError(
            "Research topic cannot be empty."
        )

    topic = str(topic).strip()

    # --------------------------------------------------------
    # Validate page count
    # --------------------------------------------------------

    try:
        max_pages = int(max_pages)
    except (TypeError, ValueError):
        max_pages = 25

    # Keep requested pages between 5 and 500
    max_pages = max(
        5,
        min(max_pages, 500),
    )

    # --------------------------------------------------------
    # Validate evidence
    # --------------------------------------------------------

    if evidence is None:
        evidence = ""

    evidence = clean_text(evidence)

    # --------------------------------------------------------
    # STEP 1 — CREATE OUTLINE
    # --------------------------------------------------------

    outline = build_outline(
        topic=topic,
        evidence=evidence,
        max_pages=max_pages,
    )

    if not outline:
        raise RuntimeError(
            "The AI failed to generate the research outline."
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

    if not report:
        raise RuntimeError(
            "The AI failed to generate the research report."
        )

    # --------------------------------------------------------
    # STEP 3 — ENFORCE PAGE LIMIT
    # --------------------------------------------------------

    report = enforce_page_limit(
        report=report,
        max_pages=max_pages,
    )

    return report
```

### What I fixed

1. **Fixed the invalid `MODEL_NAME` syntax**

   ```python
   MODEL_NAME = "openai/gpt-oss-120b"
   ```

2. **Fixed the LLM configuration**

   ```python
   llm = LLM(
       model=MODEL_NAME,
       api_key=GROQ_API_KEY,
       base_url="https://api.groq.com/openai/v1",
       temperature=0.2,
   )
   ```

3. **Added safe API-key handling**

   ```python
   GROQ_API_KEY = os.getenv("GROQ_API_KEY")
   ```

4. Added `get_result_text()` so CrewAI results are handled more safely.

5. Added validation for an empty research topic.

6. Added validation for invalid page numbers.

7. Preserved your **5–500 page range**.

8. Preserved the **LLM-controlled outline → report workflow**.

9. Added stronger instructions not to fabricate sources or statistics.

10. Removed unnecessary comments that could make the file harder to maintain.

### Important: 500 pages

Your code can **request** up to 500 pages, but a single LLM generation should not be expected to reliably produce a 250,000-word report in one CrewAI task. Groq currently documents GPT-OSS 120B with a 131,072-token context window and a maximum completion of 65,536 tokens.

For your research-agent application, the better architecture is:

**Topic → LLM decides report size → Outline → Section 1 → Section 2 → Section 3 → ... → Combine → Final report**

That will also prevent the current `enforce_page_limit()` function from simply cutting a report in the middle of a sentence when the model generates too much.

Also, the current Groq model ID you selected, `openai/gpt-oss-120b`, is valid on Groq.

If you are using this inside your **Streamlit AI Research Agent**, this corrected file can replace your current `research_agent.py`; your `app.py` can continue calling:

```python
report = generate_long_research_report(
    topic=topic,
    evidence=evidence,
    max_pages=max_pages,
)
```

One additional point: if your next error is related to **CrewAI/LiteLLM compatibility**, that will be a dependency-version issue rather than a Python syntax issue. In that case, the `requirements.txt` should be pinned to compatible versions rather than leaving CrewAI completely unpinned.
