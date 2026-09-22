import os
import re
from typing import List

import streamlit as st
from search_tool import DuckDuckGoResearchTool

MODEL_NAME = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None

def build_llm():
    from crewai import LLM
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")
    return LLM(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0.2,
        max_tokens=12000,
    )

def clean_text(text):
    return re.sub(r"\r\n?", "\n", str(text)).strip()

def extract_urls(text):
    urls = re.findall(r"https?://[^\s)\]>]+", text)
    result, seen = [], set()
    for url in urls:
        url = url.rstrip(".,;")
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result

def research_sources(topic, research_rounds):
    tool = DuckDuckGoResearchTool()
    queries = [
        topic,
        f"{topic} overview evidence statistics",
        f"{topic} latest research studies",
        f"{topic} official government reports",
        f"{topic} academic research papers",
        f"{topic} challenges risks limitations",
        f"{topic} future trends outlook",
        f"{topic} case studies examples",
        f"{topic} economic social environmental impact",
        f"{topic} technology implementation best practices",
        f"{topic} history development timeline",
        f"{topic} policy regulation standards",
    ]
    selected = queries[:max(3, min(research_rounds, len(queries)))]
    evidence = []
    for i, query in enumerate(selected, 1):
        st.write(f"🔎 Research round {i}/{len(selected)}: {query}")
        evidence.append(f"### Research Query {i}\n{query}\n\n{tool._run(query)}")
    return "\n\n".join(evidence)

def build_outline(topic, evidence):
    from crewai import Agent, Crew, Process, Task
    agent = Agent(
        role="Senior Research Architect",
        goal="Create a comprehensive logical outline for a very long research report.",
        backstory="You organize complex topics into rigorous chapters, sections, evidence areas, case studies, limitations, and references.",
        llm=build_llm(),
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=f"""Create a detailed research outline for:
{topic}

Available web evidence:
{evidence[:30000]}

Create approximately 20 to 35 major chapters.
Include executive summary, introduction, definitions, history, current state, technologies, evidence, quantitative data, case studies, regional/global perspectives, impacts, risks, competing viewpoints, regulation where relevant, future scenarios, conclusion and references.
Do not invent facts or references. Return a numbered outline.""",
        expected_output="A detailed multi-chapter research outline.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return clean_text(getattr(crew.kickoff(), "raw", ""))

def split_outline(outline):
    lines = [x.strip() for x in outline.splitlines() if x.strip()]
    chapters, current = [], []
    for line in lines:
        if re.match(r"^(?:#{1,4}\s*)?(?:Chapter\s+)?\d+[\.\):\-]\s+", line, re.I):
            if current:
                chapters.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        chapters.append("\n".join(current))
    return chapters[:35] if chapters else []

def write_chapter(topic, chapter_outline, evidence, number, total, words):
    from crewai import Agent, Crew, Process, Task
    agent = Agent(
        role="Senior Research Writer",
        goal="Write detailed evidence-grounded research chapters without fabricating citations.",
        backstory="You write academic-style reports, explain difficult subjects clearly, distinguish evidence from interpretation, and never fabricate facts, statistics, quotations or URLs.",
        llm=build_llm(),
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=f"""Write chapter {number} of {total}.

Topic:
{topic}

Chapter outline:
{chapter_outline}

Research evidence:
{evidence[:35000]}

Target length: approximately {words} words.

Rules:
- Use supplied evidence and established knowledge.
- Never invent statistics, quotations, study titles, organizations, URLs or citations.
- Say when evidence is insufficient.
- Avoid repetition.
- Use Markdown headings and subheadings.
- Use tables/bullets when useful.
- Explain concepts deeply.
- Distinguish documented facts, analysis and uncertainty.
- Do not add fake footnotes or a references section.""",
        expected_output=f"A detailed Markdown chapter of approximately {words} words.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return clean_text(getattr(crew.kickoff(), "raw", ""))

def generate_long_research_report(topic, target_pages, research_rounds=6):
    # Planning estimate only: actual pages depend on formatting.
    total_words = target_pages * 500

    st.write("🧭 Building research evidence...")
    evidence = research_sources(topic, research_rounds)

    st.write("🗂️ Building report outline...")
    outline = build_outline(topic, evidence)
    chapters = split_outline(outline)

    if len(chapters) < 10:
        chapters = [f"Chapter {i}: Extended Analysis of {topic}" for i in range(1, 21)]

    budget = max(900, total_words // len(chapters))

    parts = [
        f"# Comprehensive Research Report: {topic}",
        "",
        f"*Target planning length: approximately {target_pages} pages / {total_words:,} words.*",
        "",
        "## Important Note on Page Count",
        "Page count depends on formatting. This application plans around 500 words per page; the final printed page count varies with font, margins, headings, spacing, tables and export format.",
        "",
        "## Research Method",
        "The report uses multiple web-search rounds, structured outlining and chapter-by-chapter synthesis. Search coverage varies by topic.",
        "",
        "## Table of Contents",
        outline,
        "",
    ]

    for i, chapter in enumerate(chapters, 1):
        st.write(f"✍️ Writing chapter {i}/{len(chapters)} — target ≈ {budget:,} words")
        parts.append(write_chapter(topic, chapter, evidence, i, len(chapters), budget))
        parts.append("")

    urls = extract_urls(evidence)
    parts += ["## References", ""]
    parts += [f"{i}. {url}" for i, url in enumerate(urls, 1)]

    report = "\n\n".join(parts)
    actual_words = len(report.split())
    report += (
        "\n\n---\n\n## Generation Statistics\n\n"
        f"- Requested planning length: {target_pages:,} pages\n"
        f"- Planning estimate: {total_words:,} words\n"
        f"- Actual generated length: {actual_words:,} words\n"
        f"- Estimated pages from actual words: {max(1, round(actual_words / 500)):,}\n"
    )
    return report
