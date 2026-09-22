import streamlit as st
from research_agent import generate_long_research_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deep Research AI Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .page-info {
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 10px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔎 Deep Research AI Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Generate structured, long-form AI research reports with CrewAI + Groq.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Research Settings")

    st.markdown("### 📄 Report Length")

    max_pages = st.slider(
        "Select maximum number of pages",
        min_value=5,
        max_value=500,
        value=10,
        step=5,
        help=(
            "Select the approximate maximum length of the "
            "final research report."
        ),
    )

    st.markdown(
        f"""
        <div class="page-info">
        <b>Selected Report Length</b><br>
        📄 {max_pages} pages<br>
        📝 Approximately {max_pages * 500:,} words
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Page estimation is based on approximately 500 words per page."
    )

    st.markdown("---")

    st.markdown("### 🤖 AI Model")

    st.code(
        "openai/gpt-oss-120b",
        language="text",
    )

    st.markdown("---")

    st.markdown("### 📊 Page Examples")

    st.write("5 pages → ~2,500 words")
    st.write("10 pages → ~5,000 words")
    st.write("25 pages → ~12,500 words")
    st.write("50 pages → ~25,000 words")
    st.write("100 pages → ~50,000 words")
    st.write("250 pages → ~125,000 words")
    st.write("500 pages → ~250,000 words")


# ============================================================
# RESEARCH INPUT
# ============================================================

st.subheader("🔬 Research Topic")

topic = st.text_area(
    "Enter your research topic",
    placeholder=(
        "Example:\n"
        "Global Solar PV and BESS Outlook 2026"
    ),
    height=120,
)


# ============================================================
# OPTIONAL RESEARCH EVIDENCE
# ============================================================

st.subheader("📚 Research Evidence")

evidence = st.text_area(
    "Optional: paste research evidence, notes, sources or documents",
    placeholder=(
        "Paste research material here if available. "
        "Leave blank if your research workflow collects evidence separately."
    ),
    height=180,
)


# ============================================================
# REPORT SUMMARY
# ============================================================

if topic.strip():

    st.info(
        f"""
        **Research configuration**

        Topic: **{topic.strip()}**

        Maximum report length: **{max_pages} pages**

        Approximate target: **{max_pages * 500:,} words**
        """
    )


# ============================================================
# GENERATE BUTTON
# ============================================================

generate = st.button(
    "🚀 Generate Research Report",
    type="primary",
    use_container_width=True,
)


# ============================================================
# GENERATE REPORT
# ============================================================

if generate:

    if not topic.strip():

        st.warning(
            "⚠️ Please enter a research topic before generating the report."
        )

        st.stop()

    progress_text = (
        f"Generating research report up to approximately "
        f"{max_pages} pages..."
    )

    with st.spinner(progress_text):

        try:

            report = generate_long_research_report(
                topic=topic.strip(),
                evidence=evidence.strip(),
                max_pages=max_pages,
            )

        except Exception as e:

            st.error(
                f"❌ Research generation failed: {str(e)}"
            )

            st.stop()


    # ========================================================
    # SUCCESS
    # ========================================================

    st.success(
        f"✅ Research report generated successfully "
        f"for the selected {max_pages}-page limit."
    )


    # ========================================================
    # REPORT
    # ========================================================

    st.markdown("---")

    st.subheader("📑 Research Report")

    st.markdown(report)


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.download_button(
        label="📥 Download Markdown Report",
        data=report,
        file_name="deep_research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
