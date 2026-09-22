import streamlit as st

from research_agent import generate_long_research_report


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Deep Research AI Agent",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 25px;
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
    'Generate AI-powered professional research reports using Google Gemini.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Research Settings")

    max_pages = st.slider(
        "📄 Maximum Report Pages",
        min_value=5,
        max_value=500,
        value=10,
        step=5,
    )

    st.info(
        f"""
**Selected:** {max_pages} pages

**Approximate target:**
{max_pages * 500:,} words
"""
    )

    st.markdown("---")

    st.caption("🤖 LLM")
    st.write("Google Gemini")

    st.caption("📄 Report range")
    st.write("5 – 500 pages")


# ============================================================
# INPUT
# ============================================================

topic = st.text_area(
    "🔬 Research Topic",
    placeholder=(
        "Example:\n"
        "Global Solar PV and BESS Outlook 2026"
    ),
    height=120,
)


evidence = st.text_area(
    "📚 Additional Research Evidence (Optional)",
    placeholder=(
        "Paste additional research information, "
        "documents, notes, statistics or references here..."
    ),
    height=180,
)


# ============================================================
# GENERATE
# ============================================================

if st.button(
    "🚀 Generate Research Report",
    type="primary",
    use_container_width=True,
):

    if not topic.strip():

        st.warning(
            "⚠️ Please enter a research topic."
        )

        st.stop()

    with st.spinner(
        f"Generating approximately {max_pages} pages..."
    ):

        try:

            report = generate_long_research_report(
                topic=topic.strip(),
                evidence=evidence.strip(),
                max_pages=max_pages,
            )

        except Exception as e:

            st.error(
                f"❌ Research generation failed: {e}"
            )

            st.stop()

    # ========================================================
    # RESULT
    # ========================================================

    st.success(
        f"✅ Research report generated — "
        f"maximum target: {max_pages} pages"
    )

    st.markdown("---")

    st.markdown(report)

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.download_button(
        "📥 Download Research Report",
        data=report,
        file_name="deep_research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
