import streamlit as st

from research_agent import generate_long_research_report


st.set_page_config(
    page_title="Deep Research AI Agent",
    page_icon="🔎",
    layout="wide",
)


st.title("🔎 Deep Research AI Agent")

st.caption(
    "Generate AI-powered long-form research reports."
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


# ============================================================
# INPUT
# ============================================================

topic = st.text_area(
    "🔬 Research Topic",
    placeholder="Example: Global Solar PV and BESS Outlook 2026",
    height=120,
)


evidence = st.text_area(
    "📚 Research Evidence (Optional)",
    placeholder="Paste additional research evidence here...",
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
            "Please enter a research topic."
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


    st.success(
        f"✅ Report generated — target: {max_pages} pages"
    )

    st.markdown("---")

    st.markdown(report)


    st.download_button(

        "📥 Download Research Report",

        data=report,

        file_name="deep_research_report.md",

        mime="text/markdown",

        use_container_width=True,
    )
