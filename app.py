import streamlit as st
from research_agent import generate_long_research_report

st.set_page_config(
    page_title="Deep Research AI Agent",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Deep Research AI Agent")
st.caption("AI-powered long-form research report generator")

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Research Settings")

    max_pages = st.slider(
        "Maximum Report Pages",
        min_value=5,
        max_value=500,
        value=25,
        step=5,
        help="Select the maximum number of pages for the generated research report.",
    )

    st.info(
        f"📄 Maximum report length: **{max_pages} pages**"
    )

# ---------------------------------------------------------
# Main input
# ---------------------------------------------------------

topic = st.text_area(
    "Research Topic",
    placeholder="Example: Global Solar PV and Battery Energy Storage Market Outlook",
    height=120,
)

if st.button("🚀 Generate Research Report", type="primary"):

    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    with st.spinner(
        f"Researching and generating up to {max_pages} pages..."
    ):
        try:
            report = generate_long_research_report(
                topic=topic.strip(),
                max_pages=max_pages,
            )

            st.success("Research report generated successfully.")

            st.markdown("---")
            st.markdown(report)

            # Download
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name="deep_research_report.md",
                mime="text/markdown",
            )

        except Exception as e:
            st.error(f"Research generation failed: {e}")
