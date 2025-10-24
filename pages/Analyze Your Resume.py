import streamlit as st
import os
from langchain_core.messages import HumanMessage, ToolMessage
from agent_core import get_supervisor_executor, create_langfuse_handler, resume_analyzer_agent # Import the agent directly

CATEGORIES = sorted([
    'HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
    'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'BPO',
    'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE',
    'APPAREL', 'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS',
    'BANKING', 'ARTS', 'AVIATION'
])

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

# Initialize
try:
    # Use the direct agent instead of the supervisor graph for simplicity here
    # If you need the full graph, the state management becomes more complex
    agent_to_use = resume_analyzer_agent # Change this to use the direct agent
    langfuse_handler = create_langfuse_handler()

    if "analyzer_session_id" not in st.session_state:
        st.session_state.analyzer_session_id = f"st-analyzer-{os.urandom(4).hex()}"
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# Header
st.title("📄 Resume Analyzer")
st.write("AI-powered resume analysis with personalized feedback from industry experts")

# Feature badges
col1, col2, col3, col4, col5 = st.columns(5)
col1.info("✅ Strength Analysis")
col2.info("🔍 Gap Identification")
col3.info("💡 Actionable Tips")
col4.info("📊 Industry Comparison")
col5.info("🚀 Career Boost")

# How it works
with st.expander("How Does It Work?"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("📤 Step 1: Upload")
        st.write("Paste your resume text into the analyzer")
    with c2:
        st.subheader("🔄 Step 2: Compare")
        st.write("AI compares with industry-standard resumes")
    with c3:
        st.subheader("⬆️ Step 3: Improve")
        st.write("Get personalized actionable feedback")

st.divider()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Your Resume")
    st.caption("Copy and paste your entire resume text below:")

    user_resume_text = st.text_area(
        "Resume Text",
        height=500,
        placeholder="Paste your complete resume here...",
        label_visibility="collapsed"
    )

    # Character count
    if user_resume_text:
        char_count = len(user_resume_text)
        st.caption(f"📊 Character count: {char_count:,}")

        if char_count < 200:
            st.warning("⚠️ Your resume seems quite short. Consider adding more details for better analysis.")
        elif char_count > 5000:
            st.info("ℹ️ Long resume detected. Make sure it's well-structured!")

with col2:
    st.subheader("Target Job Category")
    st.caption("Select the field you're targeting:")

    # Search filter
    search_category = st.text_input(
        "Search categories",
        placeholder="Type to filter...",
        label_visibility="collapsed"
    )

    # Filter categories
    if search_category:
        filtered_categories = [cat for cat in CATEGORIES if search_category.upper() in cat]
    else:
        filtered_categories = CATEGORIES

    # Category selector
    target_category = st.selectbox(
        "Job Category",
        filtered_categories,
        label_visibility="collapsed"
    )

    # Selected category display
    st.success(f"**Selected:** {target_category}")

    st.divider()

    # Pro tips
    with st.container():
        st.info("**💡 Pro Tips:**")
        st.write("""
        - Use clear, professional language
        - Include quantifiable achievements
        - List relevant technical skills
        - Proofread for typos
        """)

st.divider()

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "✨ Analyze My Resume Now",
        type="primary",
        use_container_width=True
    )


if analyze_button:
    if not user_resume_text.strip():
        st.error("Please paste your resume text above before analyzing.")
    elif not target_category:
        st.error("Please select a target job category.")
    else:
        with st.status("Analyzing your resume...", expanded=True) as status:
            st.write("Preparing your resume...")

            user_message = f"""
            Please analyze my resume for the {target_category} field.

            Here is my resume:
            {user_resume_text}

            I want constructive feedback on:
            1. Strengths in my current resume
            2. Key skills or experiences I'm missing for {target_category}
            3. Specific actionable improvements I can make
            4. How my resume compares to industry standards
            """

            st.write("AI agents are working on your analysis...")

            try:
                result = agent_to_use.invoke(
                    {"messages": [HumanMessage(content=user_message)]},
                    config={
                        "callbacks": [langfuse_handler],
                        "recursion_limit": 50,
                        "configurable": {
                            "session_id": st.session_state.analyzer_session_id,
                            "trace_name": "Resume-Analyzer-Direct"
                        }
                    }
                )

                st.write("✅ Analysis complete!")
                status.update(label="Analysis Complete!", state="complete")

                # Agent Process
                st.subheader("Agent Process Log")
                if result and 'messages' in result:
                    for i, msg in enumerate(result['messages']):
                        if hasattr(msg, 'content') and msg.content:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                st.info(f"**Agent Action:** Called tool `{msg.tool_calls[0]['name']}`")
                            elif isinstance(msg, ToolMessage):
                                st.success(f"**Tool Result:** `{msg.name}`")
                                display_content = msg.content
                                st.text_area("Raw Tool Output:", value=display_content, height=200, key=f"tool_result_{i}")
                            else:
                                st.info(f"**Agent Response:**")
                                st.write(msg.content)

                st.success("Your Personalized Feedback Completed")

                final_message = None
                for msg in reversed(result['messages']):
                    if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls') and not isinstance(msg, ToolMessage):
                        final_message = msg.content
                        break

                if not final_message:
                    final_message = "Error: No final feedback response found in agent output."

                # # Display with streaming effect
                # response_container = st.container()
                # with response_container:
                #     st.markdown(final_message)

                st.divider()

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "💾 Download Feedback",
                        data=final_message,
                        file_name=f"resume_feedback_{target_category}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col2:
                    if st.button("🔄 Analyze Again", use_container_width=True):
                        st.rerun()

                st.divider()
                st.write("### 💬 Was this helpful?")
                col1, col2 = st.columns(2)
                with col1:
                    st.button("👍 Helpful", use_container_width=True)
                with col2:
                    st.button("👎 Not Helpful", use_container_width=True)

            except Exception as e:
                status.update(label="❌ Analysis Failed", state="error")
                st.error(f"An error occurred during analysis: {e}")
                st.info("💡 Try refreshing the page or contact support if the problem persists.")

# Footer
st.divider()
st.caption("🔒 **Your Privacy Matters** - Your resume data is processed securely and never stored permanently.")
st.caption("Need help? Check our FAQ or Contact Support")