import streamlit as st
from agent_core import setup_qdrant_client, setup_chat_models, create_langfuse_handler, get_supervisor_executor
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PayloadSchemaType
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
import json
from uuid import uuid4

# page title
st.set_page_config(
    page_title="Career Explorer", 
    page_icon="📊", 
    layout="wide"
)

# categories listed on the dataset
CATEGORIES = sorted([
    'HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE', 
    'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'BPO', 
    'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE', 
    'APPAREL', 'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS', 
    'BANKING', 'ARTS', 'AVIATION'
])

# initialize config
try:
    supervisor_executor = get_supervisor_executor()
    langfuse_handler = create_langfuse_handler(trace_name="CareerExplorer_Trace") 

    if "explorer_session_id" not in st.session_state:
        st.session_state.explorer_session_id = f"st-explorer-{uuid4().hex[:8]}"

    supervisor_config = {
        "callbacks": [langfuse_handler],
        "configurable": {"session_id": st.session_state.explorer_session_id},
    }
except Exception as e:
    st.error(f"Failed to load AI Supervisor: {e}")
    st.stop()

client = setup_qdrant_client()

# Create index
client.create_payload_index(
    collection_name="capstone",
    field_name="metadata.category",
    field_schema=PayloadSchemaType.KEYWORD
)

st.title("Career Explorer")

st.markdown("### What Can You Explore?")
st.markdown("- Dive deep into career data from real resumes accross 24 industries")
st.markdown("- Get AI-powered insights on skills, career paths, and more!")
st.divider()

tab1, tab2, tab3 = st.tabs(["🎯 Career Deep Dive", "⚖️ Compare Categories", "🔥 Trending Skills"])

with tab1:
    st.subheader("Single Career Deep Dive")
    st.write("Get comprehensive AI analysis about a career field using resume data.")

    # Search Category and Select Box Logic
    search_term = st.text_input("Filter categories:", placeholder="Type to filter...", key="deep_search")
    filtered_cats = [cat for cat in CATEGORIES if search_term.upper() in cat] if search_term else CATEGORIES
    category = st.selectbox(
        "Select a career category:",
        filtered_cats,
        index=0,
        format_func=lambda x: x.replace('-', ' ').title(), 
        key="deep_cat"
    )

    analyze_btn_deep = st.button("Analyze Career", type="primary", width="stretch", key="analyze_deep")

    if analyze_btn_deep and category:
        st.markdown("---") 
        st.subheader(f"Analyzing: {category.replace('-', ' ').title()}")

        user_prompt = f"Provide a deep dive analysis for the {category} category."
        inputs = {"messages": [HumanMessage(content=user_prompt)]}

        with st.spinner("AI are working on your analysis..."):
            try:
                final_state = supervisor_executor.invoke(inputs, config=supervisor_config)

                # Stream Agent Actions
                st.subheader("Agent Process Log")
                all_messages = final_state.get('messages', [])
                

                for i, msg in enumerate(all_messages):
                    if isinstance(msg, HumanMessage):
                        continue 
                    elif isinstance(msg, ToolMessage):
                        st.success(f"**Tool Result {i+1}:** `{msg.name}`")
                        display_content = msg.content 
                        st.text_area("Raw Tool Output:", value=display_content, height=200, key=f"tool_result_deep_{i}")
                    elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                        st.info(f"**Agent Action {i+1}:** Called tool `{msg.tool_calls[0]['name']}`")
                        st.code(msg.content) 

                # Iterate backwards to find the last non-tool-calling AI message
                final_response_content = ""
                for msg in reversed(all_messages):
                    if isinstance(msg, SystemMessage): 
                        continue
                    elif isinstance(msg, HumanMessage): 
                        continue
                    elif isinstance(msg, ToolMessage): 
                        continue
                    elif hasattr(msg, 'tool_calls') and msg.tool_calls: 
                        continue
                    elif isinstance(msg, AIMessage): 
                        final_response_content = msg.content
                        break
                    elif hasattr(msg, 'content'):
                        final_response_content = msg.content
                        break

                # Display the final response
                if final_response_content:
                    st.subheader("Agent Response")
                    st.markdown(final_response_content)
                else:
                    st.error("Sorry, I couldn't generate a response.")
                    st.session_state.qa_messages.append(AIMessage(content="Error"))

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                import traceback
                st.code(traceback.format_exc()) 

with tab2:
    # Compare Two Different Category
    st.subheader("Side-by-Side Career Comparison")
    st.write("Compare two career fields using AI analysis of resume data.")

    col1, col2 = st.columns(2)
    with col1:
        cat1 = st.selectbox(
            "Select first category:", CATEGORIES, index=CATEGORIES.index("INFORMATION-TECHNOLOGY"),
            format_func=lambda x: x.replace('-', ' ').title(), key="cat1"
        )
    with col2:
        cat2 = st.selectbox(
            "Select second category:", CATEGORIES, index=CATEGORIES.index("SALES"),
            format_func=lambda x: x.replace('-', ' ').title(), key="cat2"
        )

    compare_btn = st.button("Compare Careers", type="primary", width="stretch", key="compare_btn")

    if compare_btn:
        if cat1 == cat2:
            st.warning("Please select two different categories.")
        else:
            st.markdown("---")
            st.subheader(f"Comparing: {cat1.replace('-', ' ').title()} vs {cat2.replace('-', ' ').title()}")

            user_prompt = f"Compare the {cat1} and {cat2} career categories."
            inputs = {"messages": [HumanMessage(content=user_prompt)]}

            with st.spinner("AI are working on your comparison..."):
                try:
                    final_state = supervisor_executor.invoke(inputs, config=supervisor_config)

                    st.subheader("Agent Process Log")
                    all_messages = final_state.get('messages', [])
                    
                    for i, msg in enumerate(all_messages):
                        if isinstance(msg, HumanMessage):
                            continue 
                        elif isinstance(msg, ToolMessage):
                            st.success(f"**Tool Result {i+1}:** `{msg.name}`")
                            display_content = msg.content
                            st.text_area("Raw Tool Output:", value=display_content, height=100, key=f"tool_result_compare_{i}")
                        elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                            st.info(f"**Agent Action {i+1}:** Called tool `{msg.tool_calls[0]['name']}`")
                            st.code(msg.content)
                   
                    # Iterate backwards to find the last non-tool, non-tool-calling AI message
                    final_response_content = ""
                    for msg in reversed(all_messages):
                        if isinstance(msg, SystemMessage): 
                            continue
                        elif isinstance(msg, HumanMessage): 
                            continue
                        elif isinstance(msg, ToolMessage): 
                            continue
                        elif hasattr(msg, 'tool_calls') and msg.tool_calls: 
                            continue
                        elif isinstance(msg, AIMessage): 
                            final_response_content = msg.content
                            break
                        elif hasattr(msg, 'content'): 
                            final_response_content = msg.content
                            break

                    # Display the final response
                    if final_response_content:
                        st.subheader("Agent Response")
                        st.markdown(final_response_content)
                    else:
                        st.error("Sorry, I couldn't generate a response.")
                        st.session_state.qa_messages.append(AIMessage(content="Error"))

                except Exception as e:
                    st.error(f"An error occurred during comparison: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with tab3:
    # Trending Skills
    st.subheader("Cross-Industry Skill Trends")
    st.write("Discover skills in demand across multiple industries based on AI analysis.")

    trends_btn = st.button("Analyze Global Trends", type="primary", width="stretch", key="trends_btn")

    if trends_btn:
        st.markdown("---")
        st.subheader("Analyzing Trends...")

        llm = setup_chat_models() 

        try:
            langfuse_handler_trends = create_langfuse_handler(trace_name="TrendingSkills_Trace")
            trends_config = {
                "callbacks": [langfuse_handler_trends],
                "configurable": {"session_id": st.session_state.explorer_session_id + "_trends"} 
            }
        except Exception as e:
            st.error(f"Failed to setup Langfuse for Trends: {e}")
            trends_config = {} 

        TREND_PROMPT = """
        Analyze and report on current cross-industry skill trends based on your general knowledge up to your last training data. Provide comprehensive insights covering:

        1.  Universal Skills (Cross-Industry): 5-7 skills valuable across many fields (e.g., communication, problem-solving). Explain their value.
        2.  Top Technical Skills: List 5-10 in-demand technical skills (languages, platforms, tools). Mention relevant industries.
        3.  Emerging & Future-Ready Skills: Highlight skills gaining momentum (e.g., AI/ML, Cloud, Data Science, Green Tech). Discuss their future outlook.
        4.  Most Valued Soft Skills:** Identify key soft skills employers seek (e.g., leadership, adaptability).
        5.  Trending Certifications:** Mention 3-5 certifications that are currently popular or valuable across different sectors.

        Format with clear headers and bullet points. Be insightful and forward-looking.
        """

        messages = [
            SystemMessage(content="You are a labor market analyst and career strategist. Provide insights on skill trends based on your general knowledge."),
            HumanMessage(content=TREND_PROMPT)
        ]

        final_response_container = st.empty()
        full_response = ""

        with st.spinner("AI is analyzing global trends..."):
            try:
                for chunk in llm.stream(messages, config=trends_config): 
                    full_response += chunk.content
                    final_response_container.markdown(full_response + "▌")

                final_response_container.markdown(full_response)
                st.success("Trend analysis complete!")

            except Exception as e:
                st.error(f"An error occurred during trend analysis: {e}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")

st.info(
    "**Note:** All analyses use AI to process real resume data from our database. "
    "Results may vary based on the sample size and data quality."
)
