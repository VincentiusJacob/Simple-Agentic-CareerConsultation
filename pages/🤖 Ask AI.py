# Ask AI.py
import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from agent_core import get_supervisor_executor, create_langfuse_handler

st.set_page_config(page_title="Career Q&A", page_icon="🗣️")
st.title("🗣️ Career Q&A")
st.write("Ask anything about careers.")

try:
    supervisor_executor = get_supervisor_executor()
    langfuse_handler = create_langfuse_handler()
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"st-session-{os.urandom(4).hex()}"
        
    agent_config = {
        "callbacks": [langfuse_handler],
        "configurable": {"session_id": st.session_state.session_id},
        "recursion_limit": 50,
    }

except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display the initial greeting if no messages yet
if len(st.session_state.chat_history) == 0:
    st.chat_message("assistant").write("Hello! How can I help you with your career questions?")

# Display all messages in the chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:  # role == "assistant"
        st.chat_message("assistant").write(message["content"])

# Handle user input
if prompt := st.chat_input("Example: What is the most important skill for Banking?"):
    # Add user message to chat history FIRST
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        with st.spinner("AI is thinking..."):
            try:
                # Prepare messages for the agent
                messages_for_agent = [
                    SystemMessage(content="""
You are a helpful assistant for career-related questions.
- Provide clear, helpful responses to career-related queries.
- Use the appropriate tools when needed to retrieve information from the resume database.
- Be concise and professional in your responses.
""")
                ]
                
                # Add chat history (including the NEW user message we just added)
                # Take last 10 messages to keep context manageable
                recent_messages = st.session_state.chat_history[-10:]
                
                for msg in recent_messages:
                    if msg["role"] == "user":
                        messages_for_agent.append(HumanMessage(content=msg["content"]))
                    else:  # assistant
                        messages_for_agent.append(AIMessage(content=msg["content"]))
                
                # Invoke the supervisor graph
                inputs_for_graph = {"messages": messages_for_agent}
                final_state = supervisor_executor.invoke(inputs_for_graph, config=agent_config)

                # --- Display Agent Process Log ---
                with st.expander("Agent Process Log", expanded=False):
                    all_messages = final_state.get('messages', [])
                    
                    for i, msg in enumerate(all_messages):
                        if isinstance(msg, ToolMessage):
                            st.success(f"**Tool Result {i+1}:** `{msg.name}`")
                            display_content = msg.content
                            st.text_area("Raw Tool Output:", value=display_content, height=100, key=f"tool_result_{i}")
                        elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                            st.info(f"**Agent Action {i+1}:** Called tool `{msg.tool_calls[0]['name']}`")
                            if msg.content:
                                st.code(msg.content)

                # Extract final response (last AIMessage without tool_calls)
                final_response_content = ""
                for msg in reversed(all_messages):
                    if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                        final_response_content = msg.content
                        break
                    elif isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and not msg.tool_calls:
                        final_response_content = msg.content
                        break

                # Display the final response
                if final_response_content:
                    response_placeholder.markdown(final_response_content)
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": final_response_content})
                else:
                    error_msg = "Sorry, I couldn't generate a response."
                    response_placeholder.markdown(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            except Exception as e:
                st.error(f"Error during execution: {e}")
                import traceback
                st.code(traceback.format_exc())
                error_msg = "An error occurred while processing your request."
                response_placeholder.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})