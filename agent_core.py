import streamlit as st
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from uuid import uuid4
from datetime import datetime
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
import operator
import logging
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)



# --------------------- Configs ----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]

LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]
LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_HOST = st.secrets["LANGFUSE_HOST"]

Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

llm_instance = None
embeddings_instance = None
qdrant_client_instance = None

def setup_chat_models():
    global llm_instance
    if llm_instance is None:
        print("Initializing ChatOpenAI model...")
        llm_instance = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0, streaming=True, timeout=180)
    return llm_instance

def setup_embedding_models():
    global embeddings_instance
    if embeddings_instance is None:
        print("Initializing OpenAI Embeddings...") 
        embeddings_instance = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    return embeddings_instance

def setup_qdrant_client():
    global qdrant_client_instance
    if qdrant_client_instance is None:
        print("Initializing Qdrant Client...") 
        qdrant_client_instance = QdrantClient(
            api_key=QDRANT_API_KEY,
            url=QDRANT_URL,
            timeout=60
        )
    return qdrant_client_instance


# create qdrant index
def create_category_index():
    """Create payload index for category field in Qdrant"""
    try:
        client = setup_qdrant_client()
        
        client.create_payload_index(
            collection_name="capstone",
            field_name="metadata.category",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        return True
        
    except Exception as e:
        error_str = str(e)
        if "already exists" in error_str.lower() or "existing" in error_str.lower():
            return True
        
        import traceback
        print(f"Error creating index: {e}\n{traceback.format_exc()}")
        return False
    
def debug_qdrant_data():
    """Debug function to check Qdrant data structure"""
    try:
        client = setup_qdrant_client()
        
        # Get collection info
        collection_info = client.get_collection("capstone")
        st.write("### Collection Info:")
        st.json({
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": str(collection_info.status),
            "indexed_vectors_count": collection_info.indexed_vectors_count
        })
        

        st.write("### Sample Data (First 5 points):")
        scroll_result = client.scroll(
            collection_name="capstone",
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        for i, point in enumerate(scroll_result[0]):
            st.write(f"**Point {i+1}:**")
            st.json(point.payload)
        
        # Check specific category
        st.write("### Testing HR Filter:")
        hr_filter = Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value="HR")
                )
            ]
        )
        
        hr_results = client.scroll(
            collection_name="capstone",
            scroll_filter=hr_filter,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        st.write(f"Found {len(hr_results[0])} HR records")
        for point in hr_results[0]:
            st.json(point.payload)
            
        return True
    except Exception as e:
        st.error(f"Debug error: {e}")
        return False


def setup_vector_store():
    client = setup_qdrant_client()
    embeddings = setup_embedding_models()

    create_category_index() 

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="capstone",
        embedding=embeddings,
        vector_name="" 
    )
    return vector_store

def create_langfuse_handler(user_id="default_user", session_prefix="session", trace_name="default_trace", release="v1.0"):
    handler = CallbackHandler()
    return handler


# --------------------  Tools ----------------------------
@tool
def single_career_deep_dive(category: str):
    """
    Retrieves the text content of the top 10 most relevant resume chunks
    for a specific job 'category'. This raw text can then be summarized.
    Returns a single string containing the combined text of the chunks,
    or a message if no data is found.
    
    Available categories include: ADVOCATE, AGRICULTURE, BANKING, BUSINESS-DEVELOPMENT,
    CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE,
    FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER
    """
    try:
        
        vector_store = setup_vector_store()
        category_normalized = category.upper().strip()
    
        category_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.category",
                    match=MatchValue(value=category_normalized)
                )
            ]
        )
        
        results = vector_store.similarity_search(
            query=f"Resume information skills experience for {category_normalized}",
            k=10,
            filter=category_filter
        )
         
        if not results:
            msg = f"No resume data found for category '{category_normalized}'. The category might not exist in the database."
            return msg
        
        if results:
            combined_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        
        return combined_text
        
    except Exception as e:
        error_msg = f"Error in single_career_deep_dive: {str(e)}"
        return error_msg


@tool
def compare_category(category1: str, category2: str):
    """
    Retrieves the text content of the top 10 relevant resume chunks for TWO
    specified job categories. Returns a dictionary with combined text for each.
    """
    try:
        vector_store = setup_vector_store()
    
        cat1_norm = category1.upper().strip()
        cat2_norm = category2.upper().strip()
        
        filter_1 = Filter(
            must=[FieldCondition(key="metadata.category", match=MatchValue(value=cat1_norm))]
        )
        filter_2 = Filter(
            must=[FieldCondition(key="metadata.category", match=MatchValue(value=cat2_norm))]
        )
        
        result_1 = vector_store.similarity_search(
            query=f"Resume information for {cat1_norm}",
            k=10,
            filter=filter_1
        )
        
        result_2 = vector_store.similarity_search(
            query=f"Resume information for {cat2_norm}",
            k=10,
            filter=filter_2
        )
        
        if not result_1 or not result_2:
            return f"Insufficient data for comparison. Found {len(result_1)} for '{cat1_norm}' and {len(result_2)} for '{cat2_norm}'"
        
        combined_text_1 = "\n\n---\n\n".join([doc.page_content for doc in result_1])
        combined_text_2 = "\n\n---\n\n".join([doc.page_content for doc in result_2])
        
        return {
            cat1_norm: combined_text_1,
            cat2_norm: combined_text_2
        }
        
    except Exception as e:
        error_msg = f"Error in compare_category: {str(e)}"
        return error_msg


@tool 
def analyze_resume(user_resume_text: str, target_category: str):
    """
    Finds resume examples similar to the user's resume within a specific target category.
    Returns structured information including the content and metadata (like doc_id) of the similar chunks.
    """
    try:
        print(f"Starting analyze_resume for category: {target_category}")
        
        vector_store = setup_vector_store()
        category_normalized = target_category.upper().strip()
        
        category_filter = Filter(
            must=[FieldCondition(key="metadata.category", match=MatchValue(value=category_normalized))]
        )
        
        print(f"Applying filter: {category_filter}")

        # Perform the similarity search with scores
        results_with_scores = vector_store.similarity_search_with_score(
            query=user_resume_text,
            k=5, 
            filter=category_filter,
            timeout=60  
        )

        print(f"Found {len(results_with_scores)} results.")

        if not results_with_scores:
            return f"No similar resumes found in '{category_normalized}' category based on the content provided."

        
        processed_results = []
        for doc, score in results_with_scores:
            # Extract metadata
            doc_id = doc.metadata.get('doc_id', 'N/A')
            doc_category = doc.metadata.get('category', 'N/A') 
            content = doc.page_content
            
            processed_results.append({
                "doc_id": doc_id,
                "category": doc_category,
                "similarity_score": score, 
                "content": content
            })

        formatted_output = f"Found {len(processed_results)} similar resume chunks for '{category_normalized}':\n\n"
        for i, res in enumerate(processed_results):
            formatted_output += f"--- Chunk {i+1} (doc_id: {res['doc_id']}, Score: {res['similarity_score']:.4f}) ---\n"
            formatted_output += res['content'] + "\n\n"

        return formatted_output 

    except Exception as e:
        print(f"Error in analyze_resume: {str(e)}")
        return f"An unexpected error occurred while analyzing your resume. Please try again later. Details: {str(e)}"


@tool
def retrieve_information(query: str, category: str = None):
    """
    Searches the resume database for text chunks relevant to the query.
    Optionally filters by category.
    """
    try:
        vector_store = setup_vector_store()
        
        filter_obj = None
        if category:
            category_normalized = category.upper().strip()
            filter_obj = Filter(
                must=[FieldCondition(key="metadata.category", match=MatchValue(value=category_normalized))]
            )
        
        results = vector_store.similarity_search(
            query=query,
            k=5,
            filter=filter_obj
        )
        
        if not results:
            return "No information retrieved."
        
        structured_results = []
        for doc in results:
            metadata = doc.metadata or {}
            structured_results.append({
                "id": metadata.get('doc_id', 'N/A'),
                "category": metadata.get('category', 'N/A'),
                "content": doc.page_content
            })
        
        return structured_results
        
    except Exception as e:
        error_msg = f"Error in retrieve_information: {str(e)}"
        return error_msg

@tool 
def get_current_time():
    """
    Returns the current date and time as a formatted string.
    Example: 2025-10-23 23:55:58
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------- Prompts ----------------------------
career_explorer_prompt = """
You are a Career Explorer assistant. Your goal is to provide users with insights into specific career fields using data retrieved from a resume database. You have two primary tools:

1.  `single_career_deep_dive(category: str)`: Use this tool when the user wants detailed information about ONE specific career category (e.g., "Tell me about Designers", "Deep dive into Finance roles").
2.  `compare_categories(category1: str, category2: str)`: Use this tool ONLY when the user explicitly asks to compare TWO different career categories (e.g., "Compare Sales vs. Engineering", "What are the differences between HR and IT?").

**Your Task:**
- Analyze the user's request.
- Determine if they want information on one category or a comparison of two.
- Call the appropriate tool with the correct category name(s). Category names must match the known list (e.g., 'INFORMATION-TECHNOLOGY', 'SALES').
- Once the tool returns the relevant resume text data (either a single block or a dictionary with two blocks), present this information clearly to the user, perhaps suggesting you can summarize or analyze it further if needed.
- If the tool returns an error or no data, inform the user politely.
- Do NOT answer general knowledge questions. Only use the tools to retrieve and present career information from the database.
"""

resume_analyzer_prompt = """
You are a professional Resume Analyzer assistant. 

**YOUR TASK:**
1. When you receive a resume and target category, you MUST immediately call the `analyze_resume` tool
2. Pass the full resume text to `user_resume_text` parameter
3. Pass the target category to `target_category` parameter
4. The tool will return similar resumes from the database
5. After getting the tool results, provide detailed analysis comparing the user's resume with the retrieved examples

**ANALYSIS FORMAT:**
Provide your feedback in this structure:
1. **Strengths** (2-3 key strong points)
2. **Gaps & Missing Elements** (2-3 critical gaps vs industry standard)
3. **Actionable Improvements** (4-5 specific recommendations)
4. **Industry Comparison** (How it stacks up)
5. **Final Encouragement** (Motivational closing)

Be constructive, specific, and actionable. Reference the retrieved resume examples when relevant.

If the tool returns no data or an error, inform the user politely.
"""

qa_agent_prompt = """
You are a versatile Q&A assistant for career-related topics. You have access to a resume database via the `retrieve_information` tool and can provide the current time using `get_current_time`.

**Your Core Logic:**
1.  **Prioritize RAG:** If the user's query asks about resume content, specific job skills, career examples, industry details, or anything likely found in the resume database (e.g., "What skills for Sales?", "Example HR summary", "What does a Designer do?"), you MUST attempt to use the `retrieve_information(query: str, category: str = None)` tool first.
    * If a specific job category (e.g., 'DESIGNER', 'FINANCE') is mentioned, pass it to the `category` argument.
    * Base your answer primarily on the structured results returned by the tool (list of dicts with 'id', 'category', 'content'). You can summarize or synthesize the information from the retrieved content. Cite sources briefly if helpful (e.g., "Based on resumes from the [Category] field...").
    * If the tool returns an error message or "No information found...", inform the user politely. You may then offer general information if appropriate, but clearly state it's not from the database.

2.  **Use Current Time:** If the user's query implies needing current time context (e.g., "What time is it?", "Is today Monday?"), use the `get_current_time()` tool.

3.  **Handle Greetings and General Knowledge:** For simple greetings ("Hello", "Hi", "Good morning"), respond with a polite greeting (e.g., "Hello! How can I help you today?"). Do NOT echo the greeting back. For other general knowledge questions that are not career/resume specific and cannot be answered by the tools, provide a helpful response based on your general knowledge. For completely off-topic questions ("What's the weather?"), you can politely redirect to career topics or provide a brief general response.

**Conduct:**
- Be professional, helpful, and concise.
- Do not make up information about resume content. Stick to the retrieved data when using RAG.
- Do not echo the user's input unless explicitly asked to repeat something.
"""

supervisor_prompt = """
You are a supervisor agent managing three specialist agents:
1.  `CareerExplorerAgent`: Provides deep dives or comparisons of career categories based on resume data. Handles requests like "Tell me about HR" or "Compare Designer and Engineer".
2.  `ResumeAnalyzerAgent`: Analyzes a user's specific resume against a target career category. Handles requests like "Analyze my resume for Sales" or "Give me feedback on this resume for IT".
3.  `QAAgent`: Answers specific questions, using a resume database (RAG) for career/resume topics and general knowledge for other queries. Handles specific questions like "What are common skills for Finance?" or "What is Python?".

**Your Task:**
- Examine the user's request carefully.
- Determine the user's primary intent.
- Route the request to the MOST appropriate agent based on their specialty:
    - If the user wants to explore, get an overview, or compare career fields -> **`CareerExplorerAgent`**
    - If the user explicitly asks to analyze *their* resume or provides resume text for feedback -> **`ResumeAnalyzerAgent`**
    - For all other specific questions (about skills, roles, general knowledge, greetings) -> **`QAAgent`**
- You MUST route to one of these agents. Do NOT attempt to answer the question yourself. Respond only with the name of the agent to route to (e.g., `CareerExplorerAgent`, `ResumeAnalyzerAgent`, `QAAgent`).
"""
# ---------------------------- Agents ------------------------------
llm = setup_chat_models()

# Career Explorer Agent
career_explorer_agent = create_react_agent(
    model=llm,
    tools=[single_career_deep_dive, compare_category],
    prompt=career_explorer_prompt
)

# Resume Analyzer Agent
resume_analyzer_agent = create_react_agent(
    model=llm,
    tools=[analyze_resume],
    prompt=resume_analyzer_prompt
)

# Q&A Agent
chatbot_agent = create_react_agent(
    model=llm,
    tools=[get_current_time, retrieve_information],
    prompt=qa_agent_prompt
)

# Supervisor Agent
class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def supervisor_router(state: SupervisorState):
    messages = state['messages']
    
    last_message = messages[-1]
    
    supervisor_message = f"""
    Based on this user request, route to ONE agent and respond ONLY with the agent name:
    - CareerExplorerAgent: for career exploration or comparison
    - ResumeAnalyzerAgent: for resume analysis with resume text provided
    - QAAgent: for specific questions

    User request: {messages[-1].content} # This should be the latest HumanMessage

    Respond with ONLY the agent name, nothing else.
    """
    
    response = llm.invoke([SystemMessage(content=supervisor_prompt), 
                           HumanMessage(content=supervisor_message)])
    next_agent = response.content.strip()
    print(f"[DEBUG] Supervisor received from LLM: '{response.content.strip()}' -> Final Agent: '{next_agent}'") 
    
    valid_agents = ["CareerExplorerAgent", "ResumeAnalyzerAgent", "QAAgent"]
    if next_agent not in valid_agents:
        print(f"[DEBUG] Supervisor: Invalid agent name '{next_agent}', defaulting to QAAgent")
        next_agent = "QAAgent"
    
    return {
        "messages": messages,  
        "next": next_agent
    }


def run_career_explorer(state: SupervisorState):
    result = career_explorer_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

def run_resume_analyzer(state: SupervisorState):
    result = resume_analyzer_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

def run_qa_agent(state: SupervisorState):
    result = chatbot_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

# Define the graph
supervisor_graph = StateGraph(SupervisorState)

supervisor_graph.add_node("CareerExplorerAgent", run_career_explorer)
supervisor_graph.add_node("ResumeAnalyzerAgent", run_resume_analyzer)
supervisor_graph.add_node("QAAgent", run_qa_agent)
supervisor_graph.add_node("Supervisor", supervisor_router)

supervisor_graph.add_conditional_edges(
    "Supervisor",
    lambda state: state['next'],
    {
        "CareerExplorerAgent": "CareerExplorerAgent",
        "ResumeAnalyzerAgent": "ResumeAnalyzerAgent",
        "QAAgent": "QAAgent",
        "END": END
    }
)

supervisor_graph.add_edge("CareerExplorerAgent", END)
supervisor_graph.add_edge("ResumeAnalyzerAgent", END)
supervisor_graph.add_edge("QAAgent", END)

supervisor_graph.set_entry_point("Supervisor")

@st.cache_resource
def get_supervisor_executor():
   
    app = supervisor_graph.compile()

    return app

if __name__ == "__main__":
    sup_exec = get_supervisor_executor()
    print("Supervisor Executor obtained:", sup_exec)

