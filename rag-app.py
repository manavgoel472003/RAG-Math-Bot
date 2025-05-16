import os
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Annotated
from dataclasses import dataclass, field
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langgraph.graph import StateGraph, END
from vectorstore_manager import VectorstoreManager
import tempfile
import operator

# Configuration
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
TOP_K = 3

# Available models
AVAILABLE_MODELS = {
    "Qwen 2.5 (7B)": "qwen2.5:7b",
    "Llama 3 (8B)": "llama3:8b",
    "Mistral (7B)": "mistral",
    "DeepSeek R1 (8B)": "deepseek-r1:8b"
}

@dataclass
class State:
    """State class for the RAG workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add] = field(default_factory=list)
    current_question: str = ""
    instructions: Annotated[List[str], operator.add] = field(default_factory=list)
    context: str = ""
    calculations: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

# Initialize session state
if "chat_state" not in st.session_state:
    st.session_state.chat_state = State()
if "current_model" not in st.session_state:
    st.session_state.current_model = "Qwen 2.5 (7B)"

def get_llm(model_name: str) -> ChatOllama:
    """Get LLM instance for the selected model"""
    return ChatOllama(
        model=AVAILABLE_MODELS[model_name],
        temperature=0.2,
        base_url=OLLAMA_BASE
    )

def load_vectorstore(path: str):
    """Load pre-stored FAISS vectorstore"""
    embedder = OllamaEmbeddings(base_url=OLLAMA_BASE, model=EMBED_MODEL)
    return FAISS.load_local(path, embedder)

def break_question_into_instructions(state: State) -> Dict[str, Any]:
    """You are Math tutor, Break down the question into specific instructions"""
    print("DEBUG: Starting break_question_into_instructions")
    
    if not state.current_question:
        print("DEBUG: No question provided")
        return {}
    
    # Include chat history in the prompt for context
    chat_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.messages[-4:]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a precise calculation assistant that breaks down questions into specific instructions.
        Your task is to break down the given question into 2-3 specific, actionable instructions.
        DO NOT ask for the question - it will be provided to you.
        DO NOT respond with meta-instructions.
        DO NOT include phrases like "Please provide" or "I will".
        
        Previous conversation context:
        {chat_context}
        
        Example good responses:
        Question: "What is linear algebra?"
        1. Define linear algebra and its core concepts
        2. Explain key applications and importance
        3. Provide basic examples of linear algebra operations
        
        Question: "How do I solve quadratic equations?"
        1. Explain the standard form of quadratic equations
        2. Describe the quadratic formula and its components
        3. Show step-by-step solution process with an example
        
        Now break down this question into specific instructions:"""),
        ("human", "{question}")
    ])
    
    try:
        response = get_llm(st.session_state.current_model).invoke(prompt.format(
            question=state.current_question,
            chat_context=chat_context
        ))
        
        # Parse the response to extract numbered instructions
        instructions = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line and any(line.startswith(str(i) + ".") for i in range(1, 10)):
                # Remove the number and dot prefix
                instruction = line.split(".", 1)[1].strip()
                if instruction and not any(phrase in instruction.lower() for phrase in ["please", "i will", "you should"]):
                    instructions.append(instruction)
        
        # If no valid instructions were found, use default ones
        if not instructions:
            print("DEBUG: No valid instructions found, using defaults")
            instructions = [
                "Explain the main concept in simple terms",
                "Provide key examples or applications",
                "Break down the core components or steps"
            ]
        
        print(f"DEBUG: Generated instructions: {instructions}")
        return {"instructions": instructions}
        
    except Exception as e:
        print(f"DEBUG: Error in break_question_into_instructions: {str(e)}")
        # Use default instructions if there's an error
        return {
            "instructions": [
                "Explain the main concept in simple terms",
                "Provide key examples or applications",
                "Break down the core components or steps"
            ]
        }

def retrieve_relevant_context(state: State, vectorstore) -> Dict[str, Any]:
    """Retrieve all relevant context for the instructions at once"""
    print("DEBUG: Starting retrieve_relevant_context")
    # print(f"DEBUG: Current state: {state}")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # Combine all instructions into one query to get comprehensive context
    combined_query = " ".join([state.current_question] + state.instructions)
    print(f"DEBUG: Search query: {combined_query}")
    
    relevant_docs = retriever.get_relevant_documents(combined_query)
    print(f"DEBUG: Found {len(relevant_docs)} relevant documents")
    
    # Store the context in the state
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return {"context": context}

def perform_calculations(state: State) -> Dict[str, Any]:
    """Perform calculations using the retrieved context"""
    print("DEBUG: Starting perform_calculations")
    # print(f"DEBUG: Current state: {state}")
    
    calculations = {}
    for instruction in state.instructions:
        print(f"DEBUG: Processing instruction: {instruction}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise calculation assistant. 
            Use the following context to answer the instruction.
            If the instruction requires calculations, show your work step by step.
            If no calculations are needed, provide a clear and detailed explanation.
            Context: {context}"""),
            ("human", "{instruction}")
        ])
        
        response = get_llm(st.session_state.current_model).invoke(prompt.format(
            context=state.context,
            instruction=instruction
        ))
        
        calculations[instruction] = response.content
        print(f"DEBUG: Generated calculation for instruction")
    
    return {"calculations": calculations}

def generate_summary(state: State) -> Dict[str, Any]:
    """Generate a summary based on calculations"""
    print("DEBUG: Starting generate_summary")
    # print(f"DEBUG: Current state: {state}")
    
    chat_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.messages[-4:]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that provides comprehensive answers. Make sure to use the calculations to answer the question. Also output in markdown format.
        Previous conversation context:
        {chat_context}
        
        Based on the following calculations and the conversation context, 
        provide a comprehensive answer to the original question.
        Make sure to maintain consistency with previous answers.
        
        Calculations:
        {calculations}"""),
        ("human", "{question}")
    ])
    
    response = get_llm(st.session_state.current_model).invoke(prompt.format(
        calculations=str(state.calculations),
        question=state.current_question,
        chat_context=chat_context
    ))
    
    print("DEBUG: Generated summary")
    return {"summary": response.content}

def update_chat_buffer(state: State) -> Dict[str, Any]:
    """Update chat buffer with new question and answer"""
    print("DEBUG: Starting update_chat_buffer")
    # print(f"DEBUG: Current state: {state}")
    
    new_messages = [
        {"role": "user", "content": state.current_question},
        {"role": "assistant", "content": state.summary}
    ]
    
    print("DEBUG: Updated chat buffer")
    return {"messages": new_messages}

def build_workflow(vectorstore):
    """Build the LangGraph workflow"""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("break_question", break_question_into_instructions)
    workflow.add_node("retrieve_context", lambda x: retrieve_relevant_context(x, vectorstore))
    workflow.add_node("calculate", perform_calculations)
    workflow.add_node("summarize", generate_summary)
    workflow.add_node("update_buffer", update_chat_buffer)
    
    # Define edges
    workflow.add_edge("break_question", "retrieve_context")
    workflow.add_edge("retrieve_context", "calculate")
    workflow.add_edge("calculate", "summarize")
    workflow.add_edge("summarize", "update_buffer")
    workflow.add_edge("update_buffer", END)
    
    # Set entry point
    workflow.set_entry_point("break_question")
    
    # Visualize the graph
    compiled = workflow.compile()
    
    return compiled

def process_question(question: str, vectorstore, chat_state: State) -> Dict[str, Any]:
    """Process a question through the workflow"""
    print(f"DEBUG: Processing question: {question}")
    
    # Initialize state
    state = State(
        messages=chat_state.messages,
        current_question=question,
        instructions=[],
        context="",
        calculations={},
        summary=""
    )
    
    workflow = build_workflow(vectorstore)
    result = workflow.invoke(state)
    print(f"DEBUG: Workflow result: {result is not None}")
    
    # Check if result is None
    if result is None:
        print("DEBUG: Workflow returned None")
        return {
            "answer": "I apologize, but I couldn't process your question. Please try again.",
            "chat_history": chat_state.messages,
            "instructions": [],
            "calculations": {}
        }
    
    # Update the chat state
    chat_state.messages.extend(result.get("messages", []))
    chat_state.instructions.extend(result.get("instructions", []))
    chat_state.context = result.get("context", "")
    chat_state.calculations.update(result.get("calculations", {}))
    chat_state.summary = result.get("summary", "")
    
    return {
        "answer": result.get("summary", "No answer generated"),
        "chat_history": chat_state.messages,
        "instructions": result.get("instructions", []),
        "calculations": result.get("calculations", {})
    }

# Streamlit UI
st.set_page_config(page_title="Advanced RAG Chat", page_icon="🤖")
st.title("🤖 RAG-Math Bot")

# Initialize vectorstore manager
VECTORSTORE_PATH = "vector_storage"
try:
    manager = VectorstoreManager(VECTORSTORE_PATH)
    vectorstore = manager.get_vectorstore()
    st.success("✅ Vectorstore loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading vectorstore: {str(e)}")
    st.stop()

# Model selector in sidebar
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Select LLM Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.current_model),
        key="model_selector"
    )
    
    # Update model without rerunning
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.info(f"Switched to {selected_model} model. Chat history preserved.")
    
    st.header("Add New Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="document_uploader"  # Added unique key to prevent duplication
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = Path(tmp.name)
            
            try:
                # Add to vectorstore
                manager.add_file(str(tmp_path))
                st.success(f"✅ Added {uploaded_file.name} to vectorstore")
            except Exception as e:
                st.error(f"❌ Error adding {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temporary file
                tmp_path.unlink()

# Display chat history
for message in st.session_state.chat_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Process question
    with st.spinner("Thinking..."):
        # Get LLM instance for current model
        llm = get_llm(st.session_state.current_model)
        result = process_question(prompt, vectorstore, st.session_state.chat_state)
        
        # Display assistant's response
        with st.chat_message("assistant"):
            st.write(result["answer"])
            
            # Display reasoning process in expander
            with st.expander("Show reasoning process"):
                st.write("### Instructions")
                for i, instruction in enumerate(result["instructions"], 1):
                    st.write(f"{i}. {instruction}")
                
                st.write("### Thoughts")
                for instruction, calculation in result["calculations"].items():
                    st.write(f"**Instruction**: {instruction}")
                    st.write(f"**Thoughts**: {calculation}")
                    st.write("---") 