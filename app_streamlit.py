"""
app_streamlit.py - Interactive Chat UI
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
from streamlit_chat import message
from datetime import datetime
import logging
from pathlib import Path
import json

from rag_pipeline import RAGChatbot
from vector_store import VectorStoreManager
from document_processor import DocumentProcessor
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Company Document Chatbot",
    page_icon="robot_face",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-badge {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.vs_manager = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False


def initialize_chatbot():
    """Initialize the chatbot and vector database connection."""
    with st.spinner("Initializing Chatbot..."):
        try:
            st.session_state.vs_manager = VectorStoreManager()
            st.session_state.chatbot = RAGChatbot(st.session_state.vs_manager)
            st.session_state.initialized = True
            st.success("Chatbot initialized successfully")
        except Exception as e:
            st.error(f"Initialization error: {e}")
            logger.error(f"Error initializing chatbot: {e}")


def upload_documents():
    """Handle document upload and processing."""
    st.subheader("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Select documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            try:
                temp_dir = Path("./temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                for file in uploaded_files:
                    with open(temp_dir / file.name, "wb") as f:
                        f.write(file.getbuffer())
                
                processor = DocumentProcessor()
                documents = processor.process_documents(str(temp_dir))
                
                # Auto-initialize if not already done
                if not st.session_state.vs_manager:
                    initialize_chatbot()
                    
                if st.session_state.vs_manager:
                    st.session_state.vs_manager.add_documents_to_vectorstore(documents)
                    st.success(f"Successfully uploaded {len(documents)} document chunks!")
                    
                    stats = st.session_state.vs_manager.get_index_stats()
                    st.info(f"Total vectors: {stats.get('total_vectors', 0)}")
                else:
                    st.error("Failed to initialize Chatbot. Please check your settings and API keys.")
                
            except Exception as e:
                st.error(f"Document processing error: {e}")


def display_chat_interface():
    """Render the main chat interface."""
    st.subheader("Chat with your Documents")
    
    # Display previous messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=msg.get("key"))
        else:
            message(msg["content"], is_user=False, key=msg.get("key"))
            
            if msg.get("sources"):
                with st.expander("Sources"):
                    for source in msg["sources"]:
                        st.markdown(f"""
                        <div class="source-badge">
                        {source['source']} (match: {source['score']:.2%})
                        </div>
                        """, unsafe_allow_html=True)
    
    # User input
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        if not st.session_state.chatbot:
            initialize_chatbot()
        
        if st.session_state.chatbot:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "key": f"user_{datetime.now().timestamp()}"
            })
            
            with st.spinner("Searching documents..."):
                result = st.session_state.chatbot.chat(user_input)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources"),
                    "key": f"assistant_{datetime.now().timestamp()}"
                })
            
            st.rerun()
        else:
            st.warning("Please initialize the Chatbot first from the sidebar")


def sidebar_controls():
    """Render sidebar controls and settings."""
    st.sidebar.markdown("## Settings & Controls")
    
    if st.session_state.initialized:
        st.sidebar.success("Chatbot Ready")
    else:
        st.sidebar.warning("Chatbot not initialized")
    
    if st.sidebar.button("Initialize Chatbot", use_container_width=True):
        initialize_chatbot()
    
    st.sidebar.markdown("---")
    
    if st.session_state.vs_manager:
        if st.sidebar.button("Show Database Stats"):
            stats = st.session_state.vs_manager.get_index_stats()
            st.sidebar.json(stats)
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_history()
        st.success("Conversation cleared")
        st.rerun()
    
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("Advanced Settings"):
        st.write("**Model Settings:**")
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, settings.TEMPERATURE, 0.1
        )
        top_k = st.slider(
            "Number of retrieved documents",
            1, 10, settings.TOP_K_DOCUMENTS
        )
        
        if st.button("Save Settings"):
            st.success("Settings saved")


def main():
    """Main application entry point."""
    st.title("Company Document Chatbot")
    st.markdown("*Ask questions about your company documents*")
    
    sidebar_controls()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_chat_interface()
    
    with col2:
        st.markdown("### System Info")
        st.info(f"""
        **Model:** {settings.LLM_MODEL}  
        **Database:** {settings.QDRANT_COLLECTION_NAME}  
        **Chunk Size:** {settings.CHUNK_SIZE}  
        **Documents:** Loaded from index
        """)
        
        st.markdown("---")
        
        st.markdown("### Add New Documents")
        
        if st.button("Click to upload files"):
            st.session_state.show_upload = True
        
        if st.session_state.get("show_upload", False):
            upload_documents()


if __name__ == "__main__":
    main()
