"""
api_server.py - FastAPI Backend
Run with: uvicorn api_server:app --reload
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import logging
import asyncio
from datetime import datetime

from rag_pipeline import RAGChatbot
from vector_store import VectorStoreManager
from document_processor import DocumentProcessor
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Company Document Chatbot API",
    description="API for interacting with company documents via RAG",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vs_manager: Optional[VectorStoreManager] = None
chatbot: Optional[RAGChatbot] = None


# ============= Pydantic Models =============

class ChatRequest(BaseModel):
    """Chat request payload."""
    query: str
    include_sources: bool = True
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response payload."""
    answer: str
    sources: Optional[List[dict]] = None
    context_found: bool
    timestamp: str
    conversation_id: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    message: str
    documents_processed: int
    success: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    initialized: bool
    total_vectors: Optional[int] = None


# ============= Startup & Shutdown =============

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global vs_manager, chatbot
    
    logger.info("Starting application...")
    
    try:
        vs_manager = VectorStoreManager()
        chatbot = RAGChatbot(vs_manager)
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application...")


# ============= Health Check =============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health status."""
    try:
        stats = vs_manager.get_index_stats() if vs_manager else {}
        
        return HealthResponse(
            status="healthy" if chatbot else "initializing",
            initialized=chatbot is not None,
            total_vectors=stats.get("total_vectors")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Chat Endpoints =============

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot.
    
    Example:
    ```json
    {
        "query": "What is the leave policy?",
        "include_sources": true
    }
    ```
    """
    if not chatbot:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized. Please wait..."
        )
    
    try:
        logger.info(f"New query: {request.query}")
        
        result = chatbot.chat(
            user_query=request.query,
            include_sources=request.include_sources
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources"),
            context_found=result.get("context_found", False),
            timestamp=result.get("timestamp"),
            conversation_id=request.conversation_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history")
async def get_chat_history():
    """Get conversation history."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "history": chatbot.get_conversation_summary(),
        "message_count": len(chatbot.conversation_history)
    }


@app.post("/api/chat/clear")
async def clear_history():
    """Clear conversation history."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    chatbot.clear_history()
    return {"message": "History cleared"}


# ============= Document Management =============

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload new documents for processing."""
    if not vs_manager:
        raise HTTPException(status_code=503, detail="Vector Store not initialized")
    
    try:
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        saved_files = []
        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(await file.read())
            saved_files.append(file_path)
        
        logger.info(f"Saved {len(saved_files)} files")
        
        # Process in background if available
        if background_tasks:
            background_tasks.add_task(
                process_documents_background,
                str(temp_dir)
            )
            
            return DocumentUploadResponse(
                message="Processing documents in the background...",
                documents_processed=0,
                success=True
            )
        else:
            # Process immediately
            processor = DocumentProcessor()
            documents = processor.process_documents(str(temp_dir))
            
            vs_manager.add_documents_to_vectorstore(documents)
            
            return DocumentUploadResponse(
                message=f"Successfully processed {len(documents)} document chunks",
                documents_processed=len(documents),
                success=True
            )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/stats")
async def get_document_stats():
    """Get document index statistics."""
    if not vs_manager:
        raise HTTPException(status_code=503, detail="Vector Store not initialized")
    
    stats = vs_manager.get_index_stats()
    return {
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        **stats
    }


@app.post("/api/documents/clear")
async def clear_all_documents():
    """
    Delete all documents from the collection.
    WARNING: This is a destructive operation!
    """
    if not vs_manager:
        raise HTTPException(status_code=503, detail="Vector Store not initialized")
    
    success = vs_manager.delete_all_documents()
    
    if success:
        return {"message": "All documents deleted"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete documents"
        )


# ============= Search Endpoint =============

@app.post("/api/search")
async def search(query: str, top_k: int = 5):
    """Search documents directly."""
    if not vs_manager:
        raise HTTPException(status_code=503, detail="Vector Store not initialized")
    
    try:
        results = vs_manager.search_documents(query, top_k=top_k)
        
        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "source": doc.metadata.get("source"),
                    "score": score,
                    "preview": doc.page_content[:200]
                }
                for doc, score in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Helper Functions =============

async def process_documents_background(directory: str):
    """Process documents in the background."""
    try:
        processor = DocumentProcessor()
        documents = processor.process_documents(directory)
        
        if vs_manager:
            vs_manager.add_documents_to_vectorstore(documents)
            
        logger.info(f"Background processing complete: {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Background processing error: {e}")


# ============= Root Endpoint =============

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Company Document Chatbot API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "status": "healthy" if chatbot else "initializing"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
