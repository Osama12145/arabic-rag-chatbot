"""
config.py - Configuration Management
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Centralized settings management using Pydantic
    """
    
    # API Keys
    OPENROUTER_API_KEY: str
    COHERE_API_KEY: str
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/company_chatbot"
    
    # Qdrant Vector DB
    QDRANT_URL: str = "local"  # Use "local" for file-based mode, or a remote URL for production
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "company-documents"
    EMBEDDING_MODEL: str = "embed-multilingual-v3.0"
    
    # LLM Configuration
    LLM_MODEL: str = "cohere/command-r-plus"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1500
    
    # RAG Configuration
    TOP_K_DOCUMENTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.1
    
    # Chunking Strategy
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # System Prompt
    SYSTEM_PROMPT: str = """You are a helpful company assistant.
    Answer questions ONLY based on the provided company documents.
    If you can't find the answer in the documents, say: "I couldn't find information about this topic in the company documents."
    NEVER make up information.
    Be concise, professional, and always respond in Arabic."""
    
    # App Settings
    APP_NAME: str = "Company Document Chatbot"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
