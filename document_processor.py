"""
document_processor.py - Document Processing Pipeline
Handles loading, splitting, and cleaning documents for the RAG system.
"""

from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pathlib import Path
from typing import List
import logging
from config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents by loading, splitting, and cleaning them.
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_documents(self, directory_path: str) -> List[Document]:
        """
        Load all documents from a directory.
        Supports: PDF, DOCX, TXT
        """
        documents = []
        path = Path(directory_path)
        
        logger.info(f"Loading documents from: {directory_path}")
        
        # Load PDF files
        for pdf_file in path.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                logger.info(f"Loaded: {pdf_file.name} ({len(docs)} pages)")
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {pdf_file.name}: {e}")
        
        # Load Word files
        for docx_file in path.glob("*.docx"):
            try:
                loader = Docx2txtLoader(str(docx_file))
                docs = loader.load()
                logger.info(f"Loaded: {docx_file.name}")
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {docx_file.name}: {e}")
        
        # Load text files
        for txt_file in path.glob("*.txt"):
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                docs = loader.load()
                logger.info(f"Loaded: {txt_file.name}")
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {txt_file.name}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split long documents into smaller chunks with overlap.
        """
        logger.info("Splitting documents into chunks...")
        
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "source": doc.metadata.get("source", "unknown")
                    }
                )
                all_chunks.append(chunk_doc)
        
        logger.info(f"Split into {len(all_chunks)} chunks")
        return all_chunks
    
    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean documents by removing extra whitespace and empty lines.
        """
        cleaned = []
        for doc in documents:
            cleaned_content = "\n".join(
                line.strip() for line in doc.page_content.split("\n") 
                if line.strip()
            )
            
            if len(cleaned_content) > 10:  # Skip very short chunks
                doc.page_content = cleaned_content
                cleaned.append(doc)
        
        logger.info(f"Cleaned {len(cleaned)} documents")
        return cleaned
    
    def process_documents(self, directory_path: str) -> List[Document]:
        """
        Main processing pipeline: load -> split -> clean
        """
        documents = self.load_documents(directory_path)
        chunks = self.split_documents(documents)
        cleaned = self.clean_documents(chunks)
        return cleaned


if __name__ == "__main__":
    processor = DocumentProcessor()
    docs = processor.process_documents("./documents")
    
    for i, doc in enumerate(docs[:3]):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:200])
        print(f"Source: {doc.metadata}")
