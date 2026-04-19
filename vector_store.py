"""
vector_store.py - Qdrant Vector Database Manager
"""

from langchain_cohere import CohereEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.schema import Document
from typing import List, Tuple
import logging
import uuid
from config import settings

logger = logging.getLogger(__name__)

# Vector dimension for Cohere embed-multilingual-v3.0
VECTOR_SIZE = 1024


class VectorStoreManager:
    """
    Manages the Qdrant vector database for document storage and retrieval.
    """
    
    def __init__(self):
        """
        Initialize Qdrant client and Cohere embeddings.
        """
        self.embeddings = CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
        
        # Connect to Qdrant (local file-based or remote server)
        if settings.QDRANT_URL == "local":
            self.client = QdrantClient(path="./qdrant_local_db")
        else:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
            )
        
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection()
        
        logger.info("Connected to Qdrant successfully")
    
    def _ensure_collection(self):
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    
    def add_documents_to_vectorstore(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector database.
        """
        try:
            logger.info(f"Adding {len(documents)} documents to Qdrant...")
            
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate embeddings via Cohere
            vectors = self.embeddings.embed_documents(texts)
            
            # Build Qdrant points
            points = []
            for i, (vector, text, meta) in enumerate(zip(vectors, texts, metadatas)):
                point_id = str(uuid.uuid4())
                payload = {
                    "page_content": text,
                    "metadata": meta
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch {i // batch_size + 1}")
            
            logger.info("All documents added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search_documents(
        self, 
        query: str, 
        top_k: int = None,
        threshold: float = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for the most similar documents to a given query.
        
        Returns:
            List of (Document, similarity_score) tuples.
        """
        if top_k is None:
            top_k = settings.TOP_K_DOCUMENTS
        if threshold is None:
            threshold = settings.SIMILARITY_THRESHOLD
        
        try:
            logger.info(f"Searching for: '{query}'")
            
            query_vector = self.embeddings.embed_query(query)
            
            results = self.client.query_points(
                collection_name=self.collection_name, 
                query=query_vector,
                limit=top_k
            )
            
            # Filter results by similarity threshold
            filtered_results = []
            for point in results.points:
                score = point.score
                if score >= threshold:
                    doc = Document(
                        page_content=point.payload.get("page_content", ""),
                        metadata=point.payload.get("metadata", {})
                    )
                    filtered_results.append((doc, score))
            
            logger.info(f"Found {len(filtered_results)} relevant documents")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def delete_all_documents(self) -> bool:
        """
        Delete all documents from the collection and recreate it.
        """
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info("All documents deleted, collection recreated")
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
    
    def get_index_stats(self) -> dict:
        """
        Get collection statistics.
        """
        try:
            stats = self.client.get_collection(self.collection_name)
            return {
                "total_vectors": stats.points_count,
                "status": str(stats.status)
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}


if __name__ == "__main__":
    vs_manager = VectorStoreManager()
    
    results = vs_manager.search_documents("What is the leave policy?")
    
    for doc, score in results:
        print(f"\n--- Score: {score:.2f} ---")
        print(doc.page_content[:300])
