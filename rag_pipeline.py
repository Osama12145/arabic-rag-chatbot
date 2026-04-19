"""
rag_pipeline.py - RAG Pipeline
Combines retrieval and generation for document-based Q&A.
"""

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from typing import List, Tuple
import logging
from datetime import datetime
from config import settings
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    RAG-based Chatbot that answers questions using company documents.
    """
    
    def __init__(self, vs_manager: VectorStoreManager = None):
        """
        Initialize the chatbot with a vector store manager and LLM.
        """
        self.vs_manager = vs_manager or VectorStoreManager()
        
        # Initialize LLM via OpenRouter
        self.llm = ChatOpenAI(
            model_name=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        self.conversation_history = []
        
        # Prompt template for RAG
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful company assistant.
Use the information provided in the context below to answer the question.

Context (from company documents):
{context}

User question:
{question}

Instructions:
- If the answer is in the context, respond based on it directly
- If you can't find an answer, say: "Sorry, I couldn't find information about this topic in the company documents"
- Be concise and clear
- Always respond in Arabic"""
        )
        
        logger.info("RAG Chatbot initialized")
    
    def retrieve_context(self, query: str) -> Tuple[str, List[dict]]:
        """
        Retrieve relevant context from the document store.
        """
        logger.info(f"Searching for: {query}")
        
        search_results = self.vs_manager.search_documents(query)
        
        if not search_results:
            logger.warning("No relevant documents found")
            return "", []
        
        context_parts = []
        sources = []
        
        for doc, score in search_results:
            context_parts.append(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}")
            sources.append({
                "source": doc.metadata.get('source', 'Unknown'),
                "score": float(score),
                "content_preview": doc.page_content[:100]
            })
        
        context = "\n---\n".join(context_parts)
        
        logger.info(f"Found {len(sources)} relevant documents")
        return context, sources
    
    def build_messages(
        self, 
        question: str, 
        context: str
    ) -> List:
        """
        Build the message list for the LLM including system prompt,
        conversation history, and the current question with context.
        """
        messages = []
        
        # System message
        messages.append(SystemMessage(content=settings.SYSTEM_PROMPT))
        
        # Recent conversation history (last 5 exchanges)
        recent_history = self.conversation_history[-10:]
        messages.extend(recent_history)
        
        # Current question with context
        formatted_question = self.prompt_template.format(
            context=context,
            question=question
        )
        
        messages.append(HumanMessage(content=formatted_question))
        
        return messages
    
    def generate_response(self, messages: List) -> str:
        """
        Generate a response using the LLM.
        """
        try:
            logger.info("Generating response...")
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            logger.info("Response generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "Sorry, an error occurred while processing your request. Please try again later."
    
    def chat(
        self, 
        user_query: str,
        include_sources: bool = True
    ) -> dict:
        """
        Main method: process user query and return the answer with sources.
        """
        logger.info(f"User query: {user_query}")
        
        try:
            # 1. Retrieve context
            context, sources = self.retrieve_context(user_query)
            
            # 2. Build messages
            messages = self.build_messages(user_query, context)
            
            # 3. Generate response
            response = self.generate_response(messages)
            
            # 4. Save to history
            self.conversation_history.append(HumanMessage(content=user_query))
            self.conversation_history.append(AIMessage(content=response))
            
            # 5. Build result
            result = {
                "answer": response,
                "sources": sources if include_sources else None,
                "timestamp": datetime.now().isoformat(),
                "context_found": len(sources) > 0
            }
            
            logger.info("Query answered successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "Sorry, an error occurred while processing your request.",
                "sources": None,
                "error": str(e)
            }
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        summary = "Conversation Summary:\n"
        for i, msg in enumerate(self.conversation_history):
            if isinstance(msg, HumanMessage):
                summary += f"\nUser: {msg.content[:100]}...\n"
            elif isinstance(msg, AIMessage):
                summary += f"Bot: {msg.content[:100]}...\n"
        return summary


if __name__ == "__main__":
    import os
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    documents = processor.process_documents("./documents")
    
    vs_manager = VectorStoreManager()
    vs_manager.add_documents_to_vectorstore(documents)
    
    chatbot = RAGChatbot(vs_manager)
    
    queries = [
        "What is the annual leave policy?",
        "How do I request sick leave?",
        "What is my salary?"
    ]
    
    for query in queries:
        result = chatbot.chat(query)
        print(f"\nQ: {query}")
        print(f"A: {result['answer']}")
        if result['sources']:
            print(f"Sources: {len(result['sources'])} documents")
