"""Query processing and rewriting for RAG."""
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import settings


class QueryRewriter:
    """Rewrites queries for better retrieval."""
    
    def __init__(self):
        """Initialize query rewriter."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0.3,
            api_key=settings.openai_api_key
        )
    
    def rewrite_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Rewrite query to improve retrieval.
        
        Args:
            query: Original query
            context: Optional context for rewriting
            
        Returns:
            Rewritten query
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query rewriting expert. Rewrite the user's query 
            to improve information retrieval. Make it more specific, add relevant keywords,
            and clarify the intent while preserving the original meaning."""),
            ("user", "Original query: {query}\n{context}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "context": context or "No additional context."
        })
        
        return response.content


class ContextCompressor:
    """Compresses retrieved context to fit within token limits."""
    
    def __init__(self, max_tokens: int = 2000):
        """Initialize context compressor."""
        self.max_tokens = max_tokens
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
    
    def compress(self, documents: List[str], query: str) -> str:
        """
        Compress context documents.
        
        Args:
            documents: List of document texts
            context: Original query
            
        Returns:
            Compressed context
        """
        combined = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(documents)])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Compress the following documents to extract only information 
            relevant to the query. Remove redundancy and keep only essential facts."""),
            ("user", "Query: {query}\n\nDocuments:\n{documents}\n\nCompressed context:")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "documents": combined
        })
        
        return response.content


class ReRanker:
    """Re-ranks retrieved documents by relevance."""
    
    def __init__(self):
        """Initialize re-ranker."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
    
    def rerank(self, query: str, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Re-rank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of documents with 'content' and 'score' keys
            
        Returns:
            Re-ranked documents
        """
        if len(documents) <= 1:
            return documents
        
        # Create ranking prompt
        doc_text = "\n".join([
            f"[{i+1}] {doc.get('content', doc.get('text', ''))[:200]}..."
            for i, doc in enumerate(documents)
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Rank the following documents by relevance to the query.
            Return only the numbers in order of relevance (most relevant first)."""),
            ("user", "Query: {query}\n\nDocuments:\n{documents}\n\nRanking:")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "documents": doc_text
        })
        
        # Parse ranking (simplified - in production use proper parsing)
        # For now, return original order
        return documents

