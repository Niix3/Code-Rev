"""Retrieval agent for RAG-based knowledge retrieval."""
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from rag import VectorStore, QueryRewriter, ContextCompressor, ReRanker
from config import settings


class RetrievalAgent:
    """Handles knowledge retrieval using RAG."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """Initialize retrieval agent."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
        self.vector_store = vector_store or VectorStore()
        self.query_rewriter = QueryRewriter()
        self.context_compressor = ContextCompressor()
        self.reranker = ReRanker()
        
        # Initialize vector store if not already done
        if self.vector_store.store is None:
            self.vector_store.initialize()
    
    def retrieve(self, query: str, k: int = 5, use_reranking: bool = True) -> Dict[str, Any]:
        """
        Retrieve and synthesize information.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            use_reranking: Whether to use re-ranking
            
        Returns:
            Dict with 'response', 'sources', 'retrieved_docs'
        """
        # Step 1: Query rewriting
        rewritten_query = self.query_rewriter.rewrite_query(query)
        
        # Step 2: Retrieval
        documents = self.vector_store.similarity_search(rewritten_query, k=k*2)  # Get more for reranking
        
        if not documents:
            # No documents found, use LLM directly
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's question."),
                ("user", "{query}")
            ])
            chain = prompt | self.llm
            response = chain.invoke({"query": query})
            
            return {
                "response": response.content,
                "sources": [],
                "retrieved_docs": [],
                "agent": "retrieval"
            }
        
        # Step 3: Re-ranking (optional)
        if use_reranking and len(documents) > 1:
            doc_dicts = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in documents
            ]
            reranked = self.reranker.rerank(query, doc_dicts)
            documents = reranked[:k]
        else:
            documents = documents[:k]
        
        # Step 4: Context compression
        doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        compressed_context = self.context_compressor.compress(doc_texts, query)
        
        # Step 5: Generate response with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledge assistant. Answer the user's question using 
            the provided context. Cite sources when possible. If the context doesn't contain 
            relevant information, say so."""),
            ("user", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "context": compressed_context
        })
        
        # Extract sources
        sources = [
            doc.metadata.get("source", "Unknown") 
            if hasattr(doc, 'metadata') else "Unknown"
            for doc in documents
        ]
        
        return {
            "response": response.content,
            "sources": sources,
            "retrieved_docs": doc_texts[:3],  # Return first 3 for reference
            "agent": "retrieval"
        }

