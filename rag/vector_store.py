"""Vector store implementation for RAG."""
from typing import List, Dict, Optional, Any
import numpy as np
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from config import settings


class VectorStore:
    """Vector database for knowledge retrieval."""
    
    def __init__(self, store_type: str = "faiss"):
        """
        Initialize vector store.
        
        Args:
            store_type: 'faiss' or 'qdrant'
        """
        self.store_type = store_type
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        self.store: Optional[Any] = None
    
    def initialize(self, documents: Optional[List[Document]] = None):
        """Initialize or load vector store."""
        if documents:
            if self.store_type == "faiss":
                self.store = FAISS.from_documents(documents, self.embeddings)
            elif self.store_type == "qdrant":
                try:
                    from qdrant_client import QdrantClient
                    from langchain.vectorstores import Qdrant
                    
                    client = QdrantClient(
                        host=settings.qdrant_host,
                        port=settings.qdrant_port
                    )
                    self.store = Qdrant.from_documents(
                        documents,
                        self.embeddings,
                        url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
                        collection_name="knowledge_base"
                    )
                except Exception as e:
                    print(f"Qdrant initialization failed: {e}, falling back to FAISS")
                    self.store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Load existing store
            if self.store_type == "faiss":
                try:
                    self.store = FAISS.load_local("vector_store", self.embeddings)
                except:
                    self.store = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store."""
        if self.store is None:
            self.initialize(documents)
        else:
            if self.store_type == "faiss":
                self.store.add_documents(documents)
            else:
                self.store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents."""
        if self.store is None:
            return []
        return self.store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search with similarity scores."""
        if self.store is None:
            return []
        return self.store.similarity_search_with_score(query, k=k)
    
    def save(self, path: str = "vector_store"):
        """Save vector store to disk."""
        if self.store and self.store_type == "faiss":
            self.store.save_local(path)

