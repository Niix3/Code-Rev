"""FastAPI Gateway for Multi-Agent System."""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from PIL import Image
import io
from orchestrator import LangGraphOrchestrator
from rag import VectorStore
from config import settings


app = FastAPI(
    title="Multi-Agent System API",
    description="LangGraph-based multi-agent orchestration system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vector_store = VectorStore()
orchestrator = LangGraphOrchestrator(vector_store)


class QueryRequest(BaseModel):
    """Request model for text-only queries."""
    query: str
    use_rag: bool = True
    max_iterations: Optional[int] = None


class QueryResponse(BaseModel):
    """Response model."""
    response: str
    agent_used: str
    verified: bool
    sources: List[str] = []
    metadata: dict = {}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Agent System API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Text query",
            "/query-multimodal": "POST - Query with image",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "orchestrator": "initialized",
            "vector_store": "initialized" if vector_store.store else "not_loaded"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a text query through the multi-agent system.
    
    Args:
        request: Query request with text query
        
    Returns:
        QueryResponse with agent response
    """
    try:
        result = orchestrator.invoke(request.query)
        
        agent_response = result.get("agent_response", {})
        
        return QueryResponse(
            response=result.get("final_response", agent_response.get("response", "")),
            agent_used=agent_response.get("agent", "unknown"),
            verified=result.get("verified", False),
            sources=agent_response.get("sources", []),
            metadata={
                "routed_agent": result.get("routed_agent", ""),
                "verification_score": agent_response.get("verification", {}).get("score", 0),
                "all_responses": len(result.get("all_responses", []))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query-multimodal", response_model=QueryResponse)
async def query_multimodal(
    query: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Process a multimodal query (text + image) through the multi-agent system.
    
    Args:
        query: Text query
        image: Image file
        
    Returns:
        QueryResponse with agent response
    """
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Ensure RGB format
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        result = orchestrator.invoke(query, image=pil_image)
        
        agent_response = result.get("agent_response", {})
        
        return QueryResponse(
            response=result.get("final_response", agent_response.get("response", "")),
            agent_used=agent_response.get("agent", "vision"),
            verified=result.get("verified", False),
            sources=agent_response.get("sources", []),
            metadata={
                "routed_agent": result.get("routed_agent", ""),
                "has_image": True,
                "image_format": image.content_type,
                "verification_score": agent_response.get("verification", {}).get("score", 0)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing multimodal query: {str(e)}")


@app.post("/rag/add-documents")
async def add_documents(documents: List[str]):
    """
    Add documents to the RAG vector store.
    
    Args:
        documents: List of document texts
        
    Returns:
        Success message
    """
    try:
        from langchain.schema import Document
        
        docs = [Document(page_content=doc) for doc in documents]
        vector_store.add_documents(docs)
        vector_store.save()
        
        return {
            "message": f"Added {len(documents)} documents to vector store",
            "total_documents": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

