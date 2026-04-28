"""FastAPI Gateway for Multi-Agent System."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from orchestrator import LangGraphOrchestrator


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
orchestrator = LangGraphOrchestrator()


class QueryRequest(BaseModel):
    """Request model for text-only queries."""
    query: str
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
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "orchestrator": "initialized"
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
        verification = agent_response.get("verification", {}) or result.get("critic_verification", {})
        
        return QueryResponse(
            response=result.get("final_response", agent_response.get("response", "")),
            agent_used=agent_response.get("agent", "unknown"),
            verified=result.get("verified", False),
            sources=agent_response.get("sources", []),
            metadata={
                "routed_agent": result.get("routed_agent", ""),
                "verification_score": verification.get("score", 0),
                "critic_feedback": {
                    "verified": verification.get("verified", False),
                    "score": verification.get("score", 0),
                    "critique": verification.get("critique", ""),
                    "suggestions": verification.get("suggestions", ""),
                },
                "all_responses": len(result.get("all_responses", [])),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

