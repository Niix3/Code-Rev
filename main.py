"""FastAPI Gateway for Multi-Agent System."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, List, Literal, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from config import settings
from config.settings import TesterMode
from orchestrator import LangGraphOrchestrator
from orchestrator.job_manager import JobManager, JobStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent System API",
    description="LangGraph-based multi-agent orchestration system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = LangGraphOrchestrator()
job_manager = JobManager()


class QueryRequest(BaseModel):
    """Request model for text-only queries."""

    query: str
    workspace_path: Optional[str] = Field(
        default=None,
        description="Working directory for agents (defaults to WORKSPACE_PATH from settings).",
    )
    tester_mode: Optional[Literal["benchmarking", "test_generation"]] = Field(
        default=None,
        description="benchmarking: run existing repo tests; test_generation: author new tests.",
    )
    test_command: Optional[str] = Field(
        default=None,
        description="Override test command (required for SWE-bench instances in benchmarking mode).",
    )
    max_iterations: Optional[int] = None
    async_mode: bool = Field(
        default=True,
        description="If true, returns job_id immediately; poll GET /jobs/{job_id} for result.",
    )

    @field_validator("workspace_path")
    @classmethod
    def normalize_workspace_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.rstrip("/\\") or None


class QueryResponse(BaseModel):
    """Response model."""

    response: str
    agent_used: str
    verified: bool
    sources: List[str] = []
    metadata: dict = {}


class QuerySubmitResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    query: str
    current_stage: Optional[str] = None
    iteration: int = 0
    completed_stages: List[str] = []
    pipeline_trace: list = []
    error: Optional[str] = None
    created_at: str
    updated_at: str
    max_iterations: int = 10
    workspace_path: Optional[str] = None
    tester_mode: Optional[str] = None
    test_command: Optional[str] = None
    result: Optional[QueryResponse] = None


def resolve_workspace_path(requested: Optional[str]) -> str:
    return requested or settings.workspace_path


def validate_workspace_path(workspace_path: str) -> None:
    path = Path(workspace_path)
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Workspace path does not exist: {workspace_path}",
        )
    if not path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Workspace path is not a directory: {workspace_path}",
        )


def resolve_tester_mode(requested: Optional[str]) -> str:
    if requested is None:
        return settings.tester_mode.value
    return TesterMode(requested).value


def build_query_response(result: dict[str, Any]) -> QueryResponse:
    agent_response = result.get("agent_response", {})
    composer_verdict = result.get("composer_verdict", {}) or agent_response.get("composer_verdict", {})
    testing_output = result.get("testing_output", {})
    return QueryResponse(
        response=result.get("final_response", agent_response.get("response", "")),
        agent_used=agent_response.get("agent", "unknown"),
        verified=result.get("verified", False),
        sources=agent_response.get("sources", []),
        metadata={
            "iteration": result.get("iteration", 0),
            "max_iterations": result.get("max_iterations", settings.max_iterations),
            "tester_mode": result.get("tester_mode", settings.tester_mode.value),
            "test_command": result.get("test_command") or settings.tester_command,
            "composer_verdict": composer_verdict,
            "review_results": {
                "correctness": result.get("correctness_review", {}),
                "security": result.get("security_review", {}),
                "code_style": result.get("code_style_review", {}),
            },
            "all_responses": len(result.get("all_responses", [])),
            "pipeline_trace": result.get("pipeline_trace", []),
            "workspace_path": result.get("workspace_path", settings.workspace_path),
            "architecture_output": result.get("architecture_output", {}).get("response", ""),
            "test_authoring_output": result.get("test_authoring_output", {}).get("response", ""),
            "coding_output": result.get("coding_output", {}).get("response", ""),
            "testing_output": testing_output.get("response", ""),
            "testing_passed": testing_output.get("passed"),
            "testing_mode": testing_output.get("mode"),
            "test_authoring_run_id": result.get("test_authoring_output", {}).get("sdk_run_id"),
            "coding_run_id": result.get("coding_output", {}).get("sdk_run_id"),
            "testing_run_id": testing_output.get("sdk_run_id"),
            "sdk_status": {
                "write_tests": result.get("test_authoring_output", {}).get("sdk_status", "skipped"),
                "coding": result.get("coding_output", {}).get("sdk_status", "unknown"),
                "run_tests": testing_output.get("sdk_status", "unknown"),
            },
        },
    )


async def run_pipeline_job(
    job_id: str,
    query: str,
    max_iterations: int | None,
    workspace_path: str,
    tester_mode: str,
    test_command: str | None,
) -> None:
    """Execute pipeline in a worker thread so the event loop stays responsive."""
    job_manager.mark_running(job_id)
    progress_callback = job_manager.make_progress_callback(job_id)
    try:
        result = await asyncio.to_thread(
            orchestrator.invoke,
            query,
            workspace_path=workspace_path,
            max_iterations=max_iterations,
            tester_mode=tester_mode,
            test_command=test_command,
            progress_callback=progress_callback,
        )
        job_manager.complete(job_id, result)
        logger.info("[job] COMPLETED job_id=%s workspace=%s tester_mode=%s", job_id, workspace_path, tester_mode)
    except Exception as exc:
        logger.exception("[job] FAILED job_id=%s", job_id)
        job_manager.fail(job_id, str(exc))


def job_to_status_response(job_payload: dict[str, Any]) -> JobStatusResponse:
    result_payload = job_payload.get("result")
    return JobStatusResponse(
        job_id=job_payload["job_id"],
        status=job_payload["status"],
        query=job_payload["query"],
        current_stage=job_payload.get("current_stage"),
        iteration=job_payload.get("iteration", 0),
        completed_stages=job_payload.get("completed_stages", []),
        pipeline_trace=job_payload.get("pipeline_trace", []),
        error=job_payload.get("error"),
        created_at=job_payload["created_at"],
        updated_at=job_payload["updated_at"],
        max_iterations=job_payload.get("max_iterations", settings.max_iterations),
        workspace_path=job_payload.get("workspace_path"),
        tester_mode=job_payload.get("tester_mode"),
        test_command=job_payload.get("test_command"),
        result=build_query_response(result_payload) if result_payload else None,
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Agent System API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Submit query (async by default)",
            "/jobs/{job_id}": "GET - Job status and result",
            "/jobs": "GET - List recent jobs",
            "/health": "GET - Health check",
        },
        "defaults": {
            "workspace_path": settings.workspace_path,
            "tester_mode": settings.tester_mode.value,
            "tester_command": settings.tester_command,
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint — always non-blocking."""
    return {
        "status": "healthy",
        "components": {
            "orchestrator": "initialized",
        },
        "jobs": {
            "running": job_manager.count_by_status(JobStatus.RUNNING),
            "pending": job_manager.count_by_status(JobStatus.PENDING),
            "completed": job_manager.count_by_status(JobStatus.COMPLETED),
            "failed": job_manager.count_by_status(JobStatus.FAILED),
        },
    }


@app.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(limit: int = 20):
    """List recent pipeline jobs."""
    jobs = job_manager.list_jobs(limit=limit)
    return [job_to_status_response(payload) for payload in jobs]


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get pipeline job status; includes full result when completed."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job_to_status_response(job.to_dict())


@app.post("/query", response_model=QuerySubmitResponse | QueryResponse)
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Submit a query to the multi-agent pipeline.

    By default runs asynchronously: returns job_id immediately.
    Poll GET /jobs/{job_id} for progress and final result.
    Set async_mode=false to wait for completion (still non-blocking for /health).
    """
    workspace_path = resolve_workspace_path(request.workspace_path)
    validate_workspace_path(workspace_path)
    tester_mode = resolve_tester_mode(request.tester_mode)
    max_iterations = request.max_iterations or settings.max_iterations

    if tester_mode == TesterMode.BENCHMARKING.value and not request.test_command and settings.tester_command == "pytest -q":
        logger.warning(
            "[query] benchmarking mode without explicit test_command — "
            "set test_command to the repository-specific SWE-bench command"
        )

    if not request.async_mode:
        job = job_manager.create(
            request.query,
            max_iterations,
            workspace_path=workspace_path,
            tester_mode=tester_mode,
            test_command=request.test_command,
        )
        await run_pipeline_job(
            job.id,
            request.query,
            request.max_iterations,
            workspace_path,
            tester_mode,
            request.test_command,
        )
        completed = job_manager.get(job.id)
        if completed is None:
            raise HTTPException(status_code=500, detail="Job disappeared after execution")
        if completed.status == JobStatus.FAILED:
            raise HTTPException(status_code=500, detail=completed.error or "Pipeline failed")
        if completed.result is None:
            raise HTTPException(status_code=500, detail="Pipeline finished without result")
        return build_query_response(completed.result)

    job = job_manager.create(
        request.query,
        max_iterations,
        workspace_path=workspace_path,
        tester_mode=tester_mode,
        test_command=request.test_command,
    )
    background_tasks.add_task(
        run_pipeline_job,
        job.id,
        request.query,
        request.max_iterations,
        workspace_path,
        tester_mode,
        request.test_command,
    )
    status_url = f"/jobs/{job.id}"
    logger.info(
        "[job] CREATED job_id=%s async_mode=true workspace=%s tester_mode=%s",
        job.id,
        workspace_path,
        tester_mode,
    )
    return QuerySubmitResponse(
        job_id=job.id,
        status=JobStatus.PENDING.value,
        status_url=status_url,
        message="Pipeline started. Poll status_url for progress and result.",
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
