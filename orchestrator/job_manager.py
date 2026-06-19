"""In-memory job tracking for async pipeline execution."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    query: str
    status: JobStatus
    current_stage: str | None = None
    iteration: int = 0
    completed_stages: list[str] = field(default_factory=list)
    pipeline_trace: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_iterations: int = 10
    workspace_path: str | None = None
    tester_mode: str | None = None
    test_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "query": self.query,
            "current_stage": self.current_stage,
            "iteration": self.iteration,
            "completed_stages": list(self.completed_stages),
            "pipeline_trace": list(self.pipeline_trace),
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "max_iterations": self.max_iterations,
            "workspace_path": self.workspace_path,
            "tester_mode": self.tester_mode,
            "test_command": self.test_command,
            "result": self.result,
        }


ProgressCallback = Callable[[str, str, dict[str, Any]], None]


class JobManager:
    """Thread-safe store for pipeline jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(
        self,
        query: str,
        max_iterations: int,
        *,
        workspace_path: str | None = None,
        tester_mode: str | None = None,
        test_command: str | None = None,
    ) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            query=query,
            status=JobStatus.PENDING,
            max_iterations=max_iterations,
            workspace_path=workspace_path,
            tester_mode=tester_mode,
            test_command=test_command,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return Job(
                id=job.id,
                query=job.query,
                status=job.status,
                current_stage=job.current_stage,
                iteration=job.iteration,
                completed_stages=list(job.completed_stages),
                pipeline_trace=list(job.pipeline_trace),
                result=job.result,
                error=job.error,
                created_at=job.created_at,
                updated_at=job.updated_at,
                max_iterations=job.max_iterations,
                workspace_path=job.workspace_path,
                tester_mode=job.tester_mode,
                test_command=job.test_command,
            )

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.updated_at = datetime.now(timezone.utc)

    def make_progress_callback(self, job_id: str) -> ProgressCallback:
        def callback(event: str, stage: str, state: dict[str, Any]) -> None:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job.updated_at = datetime.now(timezone.utc)
                job.iteration = state.get("iteration", job.iteration)

                if event == "start":
                    job.status = JobStatus.RUNNING
                    job.current_stage = stage
                elif event == "done":
                    if stage not in job.completed_stages:
                        job.completed_stages.append(stage)
                    trace = state.get("_last_trace")
                    if trace:
                        job.pipeline_trace.append(trace)
                elif event == "failed":
                    job.current_stage = stage
                    job.error = state.get("_error", "stage failed")

        return callback

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.COMPLETED
            job.current_stage = None
            job.result = result
            job.pipeline_trace = result.get("pipeline_trace", job.pipeline_trace)
            job.iteration = result.get("iteration", job.iteration)
            job.updated_at = datetime.now(timezone.utc)

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.FAILED
            job.error = error
            job.updated_at = datetime.now(timezone.utc)

    def count_by_status(self, status: JobStatus) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status == status)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
            return [job.to_dict() for job in jobs[:limit]]
