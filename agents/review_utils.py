"""Shared helpers for structured LLM reviews."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from config.settings import TesterMode

from .workspace_context import collect_workspace_context
from .workspace_diff import collect_workspace_diff, diff_is_empty, list_changed_files

T = TypeVar("T", bound=BaseModel)


def create_review_client() -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def to_workspace_relative_path(file_path: str, workspace_path: str) -> str:
    """Normalize architect/coder paths to repo-relative posix paths."""
    normalized = file_path.replace("\\", "/").lstrip("./")
    workspace = workspace_path.rstrip("/").replace("\\", "/")
    if normalized.startswith(workspace + "/"):
        return normalized[len(workspace) + 1 :]
    workspace_name = Path(workspace).name
    marker = f"{workspace_name}/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def priority_paths_from_architecture(
    architecture_output: dict[str, Any] | None,
    workspace_path: str,
) -> set[str]:
    """Collect file paths from the architect plan for review prioritization."""
    if not architecture_output:
        return set()

    paths: set[str] = set()
    plan = architecture_output.get("plan")
    if isinstance(plan, dict):
        for step in plan.get("steps") or []:
            if not isinstance(step, dict):
                continue
            file_path = step.get("file")
            if file_path:
                paths.add(to_workspace_relative_path(str(file_path), workspace_path))
    return paths


def build_coder_changes_section(
    workspace_path: str,
    coding_output: dict[str, Any],
    *,
    architect_paths: set[str],
    benchmarking: bool = False,
) -> str:
    """Prefer patch captured right after coding; fall back to live git diff."""
    sections: list[str] = []
    edited_files = coding_output.get("edited_files") or []
    if edited_files:
        sections.append(
            "OpenHands edited files (from SDK event log):\n"
            + "\n".join(f"  - {path}" for path in edited_files)
        )

    captured_patch = (coding_output.get("workspace_patch") or "").strip()
    live_diff = collect_workspace_diff(
        workspace_path,
        priority_paths=architect_paths,
        benchmarking=benchmarking,
    )

    if captured_patch:
        sections.append(f"=== Coder patch (captured immediately after coding) ===\n{captured_patch}")
    if live_diff.strip() and (not captured_patch or not diff_is_empty(live_diff)):
        sections.append(f"=== Live git diff ===\n{live_diff}")
    elif not captured_patch:
        sections.append(f"=== Live git diff ===\n{live_diff}")

    return "\n\n".join(sections)


def build_review_context(
    query: str,
    coding_output: dict[str, Any],
    testing_output: dict[str, Any],
    workspace_path: str,
    *,
    test_authoring_output: dict[str, Any] | None = None,
    architecture_output: dict[str, Any] | None = None,
    benchmarking: bool = False,
    tester_mode: str | None = None,
) -> str:
    """Build shared context block for all reviewers."""
    test_authoring = test_authoring_output or {}
    if tester_mode == TesterMode.BENCHMARKING.value:
        benchmarking = True
    test_authoring_section = (
        "Test authoring: skipped (benchmarking mode — using pre-existing repository tests).\n\n"
        if benchmarking
        else f"Test authoring summary:\n{test_authoring.get('response', testing_output.get('test_authoring_context', ''))}\n\n"
    )
    test_run_section = (
        "Test run: skipped in pipeline (benchmarking mode — verified externally by SWE-bench harness).\n\n"
        if benchmarking
        else (
            f"Test mode: {testing_output.get('mode', 'unknown')}\n"
            f"Test command: {testing_output.get('command', settings.tester_command)}\n"
            f"Tests passed: {testing_output.get('passed', False)}\n"
            f"Test run output:\n{testing_output.get('response', '')}\n\n"
        )
    )
    review_hints = "\n".join(
        part
        for part in (
            coding_output.get("response", ""),
            testing_output.get("response", ""),
            test_authoring.get("response", ""),
        )
        if part
    )
    architect_paths = priority_paths_from_architecture(architecture_output, workspace_path)
    changed_paths = set(
        list_changed_files(workspace_path, review_only=True, benchmarking=benchmarking)
    )
    priority_paths = architect_paths | changed_paths

    architect_section = ""
    if architecture_output:
        plan = architecture_output.get("plan")
        if isinstance(plan, dict):
            steps = plan.get("steps") or []
            if steps:
                lines = []
                for index, step in enumerate(steps, start=1):
                    if not isinstance(step, dict):
                        continue
                    file_path = step.get("file") or "(unspecified)"
                    description = step.get("description", "")
                    lines.append(f"  {index}. {file_path}: {description}")
                if lines:
                    architect_section = "Architect plan steps:\n" + "\n".join(lines) + "\n\n"

    workspace_diff = build_coder_changes_section(
        workspace_path,
        coding_output,
        architect_paths=architect_paths,
        benchmarking=benchmarking,
    )
    workspace_context = collect_workspace_context(
        workspace_path,
        query=query,
        extra_hints=review_hints,
        priority_paths=priority_paths,
        max_files=15 if benchmarking else 25,
        max_chars=30_000 if benchmarking else 60_000,
    )
    return (
        f"User request:\n{query}\n\n"
        f"{architect_section}"
        f"{test_authoring_section}"
        f"Coding stage summary:\n{coding_output.get('response', '')}\n\n"
        f"{test_run_section}"
        f"Workspace path: {workspace_path}\n\n"
        f"=== Coder changes (primary review source) ===\n{workspace_diff}\n\n"
        f"Workspace context:\n{workspace_context}"
    )


def structured_completion(
    client: OpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    temperature: float = 0.1,
) -> T:
    """Request a Pydantic-validated structured response from the LLM."""
    try:
        completion = client.beta.chat.completions.parse(
            model=settings.default_llm_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_model,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed
    except Exception:
        pass

    schema_hint = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
    completion = client.chat.completions.create(
        model=settings.default_llm_model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "Respond with JSON only. Match this schema:\n"
                    f"{schema_hint}"
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    return response_model.model_validate_json(raw)

