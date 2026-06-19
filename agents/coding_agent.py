"""Coding agent that delegates implementation to OpenHands Python SDK."""
import json
from typing import Any, Optional

from config import settings
from .openhands_sdk_runner import OpenHandsSDKRunner
from .review_utils import priority_paths_from_architecture, to_workspace_relative_path
from .workspace_diff import capture_coder_patch

_CODING_RULES = """\
CRITICAL implementation rules:
1. Follow the Architecture plan EXACTLY — treat plan steps as a mandatory checklist.
2. Modify ONLY files listed in the plan steps (field "file"). Do not touch other paths.
3. Edit existing production source files in place. Do NOT create parallel modules unless the plan requires it.
4. Do NOT modify or create: tests/, docs/, locale/, static/, translation (.po/.mo) files, assets, .editorconfig, .gitignore, or other repo config.
5. Do NOT run repository-wide reformatting, mass search-and-replace, or line-ending normalization.
6. Keep the fix minimal and focused on the user task.
7. After editing, the planned file(s) must show up in `git diff` as modifications to tracked source files.
"""


class CodingAgent:
    """Calls OpenHands SDK to perform coding work in shared workspace."""

    def __init__(self):
        self.sdk_runner = OpenHandsSDKRunner()

    @staticmethod
    def _format_architecture(architecture_output: dict[str, Any]) -> str:
        plan = architecture_output.get("plan")
        if plan:
            return json.dumps(plan, ensure_ascii=False, indent=2)
        return architecture_output.get("response", "")

    @staticmethod
    def _format_architect_checklist(architecture_output: dict[str, Any]) -> str:
        plan = architecture_output.get("plan") if isinstance(architecture_output.get("plan"), dict) else {}
        steps = plan.get("steps") or architecture_output.get("steps") or []
        explanation = plan.get("explanation") or architecture_output.get("explanation") or ""

        lines: list[str] = []
        if explanation:
            lines.append(f"Approach: {explanation}")
        if not steps:
            return "\n".join(lines) if lines else "(no structured steps in plan)"

        lines.append("Mandatory checklist (complete every step in order):")
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            file_path = step.get("file") or "(file not specified — infer from description)"
            description = step.get("description", "")
            lines.append(f"  {index}. FILE: {file_path}")
            lines.append(f"     ACTION: {description}")
        return "\n".join(lines)

    @staticmethod
    def _format_refinement_feedback(refinement_feedback: dict[str, Any]) -> str:
        instructions = refinement_feedback.get("refinement_instructions", [])
        blocking = refinement_feedback.get("blocking_issues", [])
        summary = refinement_feedback.get("summary", "")
        sections = [f"Summary: {summary}"] if summary else []
        if blocking:
            sections.append("Blocking issues:\n- " + "\n- ".join(blocking))
        if instructions:
            sections.append("Required fixes:\n- " + "\n- ".join(instructions))
        for dimension in ("correctness", "security", "code_style"):
            review = refinement_feedback.get(dimension, {})
            findings = review.get("findings", [])
            if findings:
                sections.append(f"{dimension} findings:\n- " + "\n- ".join(findings))
        return "\n\n".join(sections) if sections else json.dumps(refinement_feedback, ensure_ascii=False, indent=2)

    def execute(
        self,
        query: str,
        architecture_output: dict[str, Any],
        workspace_path: str,
        refinement_feedback: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Send coding task to OpenHands SDK."""
        architecture_text = self._format_architecture(architecture_output)
        checklist = self._format_architect_checklist(architecture_output)
        prompt = (
            f"Implement the requested change in workspace {workspace_path}.\n\n"
            f"{_CODING_RULES}\n\n"
            f"User task:\n{query}\n\n"
            f"Architecture plan (JSON — authoritative):\n{architecture_text}\n\n"
            f"Architecture checklist (you MUST follow this):\n{checklist}\n\n"
        )
        if refinement_feedback:
            prompt += (
                "This is a refinement pass. Address all review feedback below before finishing.\n"
                "Still follow the architecture checklist unless feedback explicitly overrides a step.\n\n"
                f"{self._format_refinement_feedback(refinement_feedback)}\n\n"
            )
        prompt += (
            "Before finishing, verify: (1) every checklist step is done, "
            "(2) only planned production source files were edited, "
            "(3) `git diff --name-only` shows the expected paths and nothing else."
        )
        run = self.sdk_runner.run(
            prompt=prompt,
            workspace_path=workspace_path,
            timeout_seconds=settings.openhands_timeout_seconds,
        )
        priority_paths = priority_paths_from_architecture(architecture_output, workspace_path)
        edited_files = [
            to_workspace_relative_path(path, workspace_path)
            for path in run.get("edited_files") or []
        ]
        edited_files = [path for path in edited_files if path]
        benchmarking = "/workspaces/" in workspace_path.replace("\\", "/")
        workspace_patch = ""
        if run["ok"]:
            workspace_patch = capture_coder_patch(
                workspace_path,
                edited_paths=run.get("edited_files") or [],
                priority_paths=priority_paths,
                benchmarking=benchmarking,
            )
        result = {
            "response": run["message"],
            "openhands_result": run["raw_result"],
            "workspace_path": workspace_path,
            "agent": "coding",
            "sdk_status": run["status"],
            "sdk_run_id": run["run_id"],
            "edited_files": edited_files,
            "workspace_patch": workspace_patch,
        }
        if not run["ok"]:
            result["error"] = run.get("error", "openhands_sdk_error")
        return result
