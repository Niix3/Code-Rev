"""Shared OpenHands SDK runner for coding and testing stages."""
from __future__ import annotations

from typing import Any, Dict, Optional

from config import settings

from .openhands_edits import extract_edited_files

_SDK_IMPORT_ERROR: str | None = None
try:
    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool
except Exception as exc:  # pragma: no cover - handled at runtime
    _SDK_IMPORT_ERROR = str(exc)
    LLM = Agent = Conversation = Tool = None  # type: ignore[assignment]
    FileEditorTool = TaskTrackerTool = TerminalTool = None  # type: ignore[assignment]


class OpenHandsSDKRunner:
    """Runs OpenHands SDK conversations in a local workspace."""

    @staticmethod
    def _sdk_available() -> bool:
        return all([LLM, Agent, Conversation, Tool, FileEditorTool, TaskTrackerTool, TerminalTool])

    @staticmethod
    def _extract_response_text(run_result: Any) -> str:
        if isinstance(run_result, str):
            return run_result
        if isinstance(run_result, dict):
            for key in ("summary", "message", "result", "output"):
                if run_result.get(key):
                    return str(run_result[key])
        return "OpenHands SDK run completed."

    def run(self, prompt: str, workspace_path: str, timeout_seconds: int) -> Dict[str, Any]:
        """Execute prompt with OpenHands SDK and return normalized response."""
        if not self._sdk_available():
            detail = _SDK_IMPORT_ERROR or "import failed"
            return {
                "ok": False,
                "status": "error",
                "run_id": None,
                "message": (
                    "OpenHands SDK is not usable in this environment. "
                    f"Import error: {detail}. "
                    "Install matching versions, e.g. openhands-sdk==1.20.0 and openhands-tools==1.20.0."
                ),
                "raw_result": {},
                "error": "missing_openhands_sdk",
            }

        try:
            llm = LLM(
                model=settings.openhands_model,
                api_key=settings.openhands_api_key or settings.openai_api_key,
                base_url=settings.openhands_llm_base_url or settings.openai_base_url,
                timeout=timeout_seconds,
            )
            agent = Agent(
                llm=llm,
                tools=[
                    Tool(name=TerminalTool.name),
                    Tool(name=FileEditorTool.name),
                    Tool(name=TaskTrackerTool.name),
                ],
            )
            conversation = Conversation(agent=agent, workspace=workspace_path)
            conversation.send_message(prompt)
            run_result = conversation.run()
            run_id = getattr(conversation, "id", None) or getattr(run_result, "id", None)
            edited_files = extract_edited_files(conversation)
            return {
                "ok": True,
                "status": "finished",
                "run_id": run_id,
                "message": self._extract_response_text(run_result),
                "raw_result": run_result if isinstance(run_result, dict) else {"result": str(run_result)},
                "edited_files": edited_files,
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "run_id": None,
                "message": f"OpenHands SDK run failed: {str(exc)}",
                "raw_result": {},
                "error": str(exc),
            }
