"""Tester agent: benchmark existing tests or generate new ones."""
from typing import Any, Optional

from config import settings
from config.settings import TesterMode
from .openhands_sdk_runner import OpenHandsSDKRunner


class TesterAgent:
    """Runs tests in benchmarking mode or authors tests in test_generation mode."""

    def __init__(self):
        self.sdk_runner = OpenHandsSDKRunner()

    @staticmethod
    def _resolve_mode(tester_mode: Optional[str | TesterMode]) -> TesterMode:
        if tester_mode is None:
            return settings.tester_mode
        if isinstance(tester_mode, TesterMode):
            return tester_mode
        return TesterMode(tester_mode)

    @staticmethod
    def _resolve_command(test_command: Optional[str]) -> str:
        return test_command or settings.tester_command

    def write_tests(
        self,
        query: str,
        architecture_output: dict[str, Any],
        coding_output: dict[str, Any],
        workspace_path: str,
    ) -> dict[str, Any]:
        """Create tests in the workspace after implementation exists (test_generation only)."""
        architecture_text = architecture_output.get("response", "")
        coding_summary = coding_output.get("response", "")
        prompt = (
            f"Write automated tests in workspace {workspace_path} for the existing implementation.\n\n"
            f"User task:\n{query}\n\n"
            f"Architecture plan:\n{architecture_text}\n\n"
            f"Coding stage summary:\n{coding_summary}\n\n"
            "Requirements:\n"
            "- Review the existing code in the workspace before writing tests.\n"
            "- Use pytest conventions (test_*.py or tests/ directory).\n"
            f"- Tests should be runnable with: {settings.tester_command}\n"
            "- Cover the user request with meaningful assertions and edge cases.\n"
            "- Do not rewrite production code unless a minimal fix is required for testability.\n"
            "- Summarize which test files were created and what they verify."
        )
        run = self.sdk_runner.run(
            prompt=prompt,
            workspace_path=workspace_path,
            timeout_seconds=settings.tester_timeout_seconds,
        )
        result = {
            "response": run["message"],
            "phase": "write_tests",
            "mode": TesterMode.TEST_GENERATION.value,
            "command": settings.tester_command,
            "workspace_path": workspace_path,
            "architecture_context": architecture_text,
            "coding_context": coding_summary,
            "agent": "tester",
            "sdk_status": run["status"],
            "sdk_run_id": run["run_id"],
            "openhands_result": run["raw_result"],
        }
        if not run["ok"]:
            result["error"] = run.get("error", "openhands_sdk_error")
        return result

    def run_benchmarking_tests(
        self,
        query: str,
        coding_output: dict[str, Any],
        workspace_path: str,
        test_command: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run pre-existing repository tests without creating or modifying test files."""
        command = self._resolve_command(test_command)
        coding_summary = coding_output.get("response", "")
        prompt = (
            f"Run the EXISTING test suite in workspace {workspace_path}.\n\n"
            f"Command: {command}\n\n"
            f"User task / issue:\n{query}\n\n"
            f"Coding stage summary:\n{coding_summary}\n\n"
            "Requirements:\n"
            "- DO NOT create, modify, or delete any test files.\n"
            "- DO NOT modify production or source code.\n"
            "- Only execute the test command from the repository root.\n"
            "- Report pass/fail and include key stdout/stderr details."
        )
        run = self.sdk_runner.run(
            prompt=prompt,
            workspace_path=workspace_path,
            timeout_seconds=settings.tester_timeout_seconds,
        )
        passed = run["ok"] and run["status"] == "finished"
        result = {
            "response": run["message"] if passed else f"Tests failed.\n\n{run['message']}",
            "phase": "run_tests",
            "mode": TesterMode.BENCHMARKING.value,
            "passed": passed,
            "return_code": 0 if passed else 1,
            "command": command,
            "workspace_path": workspace_path,
            "coding_context": coding_summary,
            "test_authoring_context": "",
            "agent": "tester",
            "sdk_status": run["status"],
            "sdk_run_id": run["run_id"],
            "openhands_result": run["raw_result"],
        }
        if not run["ok"]:
            result["error"] = run.get("error", "openhands_sdk_error")
        return result

    def run_generated_tests(
        self,
        query: str,
        coding_output: dict[str, Any],
        test_authoring_output: dict[str, Any],
        workspace_path: str,
        test_command: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute test command after tests were authored in test_generation mode."""
        command = self._resolve_command(test_command)
        coding_summary = coding_output.get("response", "")
        tests_summary = test_authoring_output.get("response", "")
        prompt = (
            f"Run tests in workspace {workspace_path}.\n\n"
            f"Command: {command}\n\n"
            f"Tests authored:\n{tests_summary}\n\n"
            f"Coding stage summary:\n{coding_summary}\n\n"
            "Execute the command, report pass/fail and include key stdout/stderr details."
        )
        run = self.sdk_runner.run(
            prompt=prompt,
            workspace_path=workspace_path,
            timeout_seconds=settings.tester_timeout_seconds,
        )
        passed = run["ok"] and run["status"] == "finished"
        result = {
            "response": run["message"] if passed else f"Tests failed.\n\n{run['message']}",
            "phase": "run_tests",
            "mode": TesterMode.TEST_GENERATION.value,
            "passed": passed,
            "return_code": 0 if passed else 1,
            "command": command,
            "workspace_path": workspace_path,
            "coding_context": coding_summary,
            "test_authoring_context": tests_summary,
            "agent": "tester",
            "sdk_status": run["status"],
            "sdk_run_id": run["run_id"],
            "openhands_result": run["raw_result"],
        }
        if not run["ok"]:
            result["error"] = run.get("error", "openhands_sdk_error")
        return result

    def run_tests(
        self,
        query: str,
        coding_output: dict[str, Any],
        test_authoring_output: dict[str, Any],
        workspace_path: str,
        *,
        tester_mode: Optional[str | TesterMode] = None,
        test_command: Optional[str] = None,
    ) -> dict[str, Any]:
        """Dispatch to benchmarking or test_generation test execution."""
        mode = self._resolve_mode(tester_mode)
        if mode == TesterMode.BENCHMARKING:
            return self.run_benchmarking_tests(
                query=query,
                coding_output=coding_output,
                workspace_path=workspace_path,
                test_command=test_command,
            )
        return self.run_generated_tests(
            query=query,
            coding_output=coding_output,
            test_authoring_output=test_authoring_output,
            workspace_path=workspace_path,
            test_command=test_command,
        )
