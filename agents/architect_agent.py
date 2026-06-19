"""Architect agent that produces structured implementation plan."""
import json
import re
from typing import Any

from openai import OpenAI

from config import settings

from .workspace_context import collect_workspace_context

ARCHITECT_SYSTEM_PROMPT = (
    "You are a senior software architect. "
    "Given a task, produce a JSON object with two keys:\n"
    "  'explanation': a one-sentence description of the approach\n"
    "  'steps': an array of objects, each with:\n"
    "    'agent': one of coder | tester | debugger | critic | tool\n"
    "    'description': what this step does\n"
    "    'file': (optional) path of the file to edit\n"
    "    'parallel_group': (optional) integer for parallel execution (steps with same number run in parallel)\n"
    "    'tool_name': (required if agent=tool) the tool to execute (e.g., git_clone, pip_install)\n"
    "    'tool_params': (required if agent=tool) parameters dict for the tool\n"
    "Return ONLY valid JSON — no markdown fences, no extra text."
)


class ArchitectAgent:
    """Creates architecture/implementation guidance for downstream agents."""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    @staticmethod
    def _parse_plan(raw_text: str) -> dict[str, Any] | None:
        text = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    def design(self, query: str, workspace_path: str) -> dict[str, Any]:
        """Build architecture plan as structured JSON for coding and testing stages."""
        project_context = collect_workspace_context(
            workspace_path,
            query=query,
            max_files=30,
            max_chars=80_000,
        )
        user_prompt = (
            f"User request: {query}\n"
            f"Workspace path in shared volume: {workspace_path}\n\n"
            f"Current project context (structure + relevant files):\n{project_context}\n\n"
            "Use the project structure and relevant files above to choose concrete paths "
            "for each plan step. Produce the architecture plan JSON."
        )
        completion = self.client.chat.completions.create(
            model=settings.default_llm_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        response_text = completion.choices[0].message.content or ""
        plan = self._parse_plan(response_text)
        return {
            "response": response_text,
            "plan": plan,
            "explanation": plan.get("explanation") if plan else None,
            "steps": plan.get("steps", []) if plan else [],
            "workspace_path": workspace_path,
            "agent": "architect",
        }
