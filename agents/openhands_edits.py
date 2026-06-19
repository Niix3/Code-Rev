"""Extract file-edit metadata from OpenHands SDK conversations."""
from __future__ import annotations

from typing import Any

_FILE_EDITOR_TOOL_NAMES = frozenset({
    "file_editor",
    "FileEditorTool",
    "file_editor_tool",
    "str_replace_editor",
    "str_replace",
})

_MUTATING_COMMANDS = frozenset({"create", "str_replace", "insert", "undo_edit"})


def _action_field(action: Any, name: str) -> Any:
    if action is None:
        return None
    if isinstance(action, dict):
        return action.get(name)
    return getattr(action, name, None)


def extract_edited_files(conversation: Any) -> list[str]:
    """Return absolute/relative paths OpenHands file_editor touched."""
    try:
        from openhands.sdk.event import ActionEvent
    except Exception:
        return []

    events = getattr(conversation, "events", None)
    if events is None:
        state = getattr(conversation, "state", None)
        events = getattr(state, "events", None) if state is not None else None
    if events is None:
        return []

    edited: list[str] = []
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, ActionEvent):
            continue
        tool_name = getattr(event, "tool_name", "") or ""
        if tool_name not in _FILE_EDITOR_TOOL_NAMES:
            continue
        command = _action_field(event.action, "command")
        if command not in _MUTATING_COMMANDS:
            continue
        path = _action_field(event.action, "path")
        if not path or not isinstance(path, str):
            continue
        normalized = path.replace("\\", "/")
        if normalized not in seen:
            seen.add(normalized)
            edited.append(normalized)
    return edited
