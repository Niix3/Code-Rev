"""Collect git diffs and changed-file lists from agent workspaces."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path, PurePosixPath

# Paths reviewers and SWE-bench should ignore (noise / out of scope).
_REVIEW_EXCLUDE_DIR_PARTS = frozenset({
    "locale",
    "locales",
    "static",
    "docs",
    "node_modules",
    "__pycache__",
    ".git",
    ".github",
    ".tx",
})

_REVIEW_EXCLUDE_BASENAMES = frozenset({
    ".editorconfig",
    ".gitignore",
    ".gitattributes",
    ".eslintignore",
    ".pre-commit-config.yaml",
    "package-lock.json",
    "yarn.lock",
})

_REVIEW_EXCLUDE_SUFFIXES = frozenset({
    ".po",
    ".mo",
    ".pot",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".map",
    ".min.js",
    ".min.css",
})

_REVIEW_INCLUDE_SUFFIXES = frozenset({
    ".py",
    ".pyi",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rb",
    ".php",
})


def _git_available() -> bool:
    return shutil.which("git") is not None


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    if not _git_available():
        return None
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[diff truncated]...\n"


def is_review_relevant_path(relative: str, *, benchmarking: bool = False) -> bool:
    """Return True if a changed path should appear in review / SWE-bench patch."""
    normalized = relative.replace("\\", "/").lstrip("./")
    if not normalized:
        return False

    basename = PurePosixPath(normalized).name
    if basename in _REVIEW_EXCLUDE_BASENAMES:
        return False

    suffix = PurePosixPath(normalized).suffix.lower()
    if suffix in _REVIEW_EXCLUDE_SUFFIXES:
        return False

    parts = PurePosixPath(normalized).parts
    for part in parts:
        if part in _REVIEW_EXCLUDE_DIR_PARTS:
            return False
        if benchmarking and part == "tests":
            return False

    if suffix in _REVIEW_INCLUDE_SUFFIXES:
        return True

    # Allow extensionless source files in known source trees.
    if not suffix and parts and parts[0] in {"django", "astropy", "src", "lib"}:
        return True

    return False


def _order_paths_for_review(paths: list[str], priority_paths: set[str] | None) -> list[str]:
    priority = priority_paths or set()
    head = [path for path in paths if path in priority]
    tail = [path for path in paths if path not in priority]
    return head + tail


def _collect_all_changed_paths(root: Path) -> list[str]:
    changed: list[str] = []
    for args in (["diff", "--name-only"], ["diff", "--cached", "--name-only"]):
        result = _run_git(args, root)
        if result is not None and result.returncode == 0 and result.stdout.strip():
            changed.extend(line.strip() for line in result.stdout.splitlines() if line.strip())

    untracked = _run_git(["ls-files", "--others", "--exclude-standard"], root)
    if untracked is not None and untracked.returncode == 0 and untracked.stdout.strip():
        changed.extend(line.strip() for line in untracked.stdout.splitlines() if line.strip())

    seen: set[str] = set()
    ordered: list[str] = []
    for path in changed:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def list_changed_files(
    workspace_path: str,
    *,
    review_only: bool = False,
    benchmarking: bool = False,
) -> list[str]:
    """Return repo-relative paths changed by the coding agent."""
    root = Path(workspace_path)
    if not _git_available() or not (root / ".git").exists():
        return []

    paths = _collect_all_changed_paths(root)
    if not review_only:
        return paths

    return [path for path in paths if is_review_relevant_path(path, benchmarking=benchmarking)]


def _diff_for_paths(root: Path, paths: list[str], *, cached: bool) -> str:
    if not paths:
        return ""
    args = ["diff", "--cached" if cached else "diff", "--no-color", "--", *paths]
    result = _run_git(args, root)
    if result is None or result.returncode != 0:
        return ""
    return result.stdout.rstrip()


def _new_file_previews(
    root: Path,
    paths: list[str],
    *,
    per_file_max_chars: int = 4000,
) -> list[str]:
    sections: list[str] = []
    for relative in paths:
        file_path = root / relative
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        preview = content if len(content) <= per_file_max_chars else content[:per_file_max_chars] + "\n...[truncated]..."
        sections.append(f"=== New file: {relative} ===\n{preview}")
    return sections


def _untracked_paths(root: Path) -> set[str]:
    result = _run_git(["ls-files", "--others", "--exclude-standard"], root)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _split_tracked_untracked(root: Path, paths: list[str]) -> tuple[list[str], list[str]]:
    untracked_set = _untracked_paths(root)
    tracked = [path for path in paths if path not in untracked_set]
    untracked = [path for path in paths if path in untracked_set]
    return tracked, untracked


def _normalize_to_repo_relative(path: str, root: Path) -> str | None:
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return None
    root_posix = root.as_posix().rstrip("/")
    if normalized.startswith(root_posix + "/"):
        return normalized[len(root_posix) + 1 :]
    root_name = root.name
    marker = f"{root_name}/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return normalized.lstrip("./")


def stage_coder_changes(
    workspace_path: str,
    edited_paths: list[str] | None = None,
) -> None:
    """Stage coder edits so git diff reliably reflects OpenHands file changes."""
    root = Path(workspace_path)
    if not _git_available() or not (root / ".git").exists():
        return

    _run_git(["config", "core.autocrlf", "false"], root)
    _run_git(["config", "core.filemode", "false"], root)

    if edited_paths:
        relative_paths: list[str] = []
        for path in edited_paths:
            relative = _normalize_to_repo_relative(path, root)
            if relative:
                relative_paths.append(relative)
        for relative in relative_paths:
            _run_git(["add", "-N", "--", relative], root)
            _run_git(["add", "--", relative], root)
    else:
        _run_git(["add", "-u"], root)


def capture_coder_patch(
    workspace_path: str,
    *,
    edited_paths: list[str] | None = None,
    priority_paths: set[str] | None = None,
    benchmarking: bool = False,
) -> str:
    """Stage edits and return a unified diff snapshot for reviewers."""
    stage_coder_changes(workspace_path, edited_paths)
    return collect_unified_patch(
        workspace_path,
        benchmarking=benchmarking,
        priority_paths=priority_paths,
    )


def diff_is_empty(summary: str) -> bool:
    lowered = summary.lower()
    return (
        "no git changes detected" in lowered
        or "no production-source diff" in lowered
        or "no unified diff output" in lowered
    )


def collect_unified_patch(
    workspace_path: str,
    *,
    benchmarking: bool = True,
    priority_paths: set[str] | None = None,
) -> str:
    """Return a filtered unified diff suitable for SWE-bench submission."""
    root = Path(workspace_path)
    if not root.exists() or not (root / ".git").exists() or not _git_available():
        return ""

    all_changed = _collect_all_changed_paths(root)
    relevant = [path for path in all_changed if is_review_relevant_path(path, benchmarking=benchmarking)]
    ordered = _order_paths_for_review(relevant, priority_paths)
    tracked, untracked = _split_tracked_untracked(root, ordered)

    parts: list[str] = []
    for cached in (False, True):
        chunk = _diff_for_paths(root, tracked, cached=cached)
        if chunk:
            parts.append(chunk)

    parts.extend(_new_file_previews(root, untracked))
    return "\n\n".join(part for part in parts if part.strip())


def collect_workspace_diff(
    workspace_path: str,
    *,
    priority_paths: set[str] | None = None,
    benchmarking: bool = False,
    max_chars: int = 40_000,
) -> str:
    """Build a review-friendly summary of coder changes via git."""
    root = Path(workspace_path)
    if not root.exists():
        return f"Workspace not available at {workspace_path}."
    if not _git_available():
        return (
            "Git diff unavailable: git binary is not installed in the runtime environment. "
            "Reviewers must rely on workspace file snippets only."
        )
    if not (root / ".git").exists():
        return (
            "Git diff unavailable: workspace is not a git repository. "
            "Reviewers must rely on workspace file snippets only."
        )

    all_changed = _collect_all_changed_paths(root)
    relevant = [path for path in all_changed if is_review_relevant_path(path, benchmarking=benchmarking)]
    excluded = [path for path in all_changed if path not in relevant]
    ordered = _order_paths_for_review(relevant, priority_paths)

    sections: list[str] = []
    if ordered:
        sections.append(
            "Review-relevant changed files:\n" + "\n".join(f"  - {path}" for path in ordered)
        )
    if excluded:
        preview = ", ".join(excluded[:12])
        suffix = f" (and {len(excluded) - 12} more)" if len(excluded) > 12 else ""
        sections.append(
            f"Excluded from review diff ({len(excluded)} unrelated paths): {preview}{suffix}"
        )

    if priority_paths:
        missing = sorted(path for path in priority_paths if path not in all_changed)
        if missing:
            sections.append(
                "Architect-planned files with no detected changes:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )

    tracked, untracked = _split_tracked_untracked(root, ordered)

    diff_parts: list[str] = []
    header_len = sum(len(part) for part in sections) + 2 * len(sections)
    budget = max(8000, max_chars - header_len)

    priority = priority_paths or set()
    priority_tracked = [path for path in tracked if path in priority]
    other_tracked = [path for path in tracked if path not in priority]

    for label, cached in (("Unstaged diff", False), ("Staged diff", True)):
        for group_label, paths in (
            ("priority planned files", priority_tracked),
            ("other relevant files", other_tracked),
        ):
            if not paths or budget <= 500:
                continue
            chunk = _diff_for_paths(root, paths, cached=cached)
            if not chunk:
                continue
            piece = f"=== {label} ({group_label}) ===\n{_truncate(chunk, budget)}"
            diff_parts.append(piece)
            budget -= len(piece)

    if untracked and budget > 500:
        per_file = max(800, budget // max(len(untracked), 1))
        previews = _new_file_previews(root, untracked, per_file_max_chars=min(per_file, 4000))
        for preview in previews:
            if budget <= 500:
                break
            clipped = preview if len(preview) <= budget else preview[:budget] + "\n...[truncated]...\n"
            diff_parts.append(clipped)
            budget -= len(clipped)

    if diff_parts:
        sections.append("\n\n".join(diff_parts))
    elif ordered:
        sections.append("No unified diff output for review-relevant files.")
    elif all_changed:
        sections.append(
            "Changes detected only in excluded/unrelated paths; no production-source diff to review."
        )
    else:
        sections.append("No git changes detected (empty diff).")

    return _truncate("\n\n".join(sections), max_chars)
