"""RAG-style workspace context: structure overview + query-relevant file selection."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".md", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"}

_SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "site-packages",
    ".eggs",
    "htmlcov",
    ".coverage",
}

_ANCHOR_FILENAMES = {
    "readme.md",
    "readme.rst",
    "readme.txt",
    "readme",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "package-lock.json",
    "cargo.toml",
    "go.mod",
    "makefile",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "main.py",
    "app.py",
    "__main__.py",
    "index.js",
    "index.ts",
    "conftest.py",
}

_ENTRYPOINT_BASENAMES = {"main.py", "app.py", "__main__.py", "index.js", "index.ts", "server.py", "wsgi.py", "asgi.py"}

_STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "are", "was", "were",
    "will", "can", "should", "would", "could", "into", "about", "when", "what", "which",
    "who", "how", "all", "any", "not", "but", "you", "your", "our", "their", "its",
    "add", "fix", "update", "change", "make", "use", "using", "need", "please", "file",
    "files", "code", "function", "class", "method", "implement", "create", "delete",
    "remove", "modify", "refactor", "test", "tests", "bug", "issue", "error",
}


@dataclass(frozen=True)
class WorkspaceFile:
    path: Path
    relative: str
    suffix: str
    content: str
    size: int


def _tokenize(text: str) -> set[str]:
    tokens = {word.lower() for word in re.findall(r"[\w]+", text) if len(word) > 2}
    return tokens - _STOP_WORDS


def _should_skip_path(path: Path) -> bool:
    return any(part in _SKIP_DIR_NAMES or part.startswith(".") for part in path.parts)


def _is_code_file(path: Path) -> bool:
    if path.suffix.lower() in _CODE_EXTENSIONS:
        return True
    return path.name.lower() in _ANCHOR_FILENAMES


def _scan_workspace(root: Path) -> list[WorkspaceFile]:
    files: list[WorkspaceFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _should_skip_path(path.relative_to(root)):
            continue
        if not _is_code_file(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            WorkspaceFile(
                path=path,
                relative=relative,
                suffix=path.suffix.lower(),
                content=content,
                size=len(content),
            )
        )
    return files


def _build_tree_summary(root: Path, files: list[WorkspaceFile], *, max_depth: int = 4) -> str:
    """Compact directory tree with per-folder file counts."""
    dir_counts: dict[str, dict[str, int]] = {}
    top_level: list[str] = []

    for item in files:
        parts = Path(item.relative).parts
        if len(parts) == 1:
            top_level.append(item.relative)
            continue
        for depth in range(1, min(len(parts), max_depth + 1)):
            dir_path = "/".join(parts[:depth])
            bucket = dir_counts.setdefault(dir_path, {"files": 0, "exts": {}})
            bucket["files"] += 1
            ext = item.suffix or "(no ext)"
            bucket["exts"][ext] = bucket["exts"].get(ext, 0) + 1

    ext_totals: dict[str, int] = {}
    for item in files:
        ext = item.suffix or "(no ext)"
        ext_totals[ext] = ext_totals.get(ext, 0) + 1

    lines = [
        f"Root: {root.name or root.as_posix()}",
        f"Total source files: {len(files)}",
        "File types: " + ", ".join(f"{ext}={count}" for ext, count in sorted(ext_totals.items(), key=lambda x: -x[1])[:8]),
        "",
        "Directory layout:",
    ]

    shown_dirs = sorted(dir_counts.items(), key=lambda x: x[0])[:40]
    for dir_path, stats in shown_dirs:
        depth = dir_path.count("/")
        indent = "  " * depth
        ext_summary = ", ".join(
            f"{count}{ext}" for ext, count in sorted(stats["exts"].items(), key=lambda x: -x[1])[:4]
        )
        lines.append(f"{indent}{dir_path}/ ({stats['files']} files: {ext_summary})")

    if top_level:
        lines.append("")
        lines.append("Root-level files:")
        for name in sorted(top_level)[:20]:
            lines.append(f"  - {name}")
        if len(top_level) > 20:
            lines.append(f"  ... and {len(top_level) - 20} more")

    return "\n".join(lines)


def _detect_entry_points(files: list[WorkspaceFile]) -> list[str]:
    found: list[str] = []
    for item in files:
        basename = item.path.name.lower()
        if basename in _ENTRYPOINT_BASENAMES:
            found.append(item.relative)
            continue
        if basename == "__init__.py" and item.size < 500:
            found.append(item.relative)
    return found[:10]


def _score_file(
    item: WorkspaceFile,
    query_terms: set[str],
    *,
    priority_paths: set[str] | None = None,
) -> float:
    score = 0.0
    relative_lower = item.relative.lower()
    basename_lower = item.path.name.lower()
    content_lower = item.content.lower()

    if basename_lower in _ANCHOR_FILENAMES:
        score += 100.0

    if priority_paths and item.relative in priority_paths:
        score += 200.0

    for term in query_terms:
        if term in relative_lower:
            score += 8.0
        if term in basename_lower:
            score += 5.0
        occurrences = content_lower.count(term)
        if occurrences:
            score += min(occurrences, 15) * 1.5

    if "/test" in relative_lower or relative_lower.startswith("test") or "_test." in basename_lower:
        if any(t in query_terms for t in {"test", "pytest", "unittest", "spec"}):
            score += 12.0

    if item.size < 4000:
        score += 2.0
    elif item.size > 20_000:
        score -= 5.0

    return score


def _extract_relevant_snippets(content: str, query_terms: set[str], max_chars: int) -> str:
    if len(content) <= max_chars:
        return content

    lines = content.splitlines()
    selected: set[int] = set(range(min(35, len(lines))))

    for index, line in enumerate(lines):
        lower = line.lower()
        if any(term in lower for term in query_terms):
            for ctx in range(max(0, index - 2), min(len(lines), index + 3)):
                selected.add(ctx)

    if len(selected) >= len(lines) * 0.85:
        return content[:max_chars] + "\n...[truncated]...\n"

    ordered = [lines[i] for i in sorted(selected)]
    body = "\n".join(ordered)
    omitted = len(lines) - len(selected)
    suffix = f"\n...[omitted {omitted} non-matching lines]...\n" if omitted else "\n"
    result = body + suffix
    if len(result) > max_chars:
        return result[:max_chars] + "\n...[truncated]...\n"
    return result


def _extract_paths_from_text(text: str) -> set[str]:
    patterns = [
        r"[`'\"]?([a-zA-Z0-9_./-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|md|yaml|yml|toml|json))[`'\"]?",
        r"(?:file|path|module)[:\s]+[`'\"]?([a-zA-Z0-9_./-]+)[`'\"]?",
    ]
    found: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            normalized = match.replace("\\", "/").lstrip("./")
            if normalized and not normalized.startswith("http"):
                found.add(normalized)
    return found


def collect_workspace_context(
    workspace_path: str,
    *,
    query: str | None = None,
    extra_hints: str | None = None,
    priority_paths: set[str] | None = None,
    max_files: int = 25,
    max_chars: int = 60_000,
) -> str:
    """Build RAG-style workspace context: structure overview + ranked file snippets."""
    root = Path(workspace_path)
    if not root.exists() or not root.is_dir():
        return f"Workspace not available at {workspace_path}."

    files = _scan_workspace(root)
    if not files:
        return f"No source files found under {workspace_path}."

    hint_text = " ".join(part for part in (query, extra_hints) if part)
    query_terms = _tokenize(hint_text)
    inferred_paths = _extract_paths_from_text(hint_text) if hint_text else set()
    merged_priority = (priority_paths or set()) | inferred_paths

    tree_summary = _build_tree_summary(root, files)
    entry_points = _detect_entry_points(files)
    entry_section = ""
    if entry_points:
        entry_section = "Likely entry points:\n" + "\n".join(f"  - {path}" for path in entry_points) + "\n\n"

    ranked = sorted(
        files,
        key=lambda item: _score_file(item, query_terms, priority_paths=merged_priority),
        reverse=True,
    )

    anchors = [item for item in ranked if item.path.name.lower() in _ANCHOR_FILENAMES]
    anchor_relatives = {item.relative for item in anchors}
    remaining_budget_files = max_files - len(anchors)
    top_ranked = [item for item in ranked if item.relative not in anchor_relatives][: max(0, remaining_budget_files)]
    selected = anchors + top_ranked

    seen: set[str] = set()
    ordered_selection: list[WorkspaceFile] = []
    for item in selected:
        if item.relative in seen:
            continue
        seen.add(item.relative)
        ordered_selection.append(item)

    sections = [
        "=== Project structure overview ===",
        tree_summary,
        "",
        entry_section.rstrip(),
        "=== Relevant source files (ranked by task relevance) ===",
    ]
    sections = [part for part in sections if part]

    total_chars = sum(len(part) for part in sections)
    if ordered_selection:
        per_file_budget = max(800, (max_chars - total_chars) // len(ordered_selection))
    else:
        per_file_budget = max_chars

    file_chunks: list[str] = []
    for item in ordered_selection:
        snippet_body = _extract_relevant_snippets(item.content, query_terms, per_file_budget)
        chunk = f"--- {item.relative} ---\n{snippet_body}\n"
        if total_chars + len(chunk) > max_chars:
            remaining = max_chars - total_chars
            if remaining <= 200:
                break
            chunk = chunk[:remaining] + "\n...[truncated]...\n"
        file_chunks.append(chunk)
        total_chars += len(chunk)

    if not file_chunks:
        return "\n".join(sections) + "\n(No file snippets fit within character budget.)"

    return "\n".join(sections) + "\n" + "\n".join(file_chunks)
