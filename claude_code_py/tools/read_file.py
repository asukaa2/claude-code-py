"""Read a file's contents from the working directory tree."""
from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolContext, ToolResult


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a file. Supports an optional line range "
        "(1-indexed, inclusive). Use this before editing any file."
    )
    schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (relative to working dir, or absolute).",
            },
            "start_line": {
                "type": "integer",
                "description": "Optional 1-indexed start line (inclusive).",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional 1-indexed end line (inclusive).",
            },
        },
        "required": ["path"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = self._resolve(args.get("path", ""), ctx)
        if path is None:
            return ToolResult.error("Path escapes the working directory.")
        if not path.exists():
            return ToolResult.error(f"File not found: {path}")
        if path.is_dir():
            return ToolResult.error(f"Path is a directory, not a file: {path}")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult.error(f"Cannot read {path}: {e}")

        lines = text.splitlines()
        start = int(args.get("start_line") or 1)
        end = int(args.get("end_line") or len(lines))
        start = max(1, start)
        end = min(len(lines), end)
        if start > end:
            return ToolResult.error(f"Invalid line range: {start}..{end}")

        # Cap at 2000 lines to avoid blowing the context.
        max_lines = 2000
        truncated = False
        if end - start + 1 > max_lines:
            end = start + max_lines - 1
            truncated = True

        selected = lines[start - 1 : end]
        # cat -n style line numbers help the model use edit_file correctly.
        numbered = "\n".join(
            f"{i:6d}\t{line}" for i, line in enumerate(selected, start=start)
        )
        header = f"File: {self._rel(path, ctx)} ({len(lines)} lines total)"
        if truncated:
            header += f"  [showing lines {start}-{end} of {len(lines)}]"
        elif start != 1 or end != len(lines):
            header += f"  [showing lines {start}-{end}]"
        return ToolResult.ok_(
            f"{header}\n{numbered}",
            structured={"path": str(self._rel(path, ctx)), "lines": selected},
        )

    # ----- helpers -----

    def _resolve(self, raw: str, ctx: ToolContext) -> Path | None:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (ctx.config.working_dir / p)
        try:
            p = p.resolve(strict=False)
        except OSError:
            return None
        # Keep the agent inside the working directory tree.
        try:
            p.relative_to(ctx.config.working_dir)
        except ValueError:
            return None
        return p

    def _rel(self, path: Path, ctx: ToolContext) -> Path:
        try:
            return path.relative_to(ctx.config.working_dir)
        except ValueError:
            return path
