"""Write/create a file (overwrites)."""
from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolContext, ToolResult
from .read_file import ReadFileTool


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write `content` to `path`, creating or overwriting the file. "
        "Creates parent directories if needed. Use `edit_file` for targeted "
        "changes to an existing file; use this only for new files or full "
        "rewrites."
    )
    schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (relative to working dir).",
            },
            "content": {
                "type": "string",
                "description": "Full file contents to write.",
            },
        },
        "required": ["path", "content"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        reader = ReadFileTool()
        path = reader._resolve(args.get("path", ""), ctx)
        if path is None:
            return ToolResult.error("Path escapes the working directory.")
        content = str(args.get("content", ""))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Cannot write {path}: {e}")
        n_lines = content.count("\n") + (1 if content else 0)
        return ToolResult.ok_(
            f"Wrote {len(content)} bytes ({n_lines} lines) to {reader._rel(path, ctx)}",
            structured={"path": str(reader._rel(path, ctx)), "bytes": len(content)},
        )
