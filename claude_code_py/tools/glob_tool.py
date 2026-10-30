"""Glob tool — find files by name pattern."""
from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolContext, ToolResult
from .read_file import ReadFileTool


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/test_*'). "
        "Returns paths relative to the working directory, capped at 500."
    )
    schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (supports ** for recursive).",
            },
            "path": {
                "type": "string",
                "description": "Root directory for the search (default: working dir).",
            },
        },
        "required": ["pattern"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return ToolResult.error("Empty pattern.")
        reader = ReadFileTool()
        root = reader._resolve(args.get("path", "") or ".", ctx)
        if root is None:
            return ToolResult.error("Path escapes the working directory.")
        if not root.exists():
            return ToolResult.error(f"Path not found: {root}")

        matches: list[str] = []
        try:
            for p in sorted(root.glob(pattern)):
                if p.is_dir():
                    continue
                try:
                    rel = p.relative_to(ctx.config.working_dir)
                except ValueError:
                    rel = p
                matches.append(str(rel))
                if len(matches) >= 500:
                    matches.append("... [truncated at 500]")
                    break
        except (OSError, ValueError) as e:
            return ToolResult.error(f"glob failed: {e}")

        if not matches:
            return ToolResult.ok_("No files matched.", structured={"matches": []})
        return ToolResult.ok_(
            f"{len(matches)} file(s) matched '{pattern}':\n" + "\n".join(matches),
            structured={"matches": matches, "count": len(matches)},
        )
