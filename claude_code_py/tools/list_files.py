"""List files in a directory."""
from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolContext, ToolResult
from .read_file import ReadFileTool


class ListFilesTool(Tool):
    name = "list_files"
    description = (
        "List files and subdirectories in a directory (non-recursive by "
        "default). Returns one entry per line with a trailing / for "
        "directories. Use the `recursive` option for a tree view (capped at "
        "1000 entries)."
    )
    schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list (default: working dir).",
            },
            "recursive": {
                "type": "boolean",
                "description": "If true, walk recursively (max 1000 entries).",
            },
        },
        "required": [],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        reader = ReadFileTool()
        raw = args.get("path", "") or "."
        path = reader._resolve(raw, ctx)
        if path is None:
            return ToolResult.error("Path escapes the working directory.")
        if not path.exists():
            return ToolResult.error(f"Directory not found: {path}")
        if not path.is_dir():
            return ToolResult.error(f"Not a directory: {path}")

        recursive = bool(args.get("recursive", False))
        rel_root = reader._rel(path, ctx)

        if recursive:
            entries: list[str] = []
            for i, p in enumerate(sorted(path.rglob("*"))):
                if i >= 1000:
                    entries.append("... [truncated at 1000 entries]")
                    break
                rel = p.relative_to(ctx.config.working_dir)
                suffix = "/" if p.is_dir() else ""
                entries.append(f"{rel}{suffix}")
            return ToolResult.ok_(
                f"Recursive listing of {rel_root}:\n" + "\n".join(entries),
                structured={"entries": entries},
            )

        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as e:
            return ToolResult.error(f"Cannot list {path}: {e}")

        lines: list[str] = []
        for c in children:
            suffix = "/" if c.is_dir() else ""
            lines.append(f"{c.name}{suffix}")
        body = "\n".join(lines) if lines else "(empty directory)"
        return ToolResult.ok_(
            f"Listing of {rel_root}:\n{body}",
            structured={"entries": lines},
        )
