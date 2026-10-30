"""Edit a file by replacing a unique old_string with new_string.

Mirrors Claude Code's Edit tool semantics: the old_string must appear exactly
once in the file, so edits are unambiguous.
"""
from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolContext, ToolResult
from .read_file import ReadFileTool


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Edit a file by replacing `old_string` with `new_string`. The "
        "`old_string` MUST appear exactly once in the file (use enough "
        "surrounding context to make it unique). Parent directories are not "
        "created. Returns a snippet of the edited region."
    )
    schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (relative to working dir).",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find (must be unique in the file).",
            },
            "new_string": {
                "type": "string",
                "description": "Text to replace it with.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "If true, replace all occurrences (default false).",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        reader = ReadFileTool()
        path = reader._resolve(args.get("path", ""), ctx)
        if path is None:
            return ToolResult.error("Path escapes the working directory.")
        if not path.exists():
            return ToolResult.error(f"File not found: {path}")

        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        replace_all = bool(args.get("replace_all", False))
        if not old:
            return ToolResult.error("old_string is empty.")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult.error(f"Cannot read {path}: {e}")

        count = text.count(old)
        if count == 0:
            return ToolResult.error(
                "old_string not found in file. Make sure to copy it verbatim "
                "(including whitespace)."
            )
        if count > 1 and not replace_all:
            return ToolResult.error(
                f"old_string appears {count} times in the file. Add more "
                "surrounding context to make it unique, or set replace_all=true."
            )

        new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        try:
            path.write_text(new_text, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Cannot write {path}: {e}")

        # Build a short preview around the first replacement.
        idx = new_text.find(new)
        preview_start = max(0, idx - 100)
        preview_end = min(len(new_text), idx + len(new) + 100)
        preview = new_text[preview_start:preview_end]
        rel = reader._rel(path, ctx)
        return ToolResult.ok_(
            f"Edited {rel} ({count} replacement{'s' if count != 1 else ''}). "
            f"Preview:\n```\n{preview}\n```",
            structured={"path": str(rel), "replacements": count},
        )
