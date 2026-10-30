"""Todo list tool — keeps the agent organized on multi-step tasks.

State is stored in the turn_state dict on the ToolContext, so it survives
across tool calls within an agent turn but is reset on the next user turn.
"""
from __future__ import annotations

import json
from typing import Any

from .base import Tool, ToolContext, ToolResult


class TodoWriteTool(Tool):
    name = "todo_write"
    description = (
        "Create or update the agent's todo list for the current task. Pass "
        "the FULL list each time (this is a replace, not a patch). Each item "
        "has `content` and `status` (one of 'pending', 'in_progress', "
        "'completed'). Use this to plan multi-step work and show progress."
    )
    schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Full todo list (replaces existing).",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["todos"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        todos = args.get("todos") or []
        if not isinstance(todos, list):
            return ToolResult.error("`todos` must be an array.")
        cleaned: list[dict[str, Any]] = []
        for i, t in enumerate(todos):
            if not isinstance(t, dict):
                return ToolResult.error(f"todo[{i}] is not an object")
            content = str(t.get("content", "")).strip()
            status = str(t.get("status", "pending"))
            if status not in ("pending", "in_progress", "completed"):
                return ToolResult.error(f"todo[{i}] invalid status: {status}")
            cleaned.append({"content": content, "status": status})
        ctx.turn_state["todos"] = cleaned
        return ToolResult.ok_(
            f"Updated todo list ({len(cleaned)} items):\n"
            + json.dumps(cleaned, ensure_ascii=False, indent=2),
            structured={"todos": cleaned},
        )


class TodoReadTool(Tool):
    name = "todo_read"
    description = "Read the current todo list (empty if none set)."
    schema = {"type": "object", "properties": {}}

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        todos = ctx.turn_state.get("todos") or []
        return ToolResult.ok_(
            json.dumps(todos, ensure_ascii=False, indent=2),
            structured={"todos": todos},
        )
