"""Tool base class + registry.

Each tool is a small class with:
  - `name`, `description`, `schema` (JSON-Schema-ish dict for the model)
  - `run(args, ctx)` -> ToolResult

The `Registry` collects tools, renders documentation for the system prompt,
and dispatches parsed tool calls.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..session import Session


# ---------------------------------------------------------------------- #
# Execution context passed to every tool run
# ---------------------------------------------------------------------- #


@dataclass
class ToolContext:
    """Everything a tool needs to execute.

    Carried through the agent loop so tools don't need global state.
    """

    config: "AgentConfig"
    session: "Session"
    #: Scratch dict tools can use to share state within a single agent turn
    #: (e.g. the todo list lives here).
    turn_state: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Tool result
# ---------------------------------------------------------------------- #


@dataclass
class ToolResult:
    """Output of a tool execution.

    `ok=False` indicates the tool errored; `content` still carries the
    error message so the model can react to it.
    """

    ok: bool
    content: str
    #: Optional structured payload (e.g. file lists) for UI rendering.
    structured: Optional[dict] = None

    @classmethod
    def ok_(cls, content: str, structured: Optional[dict] = None) -> "ToolResult":
        return cls(ok=True, content=content, structured=structured)

    @classmethod
    def error(cls, content: str, structured: Optional[dict] = None) -> "ToolResult":
        return cls(ok=False, content=content, structured=structured)


# ---------------------------------------------------------------------- #
# Tool abstract base
# ---------------------------------------------------------------------- #


class Tool(abc.ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    #: JSON-schema-style description of arguments, shown to the model.
    schema: dict = {}

    @abc.abstractmethod
    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Execute the tool. Must be pure-ish: same args+ctx -> same result."""
        raise NotImplementedError

    # Convenience: render this tool's doc for the system prompt.
    def render_doc(self) -> str:
        params = self.schema.get("properties", {})
        required = self.schema.get("required", [])
        lines = [f"### {self.name}", "", self.description, ""]
        if params:
            lines.append("Parameters:")
            for pname, pschema in params.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req = " (required)" if pname in required else ""
                lines.append(f"  - `{pname}` ({ptype}{req}): {pdesc}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #


class Registry:
    """Holds all registered tools and dispatches calls."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"Tool {tool!r} has no name")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def render_doc(self) -> str:
        """Render the full tools section for the system prompt."""
        return "\n".join(t.render_doc() for t in self._tools.values())

    def dispatch(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult.error(
                f"Unknown tool: {name!r}. Available: {', '.join(self.names())}"
            )
        try:
            return tool.run(args, ctx)
        except Exception as e:  # noqa: BLE001 — surface every error to the model
            return ToolResult.error(f"Tool {name!r} crashed: {type(e).__name__}: {e}")
