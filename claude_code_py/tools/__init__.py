"""Tool registry for claude_code_py.

Importing this module gives you `ALL_TOOLS` (a list of Tool instances) and
`default_registry()` (a Registry pre-populated with every built-in tool).
"""
from __future__ import annotations

from .base import Registry, Tool, ToolContext, ToolResult
from .bash import BashTool
from .edit_file import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .list_files import ListFilesTool
from .read_file import ReadFileTool
from .todo import TodoReadTool, TodoWriteTool
from .write_file import WriteFileTool

#: Instantiated once, shared across registries.
ALL_TOOLS: list[Tool] = [
    BashTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    ListFilesTool(),
    GrepTool(),
    GlobTool(),
    TodoWriteTool(),
    TodoReadTool(),
]


def default_registry() -> Registry:
    """Return a fresh Registry with every built-in tool registered."""
    reg = Registry()
    for t in ALL_TOOLS:
        reg.register(t)
    return reg


__all__ = [
    "Registry",
    "Tool",
    "ToolContext",
    "ToolResult",
    "ALL_TOOLS",
    "default_registry",
    "BashTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "ListFilesTool",
    "GrepTool",
    "GlobTool",
    "TodoWriteTool",
    "TodoReadTool",
]
