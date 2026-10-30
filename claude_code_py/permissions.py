"""Permission policy for tool execution.

Mirrors Claude Code's allow/ask/deny model: each tool call is classified and
either auto-approved, requires user approval, or is hard-denied. The Gradio
UI surfaces the "ask" decisions as buttons; the CLI uses a y/N prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .config import AgentConfig


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# Callback shape: (tool_name, args) -> PermissionDecision.
# The UI/CLI implements this to prompt the user.
PermissionCallback = Callable[[str, dict], "PermissionDecision"]


@dataclass
class PermissionPolicy:
    """Classify tool calls into allow / ask / deny."""

    config: "AgentConfig"

    def classify(self, tool_name: str, args: dict) -> PermissionDecision:
        """Return the policy decision for a tool call.

        Hard-deny rules always win. Then per-tool auto-approve flags apply.
        Anything else falls through to ASK.
        """
        # 1. Hard-deny dangerous bash patterns — always.
        if tool_name == "bash":
            cmd = str(args.get("command", ""))
            for pat in self.config.bash_deny_patterns:
                if re.search(pat, cmd):
                    return PermissionDecision.DENY
            if self.config.auto_approve_bash:
                return PermissionDecision.ALLOW
            return PermissionDecision.ASK

        # 2. Read-only file tools.
        if tool_name in ("read_file", "list_files", "grep", "glob", "todo_read"):
            if self.config.auto_approve_read_tools:
                return PermissionDecision.ALLOW
            return PermissionDecision.ASK

        # 3. Write tools.
        if tool_name in ("write_file", "edit_file", "bash_write"):
            if self.config.auto_approve_write_tools:
                return PermissionDecision.ALLOW
            return PermissionDecision.ASK

        # 4. Pure in-memory tools.
        if tool_name in ("todo_write",):
            return PermissionDecision.ALLOW

        # 5. Unknown tool — never auto-run.
        return PermissionDecision.ASK

    def reason(self, tool_name: str, args: dict, decision: PermissionDecision) -> str:
        """Human-readable explanation, used by the UI's approval dialog."""
        if decision is PermissionDecision.DENY:
            if tool_name == "bash":
                return "Command matches a hard-deny pattern (destructive or unsafe)."
            return "Tool blocked by policy."
        if decision is PermissionDecision.ALLOW:
            return "Auto-approved by current permission settings."
        return "This tool call needs your approval."
