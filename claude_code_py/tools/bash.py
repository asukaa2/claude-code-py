"""Run a bash command in the working directory."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import Tool, ToolContext, ToolResult


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a bash command in the working directory. Returns combined "
        "stdout+stderr (truncated to ~10KB) and the exit code. Use this for "
        "running tests, git, building, inspecting files, etc. Avoid "
        "long-running or interactive commands."
    )
    schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Optional timeout in seconds (default 120).",
            },
        },
        "required": ["command"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolResult.error("Empty command.")
        timeout = int(args.get("timeout") or ctx.config.bash_timeout)
        env = os.environ.copy()
        env.update(ctx.config.bash_env)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(ctx.config.working_dir),
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error(
                f"Command timed out after {timeout}s: {command}"
            )

        out = (proc.stdout or "") + (proc.stderr or "")
        truncated = False
        if len(out) > 10240:
            out = out[:10240] + f"\n... [truncated, {len(out)} bytes total]"
            truncated = True
        out = out.rstrip() or "(no output)"
        return ToolResult.ok_(
            f"$ {command}\n{out}\n[exit code: {proc.returncode}]",
            structured={
                "command": command,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
                "truncated": truncated,
            },
        )
