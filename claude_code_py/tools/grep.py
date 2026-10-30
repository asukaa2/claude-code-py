"""Grep tool — content search via ripgrep (rg)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import Tool, ToolContext, ToolResult
from .read_file import ReadFileTool


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents using ripgrep (regex by default). Returns "
        "matching lines with file:line:prefix. Use `glob` to filter by "
        "filename pattern. Caps at 200 matches."
    )
    schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: working dir).",
            },
            "glob": {
                "type": "string",
                "description": "Optional glob filter, e.g. '*.py' or 'src/**/*.ts'.",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "If true, -i flag is added.",
            },
            "max_results": {
                "type": "integer",
                "description": "Optional cap (default 200).",
            },
        },
        "required": ["pattern"],
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        if not shutil.which("rg"):
            return ToolResult.error("ripgrep (rg) is not installed.")
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return ToolResult.error("Empty pattern.")
        reader = ReadFileTool()
        target = reader._resolve(args.get("path", "") or ".", ctx)
        if target is None:
            return ToolResult.error("Path escapes the working directory.")
        if not target.exists():
            return ToolResult.error(f"Path not found: {target}")

        cmd: list[str] = ["rg", "--line-number", "--no-heading", "--color=never"]
        if args.get("case_insensitive"):
            cmd.append("-i")
        if args.get("glob"):
            cmd += ["--glob", str(args["glob"])]
        max_results = int(args.get("max_results") or 200)
        cmd += ["--max-count", str(max_results), pattern, str(target)]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(ctx.config.working_dir),
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error("grep timed out after 30s.")

        # rg exit 0 = matches, 1 = no matches, 2 = error.
        if proc.returncode == 2:
            return ToolResult.error(f"rg error: {proc.stderr.strip()}")

        out = (proc.stdout or "").rstrip()
        if not out:
            return ToolResult.ok_("No matches found.", structured={"matches": []})

        # Make paths relative for readability.
        rel_lines: list[str] = []
        for line in out.splitlines():
            # Try to shorten absolute paths to working-dir-relative.
            try:
                abs_path, _, rest = line.split(":", 2)
                p = Path(abs_path)
                rel = p.relative_to(ctx.config.working_dir)
                rel_lines.append(f"{rel}:{rest}")
            except ValueError:
                rel_lines.append(line)

        truncated = False
        if len(rel_lines) > max_results:
            rel_lines = rel_lines[:max_results]
            truncated = True
        body = "\n".join(rel_lines)
        if truncated:
            body += f"\n... [truncated at {max_results} matches]"
        return ToolResult.ok_(
            f"{len(rel_lines)} match(es):\n{body}",
            structured={"matches": rel_lines, "count": len(rel_lines)},
        )
