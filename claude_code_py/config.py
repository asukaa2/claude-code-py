"""Configuration for the claude_code_py agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Where session transcripts and runtime artifacts live.
DEFAULT_HOME = Path(os.environ.get("CLAUDE_CODE_PY_HOME", Path.home() / ".claude_code_py"))
DEFAULT_SESSIONS_DIR = DEFAULT_HOME / "sessions"
DEFAULT_LOG_DIR = DEFAULT_HOME / "logs"


@dataclass
class AgentConfig:
    """Tunable knobs for an Agent instance."""

    # --- LLM ---
    model: str = "glm-4-plus"
    #: CLI binary used to reach the LLM. Must accept -p / -s / -o and emit
    #: OpenAI-compatible JSON. Defaults to the bundled `z-ai` CLI.
    llm_cli: str = "z-ai"
    max_tokens: int = 8192
    #: Hard cap on agentic loop iterations per user turn — prevents runaway
    #: tool-call loops from burning through tokens.
    max_iterations: int = 25
    temperature: float = 0.7
    enable_thinking: bool = False

    # --- Agent behavior ---
    #: One of "default", "plan", "code", "explain". See prompts.MODE_PROMPTS.
    mode: str = "default"
    #: Working directory the agent operates in. File tools are rooted here.
    working_dir: Path = field(default_factory=lambda: Path.cwd())
    #: Optional override for the system prompt. When set, `mode` is ignored.
    system_prompt_override: Optional[str] = None

    # --- Permissions ---
    #: Policy used to decide whether a tool call needs user approval.
    auto_approve_read_tools: bool = True
    auto_approve_write_tools: bool = False
    auto_approve_bash: bool = False
    #: When True, dangerous bash patterns (rm -rf /, sudo, etc.) are always
    #: denied regardless of auto_approve_bash.
    bash_deny_patterns: tuple = (
        r"\brm\s+-rf\s+/(?:\s|$)",
        r"\bsudo\b",
        r"\bmkfs\b",
        r"\bdd\b.*\bof=/dev/",
        r":\(\)\s*\{",
        r"\bshutdown\b",
        r"\breboot\b",
    )

    # --- Session ---
    sessions_dir: Path = field(default_factory=lambda: DEFAULT_SESSIONS_DIR)
    autosave_sessions: bool = True

    # --- Misc ---
    #: Extra environment variables injected into bash tool subprocesses.
    bash_env: dict = field(default_factory=dict)
    bash_timeout: int = 120  # seconds

    def __post_init__(self) -> None:
        if isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir).expanduser().resolve()
        if isinstance(self.sessions_dir, str):
            self.sessions_dir = Path(self.sessions_dir).expanduser().resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)


def get_default_config(working_dir: Optional[Path | str] = None) -> AgentConfig:
    """Convenience factory for a default config rooted at `working_dir`."""
    cwd = Path(working_dir).expanduser().resolve() if working_dir else Path.cwd()
    return AgentConfig(working_dir=cwd)
