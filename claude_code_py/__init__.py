"""
claude_code_py — A Python reimplementation of Anthropic's Claude Code
agentic coding assistant, with a Gradio web UI for easier control than the CLI.

Public API:
    from claude_code_py import Agent, AgentConfig, Session, PermissionPolicy
    from claude_code_py.tools import ALL_TOOLS
"""

from .config import AgentConfig, get_default_config
from .agent import Agent, AgentEvent, EventType
from .session import Session, SessionStore
from .permissions import PermissionPolicy, PermissionDecision
from .llm import LLMClient, LLMResponse
from .prompts import build_system_prompt, MODE_PROMPTS

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "AgentConfig",
    "AgentEvent",
    "EventType",
    "Session",
    "SessionStore",
    "PermissionPolicy",
    "PermissionDecision",
    "LLMClient",
    "LLMResponse",
    "build_system_prompt",
    "MODE_PROMPTS",
    "get_default_config",
]
