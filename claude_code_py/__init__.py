"""
claude_code_py — A Python reimplementation of Anthropic's Claude Code
agentic coding assistant, with a Gradio web UI for easier control than the CLI.

Supports multiple LLM providers (Anthropic is the default; DeepSeek, Gemini,
HuggingFace, and Custom OpenAI-compatible endpoints are also supported).

Public API:
    from claude_code_py import Agent, AgentConfig, ProviderConfig
    from claude_code_py.tools import ALL_TOOLS
"""

from .config import AgentConfig, get_default_config, config_for_provider
from .agent import Agent, AgentEvent, EventType
from .session import Session, SessionStore
from .permissions import PermissionPolicy, PermissionDecision
from .llm import LLMClient, LLMMessage, LLMResponse
from .prompts import build_system_prompt, MODE_PROMPTS
from .providers import (
    ProviderConfig,
    LLMProvider,
    KNOWN_PROVIDERS,
    PROVIDER_IDS,
    default_model_for,
    env_key_for,
)

__version__ = "0.2.0"
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
    "LLMMessage",
    "LLMResponse",
    "ProviderConfig",
    "LLMProvider",
    "KNOWN_PROVIDERS",
    "PROVIDER_IDS",
    "default_model_for",
    "env_key_for",
    "build_system_prompt",
    "MODE_PROMPTS",
    "get_default_config",
    "config_for_provider",
]
