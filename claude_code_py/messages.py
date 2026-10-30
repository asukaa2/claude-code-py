"""Shared LLM message + response types.

Lives in its own module so both `llm.py` (the client facade) and
`providers/` (the implementations) can import it without a circular dep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMMessage:
    """A single chat message.

    Roles: "system" | "user" | "assistant" | "tool".
    For "tool" messages, `content` is the tool result and `tool_call_id` /
    `tool_name` carry metadata. For "assistant" tool-call turns, `tool_args`
    holds the call arguments.
    """

    role: str
    content: str
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Parsed response from the LLM."""

    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict = field(default_factory=dict)


__all__ = ["LLMMessage", "LLMResponse"]
