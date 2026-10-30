"""Anthropic provider — uses the native `anthropic` SDK with tool-calling.

This is the canonical provider. Messages are translated to Anthropic's
format (system prompt is a separate top-level field, not a message), and
tools are passed via the SDK's `tools=` parameter. The agent loop receives
structured tool calls back instead of having to parse them from prose.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from ..messages import LLMMessage, LLMResponse
from . import ProviderConfig, env_key_for


class AnthropicProvider:
    """Wraps `anthropic.Anthropic` for the agent loop."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        # Resolve API key: explicit > env var.
        key = config.api_key or os.environ.get(env_key_for("anthropic"), "")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Pass api_key= in ProviderConfig "
                "or export ANTHROPIC_API_KEY."
            )
        # Import lazily so the package imports cleanly when the SDK isn't
        # installed (the user might be using a different provider).
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "The `anthropic` package is required for the Anthropic "
                "provider. Install it with: pip install anthropic"
            ) from e
        kwargs: dict[str, Any] = {"api_key": key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = anthropic.Anthropic(**kwargs)

    def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        tools_schema: Optional[list[dict]] = None,
    ) -> LLMResponse:
        # Translate messages. Anthropic wants the system prompt separate,
        # and tool_result blocks wrapped in a user message.
        api_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue  # handled separately
            if m.role == "user":
                api_msgs.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                if m.tool_args is not None:
                    # An assistant tool-call turn — emit a tool_use block.
                    api_msgs.append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": m.content} if m.content else None,
                            {
                                "type": "tool_use",
                                "id": m.tool_call_id or "",
                                "name": m.tool_name or "",
                                "input": m.tool_args,
                            },
                        ],
                    })
                else:
                    api_msgs.append({"role": "assistant", "content": m.content})
            elif m.role == "tool":
                # Tool results go in a user-role message with tool_result block.
                api_msgs.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id or "",
                            "content": m.content,
                        }
                    ],
                })

        # Build kwargs for the SDK call.
        kwargs: dict[str, Any] = {
            "model": self.config.model or "claude-3-5-sonnet-latest",
            "max_tokens": self.config.max_tokens,
            "system": system_prompt,
            "messages": api_msgs,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if tools_schema:
            kwargs["tools"] = tools_schema

        resp = self._client.messages.create(**kwargs)

        # Walk the content blocks. We need to extract:
        #   - text → assistant prose
        #   - tool_use → tool calls (we encode them inline so the agent's
        #     parser picks them up via the same path as text-mode providers)
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                })

        # If the model returned native tool calls, we re-serialize them in the
        # ```tool_call fenced format the agent loop already parses. This keeps
        # the agent's dispatch path identical across all providers.
        content = "\n\n".join(t for t in text_parts if t).strip()
        if tool_calls:
            import json as _json
            tc_blocks = []
            for tc in tool_calls:
                tc_blocks.append(
                    "```tool_call\n"
                    f"{_json.dumps({'name': tc['name'], 'arguments': tc['input']}, ensure_ascii=False)}\n"
                    "```"
                )
            tc_text = "\n\n".join(tc_blocks)
            content = (content + "\n\n" + tc_text).strip() if content else tc_text

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=content,
            model=getattr(resp, "model", self.config.model),
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            raw={"native_tool_calls": tool_calls},
        )
