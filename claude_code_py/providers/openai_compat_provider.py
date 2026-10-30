"""OpenAI-compatible provider.

Covers any backend that speaks the OpenAI Chat Completions API:
  - DeepSeek       (base_url: https://api.deepseek.com/v1)
  - Google Gemini  (base_url: https://generativelanguage.googleapis.com/v1beta/openai/)
  - HuggingFace    (base_url: https://api-inference.huggingface.co/v1)
  - Custom         (user-supplied base_url)

Tool calling: we attempt the SDK's native `tools=` parameter when
`config.native_tools` is True. If the upstream returns tool_calls in the
response, we re-serialize them as ```tool_call fenced blocks so the agent
loop's single dispatch path works. If the upstream doesn't support tools,
we fall back to the structured-output protocol (model emits fenced blocks
in prose).
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..messages import LLMMessage, LLMResponse
from . import ProviderConfig, env_key_for, provider_meta


class OpenAICompatProvider:
    """One class for every OpenAI-compatible backend."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        pid = config.provider
        meta = provider_meta(pid)
        # Resolve API key: explicit > env var named in the provider metadata.
        env_var = meta.get("env_key", "")
        key = config.api_key or (os.environ.get(env_var) if env_var else "")
        if not key:
            raise RuntimeError(
                f"No API key for provider {pid!r}. "
                f"Pass api_key= in ProviderConfig or export {env_var}."
            )
        # Resolve base_url: explicit > provider default.
        base_url = config.base_url or meta.get("base_url", "") or None

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "The `openai` package is required for the "
                f"{pid} provider. Install it with: pip install openai"
            ) from e

        kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._pid = pid

    def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        tools_schema: Optional[list[dict]] = None,
    ) -> LLMResponse:
        # Translate messages to OpenAI format. The system prompt goes in as
        # the first system message.
        api_msgs: list[dict[str, Any]] = []
        if system_prompt:
            api_msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            if m.role == "system":
                continue  # already prepended
            if m.role == "tool":
                # OpenAI tool result message
                api_msgs.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": m.content,
                })
            elif m.role == "assistant" and m.tool_args is not None:
                # Assistant tool-call turn
                api_msgs.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [{
                        "id": m.tool_call_id or "",
                        "type": "function",
                        "function": {
                            "name": m.tool_name or "",
                            "arguments": json.dumps(m.tool_args, ensure_ascii=False),
                        },
                    }],
                })
            else:
                api_msgs.append({"role": m.role, "content": m.content})

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_msgs,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        # HuggingFace's TGI endpoint and some custom backends don't accept the
        # tools= parameter; only pass it when we have a schema and the user
        # has opted in.
        if tools_schema and self.config.native_tools:
            kwargs["tools"] = tools_schema

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # If tools aren't supported, retry once without them.
            if "tools" in kwargs and "tool" in str(e).lower():
                kwargs.pop("tools", None)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise

        choice = (resp.choices or [None])[0]
        if choice is None:
            return LLMResponse(content="", model=self.config.model)
        msg = getattr(choice, "message", None)
        content = getattr(msg, "content", "") or ""
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Re-serialize native tool_calls into the fenced format the agent
        # loop parses.
        if tool_calls:
            tc_blocks = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                name = getattr(fn, "name", "")
                args_raw = getattr(fn, "arguments", "{}") or "{}"
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {"_raw": args_raw}
                tc_id = getattr(tc, "id", "")
                tc_blocks.append(
                    "```tool_call\n"
                    f"{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n"
                    "```"
                )
                # Stash the call_id so the agent can match results back.
                tc_blocks.append(f"<!-- call_id: {tc_id} -->")
            tc_text = "\n\n".join(tc_blocks)
            content = (content + "\n\n" + tc_text).strip() if content else tc_text

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=content,
            model=getattr(resp, "model", self.config.model),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            raw={},
        )
