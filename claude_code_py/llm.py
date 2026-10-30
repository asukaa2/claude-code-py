"""LLM client wrapping the `z-ai chat` CLI.

The CLI accepts a single user prompt + an optional system prompt and returns
an OpenAI-compatible JSON document. We use it as a subprocess so the rest of
the package stays pure-Python and dependency-free (no SDK install needed).

We deliberately do NOT use native function-calling here — instead, the agent
loop instructs the model to emit tool calls in a fenced JSON block, which we
parse. This keeps the LLM backend swappable.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import AgentConfig


@dataclass
class LLMMessage:
    """A single chat message.

    Roles: "system" | "user" | "assistant" | "tool".
    For "tool" messages, `content` is the tool result and `tool_call_id` /
    `tool_name` carry metadata.
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


class LLMClient:
    """Subprocess-based client around the `z-ai chat` CLI.

    The CLI only accepts ONE user prompt + ONE system prompt per call, so to
    support multi-turn + tool calls we serialize the full message history into
    a single system prompt (conversation transcript) and send the latest user
    turn as the `--prompt`. This is portable and works with any chat backend.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        """Send `messages` to the LLM and return the assistant's reply.

        We collapse all messages into a transcript-style system prompt and
        use the last user/tool message as the active `--prompt`. This works
        around the CLI's single-turn interface.
        """
        system_prompt, user_prompt = self._split_messages(messages)
        return self._call_cli(system_prompt, user_prompt)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_messages(
        messages: list[LLMMessage],
    ) -> tuple[str, str]:
        """Flatten a multi-turn message list into (system, user_prompt).

        - The very first `system` message (if any) becomes the head of the
          system prompt.
        - All subsequent messages are rendered as a transcript.
        - The LAST message becomes the active user prompt (the question the
          model should answer right now).
        """
        if not messages:
            return "", ""

        head_system = ""
        rest: list[LLMMessage] = []
        if messages[0].role == "system":
            head_system = messages[0].content
            rest = messages[1:]
        else:
            rest = list(messages)

        if not rest:
            return head_system, ""

        # Render all but the last as transcript lines.
        transcript_lines: list[str] = []
        for m in rest[:-1]:
            transcript_lines.append(_render_transcript_line(m))

        last = rest[-1]
        # If the last message is a tool result, render it as the active prompt
        # so the model knows to continue from that observation.
        last_prompt = _render_transcript_line(last, is_active=True)

        parts: list[str] = []
        if head_system:
            parts.append(head_system)
        if transcript_lines:
            parts.append(
                "The following is the conversation so far (for your context). "
                "Continue from here.\n\n" + "\n".join(transcript_lines)
            )
        parts.append(last_prompt)

        system_prompt = "\n\n---\n\n".join(parts)
        return system_prompt, last_prompt

    def _call_cli(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Invoke the z-ai CLI and parse its JSON output."""
        if not user_prompt.strip():
            # Defensive: never send an empty prompt to the CLI.
            user_prompt = "(continue)"

        with self._lock:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", delete=False, encoding="utf-8"
            ) as tmp:
                out_path = tmp.name

            try:
                cmd: list[str] = [
                    self.config.llm_cli,
                    "chat",
                    "--prompt", user_prompt,
                    "--output", out_path,
                ]
                if system_prompt.strip():
                    cmd += ["--system", system_prompt]
                if self.config.enable_thinking:
                    cmd += ["--thinking"]

                env = os.environ.copy()
                env.update(self.config.bash_env)

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=300,
                    check=False,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"LLM CLI failed (exit {proc.returncode}):\n"
                        f"stderr: {proc.stderr[:1000]}\nstdout: {proc.stdout[:500]}"
                    )

                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            finally:
                try:
                    os.unlink(out_path)
                except OSError:
                    pass

        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content", "") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            content=content,
            model=data.get("model", self.config.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _render_transcript_line(msg: LLMMessage, is_active: bool = False) -> str:
    """Render a single message as a transcript line for the system prompt.

    Tool calls are rendered using the SAME fenced ```tool_call block format
    the model is asked to emit — this keeps the format consistent and prevents
    the model from drifting to a different shape.
    """
    role = msg.role.upper()
    if msg.role == "tool":
        body = (
            f"[TOOL_RESULT tool={msg.tool_name} call_id={msg.tool_call_id}]\n"
            f"{msg.content}\n[/TOOL_RESULT]"
        )
    elif msg.role == "assistant" and msg.tool_args is not None:
        import json as _json
        body = (
            f"```tool_call\n"
            f"{_json.dumps({'name': msg.tool_name, 'arguments': msg.tool_args}, ensure_ascii=False)}\n"
            f"```"
        )
        if msg.content:
            body = msg.content + "\n\n" + body
    else:
        body = msg.content

    prefix = ">>> " if is_active else ""
    return f"{prefix}{role}:\n{body}"
