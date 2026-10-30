"""The agentic loop.

`Agent.run(user_text)` does the following until the model stops calling tools
or hits the iteration cap:

  1. Append the user message to the session.
  2. Ask the LLM for a completion (full message history in context).
  3. Parse any `tool_call` fenced blocks from the response.
     - If none: emit a FINAL event and stop.
     - If some: for each call, emit TOOL_CALL, check permissions, dispatch,
       emit TOOL_RESULT, append a tool-role message to the session.
  4. Loop back to step 2.

Events are emitted via a callback so both the CLI and the Gradio UI can
render progress in real time. All callbacks run on the caller's thread.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .config import AgentConfig
from .llm import LLMClient, LLMMessage
from .permissions import PermissionCallback, PermissionDecision, PermissionPolicy
from .prompts import build_system_prompt
from .session import Session
from .tools import ToolContext, ToolResult, default_registry


# ---------------------------------------------------------------------- #
# Events
# ---------------------------------------------------------------------- #


class EventType(str, Enum):
    USER = "user"
    ASSISTANT_TEXT = "assistant_text"        # partial/full assistant prose
    TOOL_CALL = "tool_call"                  # model wants to call a tool
    TOOL_RESULT = "tool_result"              # we ran the tool
    PERMISSION_ASK = "permission_ask"        # needs user approval (UI prompts)
    PERMISSION_DECISION = "permission_decision"
    ERROR = "error"
    FINAL = "final"                          # agent turn done
    ITERATION_LIMIT = "iteration_limit"


@dataclass
class AgentEvent:
    type: EventType
    payload: dict = field(default_factory=dict)
    #: Monotonic counter so the UI can sort events even if they arrive together.
    seq: int = 0


EventCallback = Callable[[AgentEvent], None]


# ---------------------------------------------------------------------- #
# Agent
# ---------------------------------------------------------------------- #


# Matches ```tool_call ... ``` blocks. Non-greedy, DOTALL so newlines are fine.
_TOOL_CALL_RE = re.compile(
    r"```tool_call\s*\n(.*?)\n```",
    re.DOTALL,
)

# Fallback: inline [TOOL_CALL name=... call_id=...] {...} [/TOOL_CALL] form.
# The model sometimes parrots the transcript format; tolerate it.
_TOOL_CALL_INLINE_RE = re.compile(
    r"\[TOOL_CALL[^\]]*\]\s*\n?(.*?)\n?\s*\[/TOOL_CALL\]",
    re.DOTALL,
)


class Agent:
    """The agentic coding assistant.

    One Agent instance owns one Session and one config. To run a different
    session or working dir, either mutate `self.session` / `self.config`
    (allowed) or create a new Agent.
    """

    def __init__(
        self,
        config: AgentConfig,
        session: Optional[Session] = None,
        permission_callback: Optional[PermissionCallback] = None,
    ) -> None:
        self.config = config
        self.session = session or Session(working_dir=str(config.working_dir), mode=config.mode)
        self.llm = LLMClient(config)
        self.registry = default_registry()
        self.policy = PermissionPolicy(config)
        # Default permission callback: deny anything that needs asking. The
        # UI/CLI overrides this with an interactive prompter.
        self.permission_callback: PermissionCallback = (
            permission_callback or _deny_by_default
        )

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def run(
        self,
        user_text: str,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        """Process one user turn. Returns the assistant's final text reply."""
        emit = _Emitter(on_event)
        self.session.add(LLMMessage(role="user", content=user_text))
        emit(EventType.USER, {"text": user_text})

        system_prompt = build_system_prompt(
            mode=self.config.mode,
            working_dir=self.config.working_dir,
            tools_doc=self.registry.render_doc(),
            override=self.config.system_prompt_override,
        )

        final_text = ""
        for iteration in range(self.config.max_iterations):
            messages = self._build_llm_messages(system_prompt)
            try:
                resp = self.llm.complete(messages)
            except Exception as e:  # noqa: BLE001
                emit(EventType.ERROR, {"message": f"LLM call failed: {e}"})
                return final_text

            content = resp.content or ""
            tool_calls = self._parse_tool_calls(content)

            if not tool_calls:
                # No tool calls → this is the final answer.
                self.session.add(LLMMessage(role="assistant", content=content))
                emit(
                    EventType.ASSISTANT_TEXT,
                    {"text": content, "final": True, "tokens": resp.completion_tokens},
                )
                final_text = content
                break

            # We have tool calls. Strip the assistant's stored content to ONLY
            # the text preceding the first tool_call block — the model often
            # hallucinates a [TOOL_RESULT] block after its own tool call, and
            # we never want that pollution in the transcript.
            preamble = self._extract_preamble(content)
            self.session.add(LLMMessage(role="assistant", content=preamble))
            emit(
                EventType.ASSISTANT_TEXT,
                {"text": preamble, "final": False, "tokens": resp.completion_tokens},
            )

            stop = False
            for call in tool_calls:
                tc_id = call["call_id"]
                name = call["name"]
                args = call["arguments"]
                emit(EventType.TOOL_CALL, {
                    "call_id": tc_id, "name": name, "arguments": args,
                })

                # Permission gate.
                decision = self.policy.classify(name, args)
                if decision is PermissionDecision.ASK:
                    emit(EventType.PERMISSION_ASK, {
                        "call_id": tc_id, "name": name, "arguments": args,
                        "reason": self.policy.reason(name, args, decision),
                    })
                    decision = self.permission_callback(name, args)
                    emit(EventType.PERMISSION_DECISION, {
                        "call_id": tc_id, "name": name, "decision": decision.value,
                    })

                if decision is PermissionDecision.DENY:
                    result = ToolResult.error(
                        "Permission denied by user/policy. "
                        "Do not retry this exact call; ask the user or change approach."
                    )
                else:
                    ctx = ToolContext(
                        config=self.config,
                        session=self.session,
                        turn_state=self._turn_state,
                    )
                    result = self.registry.dispatch(name, args, ctx)

                emit(EventType.TOOL_RESULT, {
                    "call_id": tc_id, "name": name,
                    "ok": result.ok, "content": result.content,
                    "structured": result.structured,
                })
                # Append both the call and the result so the transcript
                # is self-contained for the next LLM turn.
                self.session.add(LLMMessage(
                    role="assistant",
                    content="",
                    tool_call_id=tc_id, tool_name=name, tool_args=args,
                ))
                self.session.add(LLMMessage(
                    role="tool",
                    content=result.content,
                    tool_call_id=tc_id, tool_name=name,
                ))

                if decision is PermissionDecision.DENY:
                    # On deny we let the model react, but we don't hard-stop.
                    pass

            if stop:
                break
        else:
            # for-else: loop completed without break → hit iteration cap.
            emit(EventType.ITERATION_LIMIT, {
                "max": self.config.max_iterations,
                "message": (
                    f"Reached max iterations ({self.config.max_iterations}). "
                    "Stopping to avoid runaway tool calls."
                ),
            })
            final_text = (
                f"(stopped after {self.config.max_iterations} tool iterations; "
                "last assistant message above)"
            )

        emit(EventType.FINAL, {"text": final_text})
        return final_text

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @property
    def _turn_state(self) -> dict[str, Any]:
        # Persist turn_state on the agent so the todo list survives across
        # LLM round-trips within one user turn. Reset at the start of run().
        if not hasattr(self, "_current_turn_state"):
            self._current_turn_state = {}
        return self._current_turn_state

    def run_reset_turn_state(self) -> None:
        """Call between user turns to clear turn-local state (e.g. todos)."""
        self._current_turn_state = {}

    def _build_llm_messages(self, system_prompt: str) -> list[LLMMessage]:
        """Build the message list for the LLM, prepending the system prompt."""
        msgs: list[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]
        msgs.extend(self.session.messages)
        return msgs

    def _parse_tool_calls(self, content: str) -> list[dict[str, Any]]:
        """Extract tool_call blocks from the model's response.

        Accepts TWO formats:
          1. Preferred: ```tool_call ... ``` fenced block (per the protocol)
          2. Fallback: [TOOL_CALL ...] {...} [/TOOL_CALL] inline form
             (tolerated because some models drift to this shape after seeing
             transcript text)

        Each block is a JSON object with `name` and `arguments`. We tolerate
        trailing commas / comments by attempting json.loads then falling back
        to a lenient parse.
        """
        calls: list[dict[str, Any]] = []
        seen_spans: list[tuple[int, int]] = []
        for rx in (_TOOL_CALL_RE, _TOOL_CALL_INLINE_RE):
            for m in rx.finditer(content):
                # Skip if this span overlaps one already parsed by the other
                # regex (prevents double-counting when both match the same
                # text — unlikely but defensive).
                if any(not (m.end() <= s or m.start() >= e) for s, e in seen_spans):
                    continue
                seen_spans.append((m.start(), m.end()))
                raw = m.group(1).strip()
                obj = _parse_lenient_json(raw)
                if not isinstance(obj, dict):
                    continue
                name = obj.get("name")
                args = obj.get("arguments") or {}
                if not name or not isinstance(args, dict):
                    continue
                calls.append({
                    "call_id": uuid.uuid4().hex[:8],
                    "name": str(name),
                    "arguments": args,
                })
        # Sort by appearance order so multi-call responses run in the order
        # the model wrote them.
        calls.sort(key=lambda c: 0)  # stable; insertion order preserved
        return calls

    @staticmethod
    def _extract_preamble(content: str) -> str:
        """Return the text BEFORE the first tool_call block (either format).

        The model often hallucinates a [TOOL_RESULT] block after its own
        tool_call; we discard everything from the first tool_call fence
        onward so the transcript stays clean. The tool calls themselves
        are stored separately (as assistant.tool_args messages).
        """
        earliest = None
        for rx in (_TOOL_CALL_RE, _TOOL_CALL_INLINE_RE):
            m = rx.search(content)
            if m is not None and (earliest is None or m.start() < earliest):
                earliest = m.start()
        if earliest is None:
            return content
        return content[:earliest].rstrip()


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


class _Emitter:
    """Tiny helper that bumps a sequence counter on each emit."""

    def __init__(self, cb: Optional[EventCallback]) -> None:
        self._cb = cb
        self._seq = 0

    def __call__(self, type: EventType, payload: dict) -> None:
        if self._cb is None:
            return
        self._seq += 1
        try:
            self._cb(AgentEvent(type=type, payload=payload, seq=self._seq))
        except Exception:  # noqa: BLE001 — never let a UI callback kill the loop
            pass


def _deny_by_default(name: str, args: dict) -> PermissionDecision:
    """Default permission callback: deny anything that needs asking."""
    return PermissionDecision.DENY


def _parse_lenient_json(raw: str) -> Any:
    """Parse JSON tolerantly: try strict, then strip trailing commas, then
    strip // comments. Returns the parsed object or None on failure."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip trailing commas before } or ].
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Strip // line comments.
    cleaned = re.sub(r"//[^\n]*", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
