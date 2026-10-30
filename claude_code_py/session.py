"""Session persistence — save/load/list conversations as JSON files.

Each session file is a single JSON document with:
  - meta: id, created_at, updated_at, working_dir, mode, title
  - messages: list of LLMMessage dicts (role/content/tool metadata)
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from .llm import LLMMessage


@dataclass
class SessionMeta:
    id: str
    title: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    working_dir: str = ""
    mode: str = "default"
    message_count: int = 0


@dataclass
class Session:
    """A single conversation. Holds messages and metadata."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New session"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    working_dir: str = ""
    mode: str = "default"
    messages: list[LLMMessage] = field(default_factory=list)

    # ---- convenience ----

    def add(self, msg: LLMMessage) -> None:
        self.messages.append(msg)
        self.updated_at = time.time()
        # Auto-title from the first user message.
        if self.title == "New session" and msg.role == "user":
            self.title = msg.content.strip().splitlines()[0][:60]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "working_dir": self.working_dir,
            "mode": self.mode,
            "messages": [_msg_to_dict(m) for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        msgs = [_msg_from_dict(m) for m in d.get("messages", [])]
        return cls(
            id=d.get("id") or uuid.uuid4().hex[:12],
            title=d.get("title", "New session"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            working_dir=d.get("working_dir", ""),
            mode=d.get("mode", "default"),
            messages=msgs,
        )

    def meta(self) -> SessionMeta:
        return SessionMeta(
            id=self.id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            working_dir=self.working_dir,
            mode=self.mode,
            message_count=len(self.messages),
        )


# ---------------------------------------------------------------------- #
# Store — load/save/list sessions from disk
# ---------------------------------------------------------------------- #


class SessionStore:
    """Filesystem-backed session storage. One JSON file per session."""

    def __init__(self, sessions_dir: Path | str) -> None:
        self.dir = Path(sessions_dir).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        # Sanitize: only allow hex/alnum ids.
        safe = "".join(c for c in session_id if c.isalnum())
        return self.dir / f"{safe}.json"

    def save(self, session: Session) -> Path:
        p = self._path(session.id)
        p.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p

    def load(self, session_id: str) -> Optional[Session]:
        p = self._path(session_id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return Session.from_dict(data)

    def delete(self, session_id: str) -> bool:
        p = self._path(session_id)
        if p.exists():
            p.unlink()
            return True
        return False

    def list(self) -> list[SessionMeta]:
        metas: list[SessionMeta] = []
        for p in sorted(self.dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            s = Session.from_dict(data)
            metas.append(s.meta())
        return metas


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _msg_to_dict(m: LLMMessage) -> dict[str, Any]:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.tool_name:
        d["tool_name"] = m.tool_name
    if m.tool_args is not None:
        d["tool_args"] = m.tool_args
    return d


def _msg_from_dict(d: dict[str, Any]) -> LLMMessage:
    return LLMMessage(
        role=d.get("role", "user"),
        content=d.get("content", ""),
        tool_call_id=d.get("tool_call_id"),
        tool_name=d.get("tool_name"),
        tool_args=d.get("tool_args"),
    )
