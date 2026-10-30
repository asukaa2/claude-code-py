"""
Claude Code (Python edition) — Gradio web UI.

A complete browser-based control surface for the agentic coding assistant,
easier to drive than the CLI. Run with:

    python webui.py
    python webui.py --share            # public Gradio link
    python webui.py --port 7860

Layout:
  ┌──────────────┬─────────────────────────────┬──────────────────┐
  │  Sidebar     │  Chat (main)                │  Right panel     │
  │  - sessions  │  - message list             │  - tool calls    │
  │  - work dir  │  - input box                │  - todos         │
  │  - mode      │  - send/stop buttons        │  - file browser  │
  │  - perms     │                             │                  │
  │  - settings  │                             │                  │
  └──────────────┴─────────────────────────────┴──────────────────┘
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import gradio as gr

from claude_code_py import (
    Agent,
    AgentConfig,
    AgentEvent,
    EventType,
    Session,
    SessionStore,
    ProviderConfig,
    KNOWN_PROVIDERS,
    PROVIDER_IDS,
    default_model_for,
    env_key_for,
    get_default_config,
)
from claude_code_py.permissions import PermissionDecision
from claude_code_py.tools import default_registry

# Brand colors for the UI theme.
PRIMARY = "#6C4EB1"        # muted purple, evokes Claude
PRIMARY_DARK = "#3D2B66"
ACCENT = "#D97757"         # warm orange (Claude-ish)
BG = "#FAF9F7"
SURFACE = "#FFFFFF"
BORDER = "#E5E1D8"
TEXT = "#1F1B2E"
MUTED = "#6B6478"

CSS = f"""
:root {{
    --cc-primary: {PRIMARY};
    --cc-primary-dark: {PRIMARY_DARK};
    --cc-accent: {ACCENT};
    --cc-bg: {BG};
    --cc-surface: {SURFACE};
    --cc-border: {BORDER};
    --cc-text: {TEXT};
    --cc-muted: {MUTED};
}}
.gradio-container {{ max-width: 100% !important; padding: 0 !important; }}
#cc-topbar {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
    color: white; padding: 14px 22px; border-radius: 0 0 14px 14px;
    display: flex; align-items: center; gap: 14px;
    box-shadow: 0 2px 12px rgba(108,78,177,0.25);
}}
#cc-topbar h1 {{ margin: 0; font-size: 20px; font-weight: 700; color: white; }}
#cc-topbar .sub {{ font-size: 12px; opacity: 0.85; }}
#cc-topbar .badge {{
    background: rgba(255,255,255,0.18); padding: 3px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 600;
}}
#cc-body {{ padding: 14px; }}
.cc-panel {{
    background: var(--cc-surface); border: 1px solid var(--cc-border);
    border-radius: 10px; padding: 12px;
}}
.cc-panel h3 {{
    margin: 0 0 8px 0; font-size: 13px; font-weight: 700;
    color: var(--cc-primary-dark); text-transform: uppercase;
    letter-spacing: 0.05em;
}}
#cc-chatbot {{ min-height: 480px; }}
#cc-chatbot .message {{ font-size: 14px; line-height: 1.55; }}
#cc-chatbot .user {{ background: #EFEAFB !important; border-color: #C9B8F0 !important; }}
#cc-chatbot .assistant {{ background: #FFF8F3 !important; border-color: #F4D5C2 !important; }}
#cc-tool-log {{ max-height: 480px; overflow-y: auto; }}
.cc-tool-row {{
    border-left: 3px solid var(--cc-primary); padding: 6px 10px; margin: 6px 0;
    background: #FAFAFC; border-radius: 4px; font-size: 12px;
}}
.cc-tool-row.err {{ border-left-color: #C8364B; background: #FDF2F4; }}
.cc-tool-row .head {{ font-weight: 700; color: var(--cc-primary-dark); }}
.cc-tool-row .args {{ color: var(--cc-muted); font-family: monospace; font-size: 11px; }}
.cc-tool-row .result {{ color: #333; font-family: monospace; font-size: 11px;
    white-space: pre-wrap; max-height: 180px; overflow-y: auto; margin-top: 4px;
    background: white; padding: 6px; border-radius: 3px; border: 1px solid var(--cc-border); }}
.cc-todo {{ padding: 4px 8px; font-size: 12px; border-radius: 4px; margin: 3px 0; }}
.cc-todo.completed {{ color: #2D7A3E; text-decoration: line-through; opacity: 0.7; }}
.cc-todo.in_progress {{ background: #FFF1D6; color: #8C5A00; font-weight: 600; }}
.cc-todo.pending {{ color: var(--cc-muted); }}
.cc-stat {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; margin-right: 6px; }}
.cc-stat.ok {{ background: #D6F0DC; color: #1B5E20; }}
.cc-stat.err {{ background: #FAD7DC; color: #8E1F2F; }}
.cc-stat.info {{ background: #E3DAF7; color: #3D2B66; }}
.cc-perm-banner {{
    background: #FFF8E1; border: 1px solid #F4D160; border-radius: 6px;
    padding: 8px 12px; font-size: 12px; color: #6B4D00; margin: 4px 0;
}}
footer {{ display: none !important; }}
"""

# Modes offered in the UI dropdown.
MODES = [
    ("Default", "default"),
    ("Plan (read-only)", "plan"),
    ("Code (act fast)", "code"),
    ("Explain (read-only)", "explain"),
]

# Permission presets offered as radio options.
PERM_PRESETS = [
    ("Trust reads", "trust_reads"),
    ("Yolo (auto-all)", "yolo"),
    ("Manual (ask all)", "manual"),
]


# ====================================================================== #
# App state — one instance per running webui process (single-user model)
# ====================================================================== #


class AppState:
    """Holds the live Agent + Session + UI-facing buffers.

    Gradio handlers read/mutate this. We keep everything in one object so the
    handlers stay small and the wiring is obvious.
    """

    def __init__(self) -> None:
        self.config = get_default_config()
        self.session = Session(
            working_dir=str(self.config.working_dir),
            mode=self.config.mode,
        )
        self.store = SessionStore(self.config.sessions_dir)
        self.agent: Agent | None = None
        self.events: list[AgentEvent] = []
        self.chat_history: list[dict] = []  # OpenAI-style {role, content}
        self.tool_log_html: str = ""
        self.todo_html: str = ""
        self.perm_mode: str = "trust_reads"
        self.stop_requested: bool = False
        self.apply_perm_preset(self.perm_mode)

    # ---- permission presets ----

    def apply_perm_preset(self, preset: str) -> None:
        self.perm_mode = preset
        if preset == "yolo":
            self.config.auto_approve_read_tools = True
            self.config.auto_approve_write_tools = True
            self.config.auto_approve_bash = True
        elif preset == "trust_reads":
            self.config.auto_approve_read_tools = True
            self.config.auto_approve_write_tools = False
            self.config.auto_approve_bash = False
        else:  # manual
            self.config.auto_approve_read_tools = False
            self.config.auto_approve_write_tools = False
            self.config.auto_approve_bash = False

    def permission_callback(self, name: str, args: dict) -> PermissionDecision:
        """Called from inside agent.run() (on a worker thread).

        Because we can't easily show a modal mid-run in Gradio, we treat the
        permission mode as authoritative: anything not auto-approved is denied
        with a notice surfaced in the chat. The user can flip the preset and
        re-send the message.
        """
        return PermissionDecision.DENY

    # ---- agent lifecycle ----

    def rebuild_agent(self) -> Agent:
        self.agent = Agent(
            self.config,
            session=self.session,
            permission_callback=self.permission_callback,
        )
        return self.agent


STATE = AppState()


# ====================================================================== #
# Event handling — bridge between the agent worker thread and the UI
# ====================================================================== #


class RunHandle:
    """Per-run bookkeeping shared between the worker thread and the generator."""

    def __init__(self) -> None:
        self.events_q: queue.Queue[AgentEvent | None] = queue.Queue()
        self.done = threading.Event()
        self.final_text = ""


def _worker(
    agent: Agent,
    user_text: str,
    handle: RunHandle,
) -> None:
    """Runs in a background thread; pushes events onto the queue."""
    def on_event(e: AgentEvent) -> None:
        handle.events_q.put(e)
    try:
        agent.run_reset_turn_state()
        text = agent.run(user_text, on_event=on_event)
        handle.final_text = text
    except Exception as e:  # noqa: BLE001
        # Synthesize an error event so the UI shows something.
        handle.events_q.put(AgentEvent(
            type=EventType.ERROR,
            payload={"message": f"Agent crashed: {type(e).__name__}: {e}"},
        ))
    finally:
        handle.done.set()
        handle.events_q.put(None)  # sentinel


def _render_tool_log(events: list[AgentEvent]) -> str:
    """Render the tool-call log as HTML for the right panel."""
    rows: list[str] = []
    for e in events:
        if e.type is EventType.TOOL_CALL:
            args_str = _fmt_args(e.payload["arguments"])
            rows.append(
                f'<div class="cc-tool-row" id="tc-{e.payload["call_id"]}">'
                f'<div class="head">→ {e.payload["name"]}</div>'
                f'<div class="args">{args_str}</div>'
                f'<div class="result" id="tc-result-{e.payload["call_id"]}">'
                f'<span style="color:#999">running…</span></div>'
                f'</div>'
            )
        elif e.type is EventType.TOOL_RESULT:
            cid = e.payload["call_id"]
            ok = e.payload["ok"]
            content = _esc(e.payload["content"])
            cls = "" if ok else "err"
            # Update the matching tool-call row's result div.
            rows.append(
                f'<script>/*patch*/</script>'
                f'<div id="patch-{cid}" data-target="tc-result-{cid}" '
                f'data-cls="{"err" if not ok else ""}" data-content="{_attr(content)}">'
                f'</div>'
            )
        elif e.type is EventType.PERMISSION_ASK:
            rows.append(
                f'<div class="cc-perm-banner">⚠ Permission needed for '
                f'<b>{e.payload["name"]}</b> — denied (toggle a permissive '
                f'preset in the sidebar to allow).</div>'
            )
    if not rows:
        return '<div style="color:#999;font-size:12px;padding:8px;">No tool calls yet.</div>'
    # Note: data-* attributes are read by a small JS patcher injected below.
    return (
        '<div id="cc-tool-log">'
        + "\n".join(rows)
        + '<script>' + _PATCH_JS + '</script>'
        + '</div>'
    )


# Tiny JS that applies the "patch" divs to their target result divs.
# This lets us update a tool row's result in place when the result arrives
# in a subsequent render.
_PATCH_JS = r"""
(function(){
  document.querySelectorAll('div[id^="patch-"]').forEach(function(p){
    var tgt = document.getElementById(p.dataset.target);
    if (!tgt) return;
    if (p.dataset.cls) { tgt.parentElement.classList.add('err'); }
    var c = p.dataset.content || '';
    // decode HTML entities encoded by _attr
    var txt = c.replace(/&amp;quot;/g,'"').replace(/&amp;#39;/g,"'")
               .replace(/&amp;lt;/g,'<').replace(/&amp;gt;/g,'>')
               .replace(/&amp;amp;/g,'&');
    tgt.textContent = txt;
    p.remove();
  });
})();
"""


def _render_todos(events: list[AgentEvent]) -> str:
    """Render the latest todo list as HTML."""
    todos: list[dict] | None = None
    for e in reversed(events):
        if e.type is EventType.TOOL_RESULT and e.payload.get("name") == "todo_write":
            todos = (e.payload.get("structured") or {}).get("todos")
            if todos is not None:
                break
    if not todos:
        return '<div style="color:#999;font-size:12px;padding:8px;">No todos set.</div>'
    rows = []
    for t in todos:
        cls = t.get("status", "pending")
        mark = {"completed": "✓", "in_progress": "▸", "pending": "○"}.get(cls, "○")
        rows.append(
            f'<div class="cc-todo {cls}">{mark} {_esc(t.get("content",""))}</div>'
        )
    return '<div id="cc-todos">' + "\n".join(rows) + '</div>'


def _fmt_args(args: dict) -> str:
    """Compact one-line representation of tool args for the log."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "…"
        s = s.replace("\n", " ")
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _attr(s: str) -> str:
    """Escape for use inside an HTML attribute (double-quoted)."""
    return _esc(s).replace('"', "&quot;").replace("'", "&#39;")


def _events_to_chat(
    events: list[AgentEvent],
    base_history: list[dict],
) -> list[dict]:
    """Build the chatbot message list from events + base history.

    We use base_history for prior turns (already finalized) and reconstruct
    the in-flight turn from events.
    """
    out: list[dict] = list(base_history)
    # Walk events of the current turn and produce interim messages.
    current_assistant: list[str] = []
    for e in events:
        if e.type is EventType.USER:
            out.append({"role": "user", "content": e.payload["text"]})
        elif e.type is EventType.ASSISTANT_TEXT:
            current_assistant.append(e.payload["text"])
        elif e.type is EventType.TOOL_CALL:
            # Flush any pending assistant prose as its own bubble.
            if current_assistant:
                out.append({"role": "assistant", "content": "\n\n".join(current_assistant)})
                current_assistant = []
            name = e.payload["name"]
            args = e.payload["arguments"]
            arg_str = ", ".join(f"`{k}`={_short(v)}" for k, v in args.items())
            out.append({
                "role": "assistant",
                "content": f"🔧 **Tool call:** `{name}`({arg_str})",
            })
        elif e.type is EventType.TOOL_RESULT:
            ok = e.payload["ok"]
            name = e.payload["name"]
            content = e.payload["content"]
            snippet = "\n".join(content.splitlines()[:15])
            if len(content.splitlines()) > 15:
                snippet += "\n…"
            icon = "✅" if ok else "❌"
            out.append({
                "role": "user",
                "content": f"{icon} **{name} result:**\n```\n{snippet}\n```",
            })
        elif e.type is EventType.PERMISSION_ASK:
            out.append({
                "role": "assistant",
                "content": (
                    f"⚠️ I want to call `{e.payload['name']}` but the current "
                    "permission preset would deny it. Switch to a more "
                    "permissive preset in the sidebar and re-send if you want."
                ),
            })
        elif e.type is EventType.ERROR:
            out.append({"role": "assistant", "content": f"⚠️ Error: {e.payload['message']}"})
        elif e.type is EventType.ITERATION_LIMIT:
            out.append({"role": "assistant", "content": f"⏹ {e.payload['message']}"})
    if current_assistant:
        out.append({"role": "assistant", "content": "\n\n".join(current_assistant)})
    return out


def _short(v: Any) -> str:
    s = str(v)
    if len(s) > 40:
        s = s[:37] + "…"
    return s.replace("\n", " ")


def _stat_pill(ok: bool, text: str) -> str:
    cls = "ok" if ok else "err"
    return f'<span class="cc-stat {cls}">{text}</span>'


# ====================================================================== #
# Gradio handlers
# ====================================================================== #


def h_connect_working_dir(path: str):
    """Update the working directory."""
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return (
            gr.update(),
            f'<div class="cc-perm-banner">⚠ Not a directory: {p}</div>',
        )
    STATE.config.working_dir = p
    STATE.session.working_dir = str(p)
    return (
        gr.update(value=str(p)),
        f'<div class="cc-stat info">working dir: {p}</div>',
    )


def h_set_mode(mode_value: str):
    STATE.config.mode = mode_value
    STATE.session.mode = mode_value
    label = dict((v, k) for k, v in MODES)[mode_value]
    return f'<div class="cc-stat info">mode: {label}</div>'


def h_set_perm_preset(preset: str):
    STATE.apply_perm_preset(preset)
    label = dict((v, k) for k, v in PERM_PRESETS)[preset]
    banner = (
        f'<div class="cc-stat info">permission preset: {label}</div>'
        f'<div style="font-size:11px;color:{MUTED};margin-top:4px;">'
        f'reads={STATE.config.auto_approve_read_tools} · '
        f'writes={STATE.config.auto_approve_write_tools} · '
        f'bash={STATE.config.auto_approve_bash}</div>'
    )
    return banner


# ---------------------------------------------------------------------- #
# Provider handlers — switch LLM backend at runtime
# ---------------------------------------------------------------------- #


def _provider_label(pid: str) -> str:
    for p in KNOWN_PROVIDERS:
        if p["id"] == pid:
            return p["label"]
    return pid


def h_set_provider(
    provider_id: str,
    api_key: str,
    model: str,
    base_url: str,
):
    """Apply provider settings and rebuild the agent's LLM client."""
    pid = provider_id or "anthropic"
    # Resolve the API key: explicit field > env var > empty.
    env_var = env_key_for(pid)
    resolved_key = (api_key or "").strip()
    if not resolved_key and env_var:
        import os
        resolved_key = os.environ.get(env_var, "")
    # Resolve model: explicit > provider default.
    resolved_model = (model or "").strip() or default_model_for(pid)
    # Resolve base_url: explicit > provider default (only for custom do we
    # let the user type one).
    resolved_base_url = (base_url or "").strip()
    if not resolved_base_url:
        for p in KNOWN_PROVIDERS:
            if p["id"] == pid:
                resolved_base_url = p.get("base_url", "")
                break

    STATE.config.provider = ProviderConfig(
        provider=pid,
        api_key=resolved_key,
        model=resolved_model,
        base_url=resolved_base_url,
    )
    # Rebuild the agent if it exists, so the next run() picks up the new
    # provider immediately.
    if STATE.agent is not None:
        try:
            STATE.agent.rebuild_llm()
        except Exception as e:  # noqa: BLE001
            return f'<div class="cc-stat err">provider error: {e}</div>'

    masked = (resolved_key[:4] + "…" + resolved_key[-4:]) if len(resolved_key) > 8 else ("set" if resolved_key else "<none>")
    banner = (
        f'<div class="cc-stat info">provider: {_provider_label(pid)}</div>'
        f'<div style="font-size:11px;color:{MUTED};margin-top:4px;">'
        f'model: <code>{resolved_model or "(none)"}</code> · '
        f'key: <code>{masked}</code>'
        + (f' · base_url: <code>{resolved_base_url}</code>' if resolved_base_url else "")
        + '</div>'
    )
    return banner


def h_provider_defaults(provider_id: str):
    """When the user picks a new provider, populate model + base_url defaults.

    Returns updates for the model textbox, base_url textbox, and a hint about
    which env var holds the API key.
    """
    pid = provider_id or "anthropic"
    model = default_model_for(pid)
    base_url = ""
    env_var = env_key_for(pid)
    for p in KNOWN_PROVIDERS:
        if p["id"] == pid:
            base_url = p.get("base_url", "")
            break
    hint = (
        f"API key (env var: <code>{env_var}</code>)" if env_var
        else "API key (no env var — type below)"
    )
    # For non-custom providers, base_url is read-only (we still show it as a
    # hint, but the field is non-interactive). For custom, it's editable.
    base_url_interactive = (pid == "custom")
    return (
        gr.update(value=model),
        gr.update(value=base_url, interactive=base_url_interactive),
        hint,
    )


def h_new_session():
    STATE.session = Session(
        working_dir=str(STATE.config.working_dir),
        mode=STATE.config.mode,
    )
    STATE.events = []
    STATE.chat_history = []
    STATE.tool_log_html = ""
    STATE.todo_html = ""
    return (
        [],  # chatbot
        "",  # tool log
        "",  # todos
        f'<div class="cc-stat info">new session: {STATE.session.id}</div>',
        _session_dropdown_options(STATE.session.id),
    )


def h_load_session(session_id: str):
    if not session_id:
        return (gr.update(), gr.update(), gr.update(),
                '<div class="cc-stat err">no session selected</div>', gr.update())
    s = STATE.store.load(session_id)
    if s is None:
        return (gr.update(), gr.update(), gr.update(),
                f'<div class="cc-stat err">could not load {session_id}</div>', gr.update())
    STATE.session = s
    STATE.config.mode = s.mode
    STATE.config.working_dir = Path(s.working_dir) if s.working_dir else STATE.config.working_dir
    STATE.events = []
    # Rebuild chat history from session messages.
    history: list[dict] = []
    for m in s.messages:
        if m.role == "user":
            history.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if m.tool_args is not None:
                # Tool-call message — render compactly.
                arg_str = ", ".join(f"`{k}`={_short(v)}" for k, v in m.tool_args.items())
                history.append({
                    "role": "assistant",
                    "content": f"🔧 **Tool call:** `{m.tool_name}`({arg_str})",
                })
            elif m.content:
                history.append({"role": "assistant", "content": m.content})
        elif m.role == "tool":
            snippet = "\n".join(m.content.splitlines()[:15])
            history.append({
                "role": "user",
                "content": f"↩ **{m.tool_name} result:**\n```\n{snippet}\n```",
            })
    STATE.chat_history = history
    return (
        history,
        "",
        "",
        f'<div class="cc-stat info">loaded session: {s.title} ({len(s.messages)} msgs)</div>',
        gr.update(),
    )


def h_save_session():
    p = STATE.store.save(STATE.session)
    return (
        f'<div class="cc-stat ok">saved → {p.name}</div>',
        _session_dropdown_options(STATE.session.id),
    )


def h_delete_session(session_id: str):
    if not session_id:
        return f'<div class="cc-stat err">no session selected</div>', gr.update()
    STATE.store.delete(session_id)
    return (
        f'<div class="cc-stat info">deleted {session_id}</div>',
        _session_dropdown_options(STATE.session.id if STATE.session.id != session_id else None),
    )


def h_refresh_sessions():
    return _session_dropdown_options(STATE.session.id)


def h_list_files_browser(path: str):
    """File browser handler — lists entries under `path`."""
    base = STATE.config.working_dir
    target = (base / path).resolve() if path else base
    try:
        target.relative_to(base)
    except ValueError:
        target = base
    if not target.is_dir():
        target = base
    rows = []
    try:
        children = sorted(
            target.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except OSError:
        children = []
    rel = target.relative_to(base)
    rows.append(
        f'<div style="font-size:12px;color:{MUTED};margin-bottom:6px;">'
        f'📁 /{rel}</div>'
    )
    if target != base:
        rows.append('<div style="font-size:12px;"><a href="#" '
                    'onclick="document.querySelector(\'#fb-path input\').value=\'..\';'
                    'document.querySelector(\'#fb-list-btn\').click()">↩ ..</a></div>')
    for c in children[:200]:
        suffix = "/" if c.is_dir() else ""
        crel = c.relative_to(base)
        rows.append(
            f'<div style="font-size:12px;">{"📁" if c.is_dir() else "📄"} '
            f'{crel}{suffix}</div>'
        )
    if len(children) > 200:
        rows.append(f'<div style="font-size:11px;color:{MUTED};">… and {len(children)-200} more</div>')
    return "\n".join(rows)


def h_apply_settings(
    max_iter: int,
    system_override: str,
    native_tools: bool,
    custom_bash_timeout: int,
    max_tokens: int,
):
    STATE.config.max_iterations = int(max_iter)
    STATE.config.provider.native_tools = bool(native_tools)
    STATE.config.provider.max_tokens = int(max_tokens)
    STATE.config.bash_timeout = int(custom_bash_timeout)
    override = system_override.strip()
    STATE.config.system_prompt_override = override if override else None
    bits = [
        f"max_iterations={STATE.config.max_iterations}",
        f"native_tools={STATE.config.provider.native_tools}",
        f"max_tokens={STATE.config.provider.max_tokens}",
        f"bash_timeout={STATE.config.bash_timeout}s",
    ]
    if override:
        bits.append("custom system prompt: ON")
    return f'<div class="cc-stat info">{" · ".join(bits)}</div>'


def _session_dropdown_options(current_id: str | None) -> gr.update:
    metas = STATE.store.list()
    if not metas:
        choices = []
    else:
        choices = [
            (f"{m.title}  ({m.message_count} msgs, {time.strftime('%Y-%m-%d %H:%M', time.localtime(m.updated_at))})", m.id)
            for m in metas
        ]
    value = current_id if any(c[1] == current_id for c in choices) else None
    return gr.update(choices=choices, value=value)


# ---------------------------------------------------------------------- #
# THE BIG ONE: chat submit (generator) handler
# ---------------------------------------------------------------------- #


def h_chat_submit(user_text: str, chat_history):
    """Generator: streams events from the agent into the chat UI.

    Yields (chatbot, tool_log, todos, status_banner) tuples as events arrive.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        yield chat_history, STATE.tool_log_html, STATE.todo_html, (
            '<div class="cc-stat err">empty message</div>'
        )
        return

    # Snapshot the chat history BEFORE this turn, so we can rebuild it cleanly
    # from events during streaming.
    base_history: list[dict] = list(STATE.chat_history)

    # Rebuild the agent (picks up latest config).
    agent = STATE.rebuild_agent()
    handle = RunHandle()
    STATE.stop_requested = False

    th = threading.Thread(target=_worker, args=(agent, user_text, handle), daemon=True)
    th.start()

    # Drain the event queue, yielding UI updates as we go.
    turn_events: list[AgentEvent] = []
    last_yield = 0.0
    status = '<div class="cc-stat info">working…</div>'
    while True:
        try:
            e = handle.events_q.get(timeout=0.2)
        except queue.Empty:
            e = None
        if e is not None:
            turn_events.append(e)
            STATE.events.append(e)
        # Throttle UI updates to ~10/s to avoid hammering Gradio.
        now = time.time()
        if e is None and now - last_yield < 0.1 and not handle.done.is_set():
            continue
        last_yield = now

        chat = _events_to_chat(turn_events, base_history)
        tool_html = _render_tool_log(turn_events)
        todo_html = _render_todos(turn_events)
        if handle.done.is_set():
            status = '<div class="cc-stat ok">done</div>'
            yield chat, tool_html, todo_html, status
            break
        yield chat, tool_html, todo_html, status

    th.join(timeout=5.0)
    # Persist final state into STATE for the next turn.
    STATE.chat_history = _events_to_chat(turn_events, base_history)
    STATE.tool_log_html = _render_tool_log(turn_events)
    STATE.todo_html = _render_todos(turn_events)
    # Autosave if enabled.
    if STATE.config.autosave_sessions:
        try:
            STATE.store.save(STATE.session)
        except OSError:
            pass


def h_stop():
    """Best-effort stop. We can't kill the LLM subprocess mid-call, but we
    flag stop_requested so the next iteration check can short-circuit.
    For v1 we just notify the user."""
    STATE.stop_requested = True
    return '<div class="cc-stat err">stop requested (will take effect after the current tool call)</div>'


def h_clear_chat():
    STATE.chat_history = []
    STATE.events = []
    STATE.tool_log_html = ""
    STATE.todo_html = ""
    STATE.session.messages.clear()
    return [], "", "", '<div class="cc-stat info">chat cleared</div>'


def h_load_settings_into_ui():
    """Populate the settings tab with current values."""
    return (
        STATE.config.max_iterations,
        STATE.config.system_prompt_override or "",
        STATE.config.provider.native_tools,
        STATE.config.bash_timeout,
        STATE.config.provider.max_tokens,
    )


# ====================================================================== #
# Build the UI
# ====================================================================== #


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Claude Code (Python edition)") as ui:
        # ---- top bar ----
        with gr.Row(elem_id="cc-topbar"):
            gr.Markdown(
                "# Claude Code (Python edition)\n"
                '<span class="sub">Agentic coding assistant · Gradio web UI</span>'
            )
            gr.HTML(
                f'<span class="badge">provider: {STATE.config.provider.provider}</span>'
                f'<span class="badge">v0.2.0</span>'
            )

        with gr.Row(elem_id="cc-body"):
            # ============ LEFT SIDEBAR ============
            with gr.Column(scale=1, min_width=240):
                with gr.Group(elem_classes="cc-panel"):
                    gr.Markdown("### Sessions")
                    session_dd = gr.Dropdown(
                        label="Load session",
                        choices=[],
                        value=None,
                        interactive=True,
                        scale=4,
                    )
                    with gr.Row():
                        load_btn = gr.Button("Load", size="sm", scale=1)
                        refresh_btn = gr.Button("⟳", size="sm", scale=1)
                    with gr.Row():
                        new_btn = gr.Button("New", size="sm", scale=1)
                        save_btn = gr.Button("Save", size="sm", scale=1)
                        del_btn = gr.Button("Delete", size="sm", variant="stop", scale=1)

                with gr.Group(elem_classes="cc-panel"):
                    gr.Markdown("### Project")
                    work_dir = gr.Textbox(
                        label="Working directory",
                        value=str(STATE.config.working_dir),
                        interactive=True,
                    )
                    work_dir_btn = gr.Button("Apply", size="sm")
                    mode_dd = gr.Dropdown(
                        label="Mode",
                        choices=[(lbl, val) for lbl, val in MODES],
                        value=STATE.config.mode,
                    )
                    perm_radio = gr.Radio(
                        label="Permission preset",
                        choices=[(lbl, val) for lbl, val in PERM_PRESETS],
                        value=STATE.perm_mode,
                    )

                with gr.Group(elem_classes="cc-panel"):
                    gr.Markdown("### LLM provider")
                    provider_dd = gr.Dropdown(
                        label="Provider",
                        choices=[(p["label"], p["id"]) for p in KNOWN_PROVIDERS],
                        value=STATE.config.provider.provider,
                        interactive=True,
                    )
                    api_key_tb = gr.Textbox(
                        label="API key",
                        type="password",
                        placeholder="Paste key (or set env var)",
                        value="",
                        interactive=True,
                    )
                    model_tb = gr.Textbox(
                        label="Model",
                        value=STATE.config.provider.model,
                        interactive=True,
                    )
                    base_url_tb = gr.Textbox(
                        label="Base URL (custom only)",
                        value=STATE.config.provider.base_url,
                        interactive=(STATE.config.provider.provider == "custom"),
                    )
                    provider_apply_btn = gr.Button("Apply provider", size="sm", variant="primary")
                    provider_hint = gr.Markdown(
                        f"API key (env var: <code>{env_key_for('anthropic')}</code>)",
                        elem_classes="cc-hint",
                    )

                with gr.Group(elem_classes="cc-panel"):
                    gr.Markdown("### Status")
                    status_md = gr.HTML(
                        f'<div class="cc-stat info">ready</div>'
                    )

            # ============ MAIN CHAT ============
            with gr.Column(scale=3, min_width=520):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    elem_id="cc-chatbot",
                    height=520,
                    buttons=["copy", "copy_all"],
                    placeholder="Ask me to read, edit, or run something in your project…",
                )
                with gr.Row():
                    msg_in = gr.Textbox(
                        label="Message",
                        placeholder="Ask Claude to do something (Shift+Enter for newline)…",
                        lines=2,
                        scale=8,
                        autofocus=True,
                    )
                    with gr.Column(scale=1, min_width=120):
                        send_btn = gr.Button("Send", variant="primary", size="lg")
                        with gr.Row():
                            stop_btn = gr.Button("Stop", size="sm")
                            clear_btn = gr.Button("Clear", size="sm")
                gr.HTML(
                    '<div style="font-size:11px;color:#999;margin-top:4px;">'
                    'Tip: toggle the permission preset in the sidebar before sending. '
                    'Use <code>Plan</code> mode for read-only investigation.'
                    '</div>'
                )

            # ============ RIGHT PANEL ============
            with gr.Column(scale=2, min_width=280):
                with gr.Tabs():
                    with gr.Tab("Tool calls"):
                        tool_log = gr.HTML(
                            value='<div style="color:#999;font-size:12px;padding:8px;">No tool calls yet.</div>',
                        )
                    with gr.Tab("Todos"):
                        todos_html = gr.HTML(
                            value='<div style="color:#999;font-size:12px;padding:8px;">No todos set.</div>',
                        )
                    with gr.Tab("Files"):
                        fb_path = gr.Textbox(
                            label="Path (relative)",
                            value="",
                            elem_id="fb-path",
                        )
                        fb_list_btn = gr.Button("List", size="sm", elem_id="fb-list-btn")
                        fb_out = gr.HTML(
                            value='<div style="color:#999;font-size:12px;padding:8px;">Click List to browse.</div>',
                        )
                    with gr.Tab("Settings"):
                        s_max_iter = gr.Slider(
                            label="Max iterations per turn",
                            minimum=1, maximum=100, step=1,
                            value=STATE.config.max_iterations,
                        )
                        s_native_tools = gr.Checkbox(
                            label="Use native tool-calling (when provider supports it)",
                            value=STATE.config.provider.native_tools,
                        )
                        s_max_tokens = gr.Slider(
                            label="Max tokens per response",
                            minimum=256, maximum=32768, step=256,
                            value=STATE.config.provider.max_tokens,
                        )
                        s_bash_timeout = gr.Slider(
                            label="Bash timeout (seconds)",
                            minimum=5, maximum=600, step=5,
                            value=STATE.config.bash_timeout,
                        )
                        s_sys_override = gr.Textbox(
                            label="Custom system prompt override (optional)",
                            value=STATE.config.system_prompt_override or "",
                            lines=4,
                            placeholder="Leave empty to use the mode's default role prompt.",
                        )
                        s_apply_btn = gr.Button("Apply settings", size="sm")
                        s_status = gr.HTML(value="")
                    with gr.Tab("Tools ref"):
                        gr.Markdown(
                            "### Available tools\n\n"
                            + _tools_md()
                        )

        # ---- wiring ----
        work_dir_btn.click(
            h_connect_working_dir,
            inputs=[work_dir],
            outputs=[work_dir, status_md],
        )
        mode_dd.change(h_set_mode, inputs=[mode_dd], outputs=[status_md])
        perm_radio.change(h_set_perm_preset, inputs=[perm_radio], outputs=[status_md])

        # Provider wiring.
        provider_dd.change(
            h_provider_defaults,
            inputs=[provider_dd],
            outputs=[model_tb, base_url_tb, provider_hint],
        )
        provider_apply_btn.click(
            h_set_provider,
            inputs=[provider_dd, api_key_tb, model_tb, base_url_tb],
            outputs=[status_md],
        )

        new_btn.click(
            h_new_session,
            outputs=[chatbot, tool_log, todos_html, status_md, session_dd],
        )
        load_btn.click(
            h_load_session,
            inputs=[session_dd],
            outputs=[chatbot, tool_log, todos_html, status_md, session_dd],
        )
        save_btn.click(
            h_save_session,
            outputs=[status_md, session_dd],
        )
        del_btn.click(
            h_delete_session,
            inputs=[session_dd],
            outputs=[status_md, session_dd],
        )
        refresh_btn.click(h_refresh_sessions, outputs=[session_dd])

        fb_list_btn.click(h_list_files_browser, inputs=[fb_path], outputs=[fb_out])

        s_apply_btn.click(
            h_apply_settings,
            inputs=[s_max_iter, s_sys_override, s_native_tools, s_bash_timeout, s_max_tokens],
            outputs=[s_status],
        )

        # Initial settings load on page load.
        ui.load(
            h_load_settings_into_ui,
            outputs=[s_max_iter, s_sys_override, s_native_tools, s_bash_timeout, s_max_tokens],
        )
        ui.load(h_refresh_sessions, outputs=[session_dd])

        # Chat submit — generator function for streaming.
        send_outputs = [chatbot, tool_log, todos_html, status_md]
        send_btn.click(
            h_chat_submit,
            inputs=[msg_in, chatbot],
            outputs=send_outputs,
        ).then(lambda: gr.update(value=""), outputs=[msg_in])
        msg_in.submit(
            h_chat_submit,
            inputs=[msg_in, chatbot],
            outputs=send_outputs,
        ).then(lambda: gr.update(value=""), outputs=[msg_in])

        stop_btn.click(h_stop, outputs=[status_md])
        clear_btn.click(
            h_clear_chat,
            outputs=[chatbot, tool_log, todos_html, status_md],
        )

    return ui


def _tools_md() -> str:
    """Render the tools reference as markdown for the Tools ref tab."""
    reg = default_registry()
    lines = ["| Tool | Description |", "|------|-------------|"]
    for name in reg.names():
        t = reg.get(name)
        desc = (t.description or "").split("\n")[0]
        lines.append(f"| `{name}` | {desc} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--working-dir", default=None,
                   help="Override initial working directory.")
    args = p.parse_args()
    if args.working_dir:
        p_ = Path(args.working_dir).expanduser().resolve()
        STATE.config.working_dir = p_
        STATE.session.working_dir = str(p_)

    ui = build_ui()
    ui.queue(default_concurrency_limit=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        css=CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
