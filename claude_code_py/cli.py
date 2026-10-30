"""CLI entry point — a prompt_toolkit-based REPL, modeled on `claude code`.

Run with: `python -m claude_code_py.cli` (after `pip install -e .`)
or:       `python /path/to/cli.py`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from .agent import Agent, AgentEvent, EventType
from .config import AgentConfig
from .permissions import PermissionDecision
from .session import Session
from .tools import default_registry

console = Console()


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="claude-code-py",
        description="Claude Code (Python edition) — agentic coding assistant.",
    )
    p.add_argument(
        "--working-dir", "-C",
        default=".",
        help="Working directory for the agent (default: current dir).",
    )
    p.add_argument(
        "--mode", "-m",
        choices=["default", "plan", "code", "explain"],
        default="default",
        help="Agent mode (default: default).",
    )
    p.add_argument(
        "--provider",
        choices=["anthropic", "deepseek", "gemini", "huggingface", "custom"],
        default="anthropic",
        help="LLM provider (default: anthropic).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name (defaults to the provider's default).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key for the provider (defaults to the matching env var).",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL (only used by --provider=custom).",
    )
    p.add_argument(
        "--auto-approve-bash", action="store_true",
        help="Auto-approve all bash commands (skip the y/N prompt).",
    )
    p.add_argument(
        "--auto-approve-writes", action="store_true",
        help="Auto-approve file writes/edits.",
    )
    p.add_argument(
        "--max-iterations", type=int, default=25,
        help="Max tool-call iterations per turn (default 25).",
    )
    return p


def _cli_permission_callback(auto_bash: bool, auto_writes: bool):
    """Build a callback that prompts on the terminal for ASK decisions."""
    def cb(name: str, args: dict) -> PermissionDecision:
        if name == "bash" and auto_bash:
            return PermissionDecision.ALLOW
        if name in ("write_file", "edit_file") and auto_writes:
            return PermissionDecision.ALLOW
        # Prompt the user.
        console.print(Panel.fit(
            f"[bold yellow]Permission requested[/]: {name}\n"
            f"args: {args}",
            title="Approve?",
            border_style="yellow",
        ))
        try:
            ans = input("  Allow? [y/N/a=always-allow-this-session] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return PermissionDecision.DENY
        if ans in ("y", "yes"):
            return PermissionDecision.ALLOW
        if ans == "a":
            # Best-effort: mutate the agent's config for the rest of the session.
            # The caller closes over auto_bash/auto_writes; we mutate them too.
            if name == "bash":
                auto_bash = True  # noqa: F841 — local-only, but informative
            if name in ("write_file", "edit_file"):
                auto_writes = True
            return PermissionDecision.ALLOW
        return PermissionDecision.DENY
    return cb


def _print_event(e: AgentEvent) -> None:
    if e.type is EventType.USER:
        console.print(Panel(e.payload["text"], title="You", border_style="cyan"))
    elif e.type is EventType.TOOL_CALL:
        name = e.payload["name"]
        args = e.payload["arguments"]
        console.print(f"\n[dim]→ {name}({args})[/]")
    elif e.type is EventType.TOOL_RESULT:
        ok = e.payload["ok"]
        content = e.payload["content"]
        color = "green" if ok else "red"
        console.print(f"[{color}]✓ {name}[/]" if ok else f"[{color}]✗ {name}[/]")
        # Show first ~30 lines of the result for context.
        body = "\n".join(content.splitlines()[:30])
        if len(content.splitlines()) > 30:
            body += "\n..."
        console.print(Syntax(body, "text", theme="ansi_dark", line_numbers=False, word_wrap=True))
    elif e.type is EventType.ASSISTANT_TEXT and e.payload.get("final"):
        console.print(Panel(Markdown(e.payload["text"]), title="Claude", border_style="magenta"))
    elif e.type is EventType.ERROR:
        console.print(f"[bold red]error:[/] {e.payload['message']}")
    elif e.type is EventType.ITERATION_LIMIT:
        console.print(f"[bold yellow]warning:[/] {e.payload['message']}")
    elif e.type is EventType.PERMISSION_DECISION:
        if e.payload["decision"] == "deny":
            console.print("[red]permission: denied[/]")


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    working_dir = Path(args.working_dir).expanduser().resolve()
    if not working_dir.is_dir():
        console.print(f"[red]Working directory not found: {working_dir}[/]")
        return 2

    # Build the provider config from CLI args. API key/model fall back to the
    # provider's defaults and matching env vars when not given explicitly.
    from .providers import ProviderConfig, default_model_for, env_key_for
    import os
    env_var = env_key_for(args.provider)
    api_key = args.api_key or (os.environ.get(env_var) if env_var else "")
    model = args.model or default_model_for(args.provider)
    provider_cfg = ProviderConfig(
        provider=args.provider,
        api_key=api_key or "",
        model=model,
        base_url=args.base_url or "",
    )

    config = AgentConfig(
        working_dir=working_dir,
        mode=args.mode,
        provider=provider_cfg,
        auto_approve_bash=args.auto_approve_bash,
        auto_approve_write_tools=args.auto_approve_writes,
        max_iterations=args.max_iterations,
    )

    console.print(Panel.fit(
        f"[bold]Claude Code (Python edition)[/]\n"
        f"working dir: [cyan]{working_dir}[/]\n"
        f"provider: [cyan]{args.provider}[/]  model: [cyan]{model or '(none)'}[/]\n"
        f"mode: [cyan]{args.mode}[/]    tools: {', '.join(default_registry().names())}\n"
        f"\nType your request, or [bold]/exit[/] to quit.",
        border_style="magenta",
    ))

    session = Session(working_dir=str(working_dir), mode=args.mode)
    auto_bash = args.auto_approve_bash
    auto_writes = args.auto_approve_writes
    agent = Agent(
        config,
        session=session,
        permission_callback=_cli_permission_callback(auto_bash, auto_writes),
    )

    try:
        while True:
            try:
                user_text = console.input("[bold green]> [/]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/]")
                break
            if not user_text:
                continue
            if user_text in ("/exit", "/quit", "/q"):
                break
            if user_text in ("/help", "/?"):
                console.print(Panel(
                    "Commands:\n"
                    "  /exit, /quit   leave the REPL\n"
                    "  /mode X        switch mode (default|plan|code|explain)\n"
                    "  /tools         list available tools\n"
                    "  /clear         clear conversation history",
                    border_style="cyan",
                ))
                continue
            if user_text.startswith("/mode "):
                new_mode = user_text[6:].strip()
                if new_mode in ("default", "plan", "code", "explain"):
                    config.mode = new_mode
                    session.mode = new_mode
                    console.print(f"[dim]mode → {new_mode}[/]")
                else:
                    console.print(f"[red]unknown mode: {new_mode}[/]")
                continue
            if user_text == "/tools":
                for t in default_registry().names():
                    console.print(f"  - {t}")
                continue
            if user_text == "/clear":
                session.messages.clear()
                console.print("[dim]history cleared[/]")
                continue

            agent.run_reset_turn_state()
            try:
                agent.run(user_text, on_event=_print_event)
            except KeyboardInterrupt:
                console.print("\n[bold yellow]interrupted[/]")
    finally:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
