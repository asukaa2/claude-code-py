# Claude Code (Python edition) + Gradio Web UI

A Python reimplementation of Anthropic's [Claude Code](https://github.com/anthropics/claude-code)
agentic coding assistant, with a complete **Gradio web UI** for easier control
than the CLI.

The original Claude Code is a TypeScript/Node.js tool distributed as a compiled
npm bundle. This project captures its core essence in pure Python:

- **Agentic loop** — multi-turn chat with an LLM that can call tools to read,
  write, and edit files, run bash, search code, and track its own todos.
- **Tool registry** — pluggable tools with JSON schemas, dispatched by the
  agent loop based on ` ```tool_call` blocks the model emits.
- **Permission policy** — allow / ask / deny classification for each tool call,
  with three quick presets in the UI (Yolo / Trust reads / Manual).
- **Session persistence** — every conversation is JSON-serialized to disk;
  load, save, delete from either the UI or the API.
- **CLI** — a `prompt_toolkit` + `rich` REPL with the same `/help`, `/mode`,
  `/tools`, `/clear`, `/exit` commands as the original.
- **Gradio web UI** — chat, working-dir picker, mode selector, permission
  preset, session manager, live tool-call log, todos panel, file browser,
  and a settings tab — all in one page.

## Project layout

```
.
├── claude-code/             # the cloned reference repo (read-only)
├── claude_code_py/          # ← the Python port
│   ├── __init__.py
│   ├── config.py            # AgentConfig (model, perms, paths, limits)
│   ├── llm.py               # LLMClient wrapping the `z-ai chat` CLI
│   ├── prompts.py           # system prompt + tool-call protocol
│   ├── permissions.py       # PermissionPolicy + PermissionDecision
│   ├── session.py           # Session + SessionStore (JSON persistence)
│   ├── agent.py             # the agentic loop + AgentEvent stream
│   ├── cli.py               # prompt_toolkit REPL entry point
│   └── tools/
│       ├── base.py          # Tool ABC + Registry + ToolContext + ToolResult
│       ├── bash.py          # run shell commands
│       ├── read_file.py     # read file (with line ranges, cat -n output)
│       ├── write_file.py    # create / overwrite
│       ├── edit_file.py     # unique-string replacement (Claude-Code-style)
│       ├── list_files.py    # dir listing (recursive option)
│       ├── grep.py          # ripgrep content search
│       ├── glob_tool.py     # glob pattern match
│       └── todo.py          # todo_write / todo_read
├── webui.py                 # ← the Gradio web UI entry point
├── scripts/
│   ├── launch_webui.py      # detached launcher (survives shell exit)
│   ├── smoke_test_agent.py  # headless agent test
│   └── ui_e2e.py            # drives the running webui via gradio_client
└── README.md
```

## Quick start

### 1. Run the web UI (recommended)

```bash
cd /home/z/my-project
python webui.py --port 7860
# then open http://localhost:7860
```

For a public Gradio share link:

```bash
python webui.py --share
```

The UI layout:

```
┌──────────────┬─────────────────────────────┬──────────────────┐
│  Sidebar     │  Chat (main)                │  Right panel     │
│  - sessions  │  - message list             │  - tool calls    │
│  - work dir  │  - input box                │  - todos         │
│  - mode      │  - send/stop/clear          │  - file browser  │
│  - perms     │                             │  - settings      │
└──────────────┴─────────────────────────────┴──────────────────┘
```

**Permission presets** (sidebar radio):
- **Yolo (auto-all)** — auto-approve reads, writes, AND bash. Dangerous bash
  patterns (`rm -rf /`, `sudo`, `mkfs`, `dd of=/dev/`, fork bombs, …) are
  still hard-denied.
- **Trust reads** (default) — auto-approve read-only tools; ask before writes
  / bash.
- **Manual** — ask before every tool call that touches state.

Because mid-run modal approval is awkward in Gradio, "ask" decisions are
currently denied with a notice in the chat. Flip the preset and re-send.

**Modes** (sidebar dropdown):
- **Default** — balanced agentic behavior.
- **Plan** — read-only investigation; produces a plan, waits for approval.
- **Code** — action-oriented; prefer editing files over describing changes.
- **Explain** — read-only; produce a clear explanation.

### 2. Run the CLI

```bash
PYTHONPATH=/home/z/my-project python -m claude_code_py.cli
# or:
PYTHONPATH=/home/z/my-project python -m claude_code_py.cli \
    --working-dir /path/to/project \
    --mode code \
    --auto-approve-reads \
    --max-iterations 25
```

REPL commands: `/help`, `/mode <default|plan|code|explain>`, `/tools`,
`/clear`, `/exit`.

### 3. Use the Python API directly

```python
from claude_code_py import Agent, AgentConfig, EventType
from claude_code_py.permissions import PermissionDecision

cfg = AgentConfig(
    working_dir="/path/to/project",
    auto_approve_read_tools=True,
    auto_approve_write_tools=True,
    auto_approve_bash=False,
)

def on_event(e):
    if e.type is EventType.TOOL_CALL:
        print(f"  → {e.payload['name']}({e.payload['arguments']})")
    elif e.type is EventType.TOOL_RESULT:
        print(f"  {'✓' if e.payload['ok'] else '✗'} {e.payload['name']}")
    elif e.type is EventType.ASSISTANT_TEXT and e.payload['final']:
        print(e.payload['text'])

agent = Agent(cfg, permission_callback=lambda n, a: PermissionDecision.ALLOW)
agent.run("Find all TODO comments in src/ and list them.", on_event=on_event)
```

## How tool calling works

The LLM backend is the bundled `z-ai chat` CLI (returns OpenAI-compatible
JSON, model `glm-4-plus`). It does NOT expose native function-calling, so we
implement tool-calling via a **structured-output protocol**:

1. The system prompt instructs the model to emit tool calls as fenced
   ` ```tool_call` blocks containing a JSON object with `name` and `arguments`.
2. The agent loop parses those blocks (with a lenient JSON parser that
   tolerates trailing commas and `//` comments), dispatches them through the
   `Registry`, and appends a `tool`-role message with the real result.
3. The loop continues until the model emits a response with no tool calls
   (the final answer), or the iteration cap is hit.

The parser also accepts a fallback inline `[TOOL_CALL ...] ... [/TOOL_CALL]`
format, because some models drift to that shape after seeing transcript text.
The transcript sent back to the model uses the SAME fenced ` ```tool_call`
format the model is asked to emit, which keeps the protocol self-consistent
and minimizes drift.

## Available tools

| Tool | Description |
|------|-------------|
| `bash` | Run a bash command (combined stdout+stderr, exit code, 10KB cap). |
| `read_file` | Read a file (optional 1-indexed line range, cat -n output). |
| `write_file` | Create or overwrite a file (makes parent dirs). |
| `edit_file` | Replace a unique `old_string` with `new_string` (Claude-Code-style). |
| `list_files` | List a directory (recursive option, 1000-entry cap). |
| `grep` | Ripgrep content search (regex, glob filter, 200-match cap). |
| `glob` | Find files by name pattern (500-match cap). |
| `todo_write` | Replace the agent's todo list for the current task. |
| `todo_read` | Read the current todo list. |

All file tools refuse paths outside the working directory.

## Testing

```bash
# Headless agent smoke test (no UI):
PYTHONPATH=/home/z/my-project python scripts/smoke_test_agent.py

# Launch the webui detached, then drive it end-to-end via gradio_client:
python scripts/launch_webui.py
python scripts/ui_e2e.py
```

## Configuration

All knobs live in `claude_code_py.config.AgentConfig`. The most useful:

| Field | Default | Notes |
|-------|---------|-------|
| `model` | `"glm-4-plus"` | Set by the `z-ai` CLI; informational. |
| `llm_cli` | `"z-ai"` | CLI binary used to reach the LLM. |
| `max_iterations` | `25` | Hard cap on tool-call round-trips per turn. |
| `mode` | `"default"` | One of `default` / `plan` / `code` / `explain`. |
| `working_dir` | `cwd()` | Where file tools are rooted. |
| `auto_approve_read_tools` | `True` | read_file / list_files / grep / glob / todo_read |
| `auto_approve_write_tools` | `False` | write_file / edit_file |
| `auto_approve_bash` | `False` | bash |
| `bash_deny_patterns` | (see source) | Always-denied regexes (rm -rf /, sudo, …) |
| `bash_timeout` | `120` | Per-command timeout in seconds. |
| `sessions_dir` | `~/.claude_code_py/sessions` | Where session JSONs live. |

## Dependencies

- Python 3.10+
- `gradio` (≥ 6) — for the web UI
- `rich` + `prompt_toolkit` — for the CLI
- `ripgrep` (`rg`) — for the grep tool
- the bundled `z-ai` CLI — for the LLM backend

See `requirements.txt`.

## License

This Python port is for educational/demo purposes. The original Claude Code
is © Anthropic under their license (see `claude-code/LICENSE.md`).
