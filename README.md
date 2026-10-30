# Claude Code (Python edition) + Gradio Web UI

A Python reimplementation of Anthropic's [Claude Code](https://github.com/anthropics/claude-code)
agentic coding assistant, with a complete **Gradio web UI** for easier control
than the CLI.

The original Claude Code is a TypeScript/Node.js tool distributed as a compiled
npm bundle. This project captures its core essence in pure Python:

- **Agentic loop** — multi-turn chat with an LLM that can call tools to read,
  write, and edit files, run bash, search code, and track its own todos.
- **Multiple LLM providers** — Anthropic (default), DeepSeek, Google Gemini,
  HuggingFace, or any custom OpenAI-compatible endpoint. Swap providers at
  runtime from the UI or CLI.
- **Tool registry** — pluggable tools with JSON schemas, dispatched by the
  agent loop based on native function-calling (when supported) or a
  structured-output fallback (` ```tool_call` blocks the model emits).
- **Permission policy** — allow / ask / deny classification for each tool call,
  with three quick presets in the UI (Yolo / Trust reads / Manual).
- **Session persistence** — every conversation is JSON-serialized to disk;
  load, save, delete from either the UI or the API.
- **CLI** — a `prompt_toolkit` + `rich` REPL with the same `/help`, `/mode`,
  `/tools`, `/clear`, `/exit` commands as the original.
- **Gradio web UI** — chat, working-dir picker, mode selector, permission
  preset, **provider picker + API key + model fields**, session manager, live
  tool-call log, todos panel, file browser, and a settings tab — all in one
  page.

## Project layout

```
.
├── claude_code_py/          # ← the Python port
│   ├── __init__.py
│   ├── config.py            # AgentConfig (provider, perms, paths, limits)
│   ├── messages.py          # LLMMessage + LLMResponse (shared types)
│   ├── llm.py               # LLMClient facade over the provider abstraction
│   ├── prompts.py           # system prompt + tool-call protocol
│   ├── permissions.py       # PermissionPolicy + PermissionDecision
│   ├── session.py           # Session + SessionStore (JSON persistence)
│   ├── agent.py             # the agentic loop + AgentEvent stream
│   ├── cli.py               # prompt_toolkit REPL entry point
│   ├── providers/
│   │   ├── __init__.py              # ProviderConfig + KNOWN_PROVIDERS + factory
│   │   ├── anthropic_provider.py    # native SDK + native tool-calling (default)
│   │   └── openai_compat_provider.py# DeepSeek, Gemini, HuggingFace, Custom
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

### 1. Set your API key

Pick the provider you want and export its key (or paste it in the UI later):

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY="sk-ant-…"

# DeepSeek
export DEEPSEEK_API_KEY="sk-…"

# Google Gemini
export GEMINI_API_KEY="AIza…"

# HuggingFace
export HF_TOKEN="hf_…"
```

### 2. Run the web UI (recommended)

```bash
cd /home/z/my-project
python webui.py --port 7860
# then open http://localhost:7860
```

In the sidebar:
1. Pick a **Provider** (Anthropic is the default).
2. The Model and Base URL fields auto-populate with sensible defaults; the API
   key field shows which env var it reads from. Paste a key or rely on the env
   var.
3. Click **Apply provider**.
4. Pick a working directory, set the permission preset, and start chatting.

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
│  - PROVIDER  │                             │                  │
│  - api key   │                             │                  │
│  - model     │                             │                  │
└──────────────┴─────────────────────────────┴──────────────────┘
```

**Permission presets** (sidebar radio):
- **Yolo (auto-all)** — auto-approve reads, writes, AND bash. Dangerous bash
  patterns (`rm -rf /`, `sudo`, `mkfs`, `dd of=/dev/`, fork bombs, …) are
  still hard-denied.
- **Trust reads** (default) — auto-approve read-only tools; ask before writes
  / bash.
- **Manual** — ask before every tool call that touches state.

**Modes** (sidebar dropdown):
- **Default** — balanced agentic behavior.
- **Plan** — read-only investigation; produces a plan, waits for approval.
- **Code** — action-oriented; prefer editing files over describing changes.
- **Explain** — read-only; produce a clear explanation.

### 3. Run the CLI

```bash
PYTHONPATH=/home/z/my-project python -m claude_code_py.cli \
    --provider anthropic \
    --model claude-3-5-sonnet-latest \
    --working-dir /path/to/project \
    --mode code

# Or with a different provider:
PYTHONPATH=/home/z/my-project python -m claude_code_py.cli \
    --provider deepseek \
    --model deepseek-chat

PYTHONPATH=/home/z/my-project python -m claude_code_py.cli \
    --provider gemini \
    --model gemini-1.5-flash

PYTHONPATH=/home/z/my-project python -m claude_code_py.cli \
    --provider custom \
    --base-url https://your-endpoint.com/v1 \
    --model your-model \
    --api-key sk-…
```

REPL commands: `/help`, `/mode <default|plan|code|explain>`, `/tools`,
`/clear`, `/exit`.

### 4. Use the Python API directly

```python
from claude_code_py import (
    Agent, AgentConfig, ProviderConfig, EventType,
)
from claude_code_py.permissions import PermissionDecision

cfg = AgentConfig(
    working_dir="/path/to/project",
    provider=ProviderConfig(
        provider="anthropic",                 # or deepseek/gemini/huggingface/custom
        model="claude-3-5-sonnet-latest",
        api_key="sk-ant-…",                   # or rely on ANTHROPIC_API_KEY
    ),
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

## Supported providers

| Provider | Default model | Env var | Notes |
|----------|---------------|---------|-------|
| **Anthropic** (default) | `claude-3-5-sonnet-latest` | `ANTHROPIC_API_KEY` | Native SDK + native tool-calling |
| **DeepSeek** | `deepseek-chat` | `DEEPSEEK_API_KEY` | OpenAI-compatible API |
| **Google Gemini** | `gemini-1.5-flash` | `GEMINI_API_KEY` | OpenAI-compatible endpoint |
| **HuggingFace** | `meta-llama/Llama-3.1-8B-Instruct` | `HF_TOKEN` | Inference endpoints (OpenAI-compat) |
| **Custom** | (none) | (none) | Any OpenAI-compatible base_url + key |

For providers that support native tool-calling (Anthropic, OpenAI-compatible
backends), the agent passes the tools schema via the SDK's `tools=` parameter.
For backends that don't, the agent falls back to the structured-output
protocol: the model emits ` ```tool_call` blocks in prose, which the agent
parses.

## How tool calling works

1. The agent loop builds a JSON-schema list of all registered tools and passes
   it to the provider via `tools=` (for providers that support it).
2. If the provider returns native tool_calls, they are re-serialized into the
   ` ```tool_call` fenced format so the agent's single dispatch path works
   across every provider.
3. If the provider doesn't support tools, the system prompt instructs the model
   to emit tool calls as fenced ` ```tool_call` blocks containing a JSON object
   with `name` and `arguments`.
4. The agent loop parses those blocks (with a lenient JSON parser that
   tolerates trailing commas and `//` comments), dispatches them through the
   `Registry`, and appends a `tool`-role message with the real result.
5. The loop continues until the model emits a response with no tool calls
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
| `provider` | `ProviderConfig(provider="anthropic")` | LLM provider + key + model + base_url |
| `max_iterations` | `25` | Hard cap on tool-call round-trips per turn. |
| `mode` | `"default"` | One of `default` / `plan` / `code` / `explain`. |
| `working_dir` | `cwd()` | Where file tools are rooted. |
| `auto_approve_read_tools` | `True` | read_file / list_files / grep / glob / todo_read |
| `auto_approve_write_tools` | `False` | write_file / edit_file |
| `auto_approve_bash` | `False` | bash |
| `bash_deny_patterns` | (see source) | Always-denied regexes (rm -rf /, sudo, …) |
| `bash_timeout` | `120` | Per-command timeout in seconds. |
| `sessions_dir` | `~/.claude_code_py/sessions` | Where session JSONs live. |

`ProviderConfig` fields:

| Field | Default | Notes |
|-------|---------|-------|
| `provider` | `"anthropic"` | One of: anthropic, deepseek, gemini, huggingface, custom |
| `model` | (provider default) | Model name to send to the API. |
| `api_key` | `""` | Falls back to the provider's env var if empty. |
| `base_url` | (provider default) | Only editable for the `custom` provider. |
| `native_tools` | `True` | Pass `tools=` to the API when supported. |
| `max_tokens` | `4096` | Max tokens for the assistant response. |
| `temperature` | `0.7` | Sampling temperature. |

## Dependencies

- Python 3.10+
- `gradio` (≥ 6) — for the web UI
- `rich` + `prompt_toolkit` — for the CLI
- `anthropic` — for the Anthropic provider (default)
- `openai` — for the DeepSeek / Gemini / HuggingFace / Custom providers
- `ripgrep` (`rg`) — for the grep tool
- `httpx` — transitive, used by the SDKs

See `requirements.txt`.

## License

This Python port is for educational/demo purposes. The original Claude Code
is © Anthropic under their license.
