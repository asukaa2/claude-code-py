"""End-to-end smoke test for the agent.

Uses whichever provider has an API key in the environment (checks Anthropic
first, then DeepSeek, Gemini, HuggingFace). Skips with a clear message if
no key is set.
"""
import os
import sys

from claude_code_py import Agent, AgentConfig, AgentEvent, EventType, ProviderConfig
from claude_code_py.permissions import PermissionDecision

# Pick the first provider with a key in the env.
PROVIDER_CANDIDATES = [
    ("anthropic", "ANTHROPIC_API_KEY", "claude-3-5-sonnet-latest"),
    ("deepseek", "DEEPSEEK_API_KEY", "deepseek-chat"),
    ("gemini", "GEMINI_API_KEY", "gemini-1.5-flash"),
    ("huggingface", "HF_TOKEN", "meta-llama/Llama-3.1-8B-Instruct"),
]
chosen = None
for pid, env_var, model in PROVIDER_CANDIDATES:
    if os.environ.get(env_var):
        chosen = (pid, env_var, model)
        break
if chosen is None:
    print("SKIP: no provider API key found in env. Set one of:")
    for _, env_var, _ in PROVIDER_CANDIDATES:
        print(f"  export {env_var}=…")
    sys.exit(0)

pid, env_var, model = chosen
print(f"Using provider: {pid} (model {model}, key from ${env_var})")

cfg = AgentConfig(
    working_dir="/home/z/my-project",
    provider=ProviderConfig(provider=pid, model=model),
    auto_approve_read_tools=True,
    auto_approve_write_tools=True,
    auto_approve_bash=True,
    max_iterations=8,
)

events: list[AgentEvent] = []


def on_event(e: AgentEvent) -> None:
    events.append(e)
    if e.type is EventType.TOOL_CALL:
        print(f"  [TOOL_CALL] {e.payload['name']} {e.payload['arguments']}")
    elif e.type is EventType.TOOL_RESULT:
        ok = "OK" if e.payload["ok"] else "ERR"
        snippet = e.payload["content"][:120].replace("\n", " | ")
        print(f"  [TOOL_RESULT {ok}] {snippet}")
    elif e.type is EventType.ASSISTANT_TEXT and e.payload.get("final"):
        print(f"  [FINAL] {e.payload['text'][:200]}")
    elif e.type is EventType.ERROR:
        print(f"  [ERROR] {e.payload['message']}")
    elif e.type is EventType.ITERATION_LIMIT:
        print(f"  [LIMIT] {e.payload['message']}")


agent = Agent(cfg, permission_callback=lambda n, a: PermissionDecision.ALLOW)
print("Asking agent to list files in working dir...")
final = agent.run("List the files in the current working directory using the list_files tool. Be brief.", on_event=on_event)
print("=== FINAL ===")
print(final[:500])
print("=== events:", len(events))
