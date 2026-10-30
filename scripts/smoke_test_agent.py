"""End-to-end smoke test for the agent: ask it to list files in /tmp."""
from claude_code_py import Agent, AgentConfig, AgentEvent, EventType
from claude_code_py.permissions import PermissionDecision

# Auto-approve everything for the smoke test.
cfg = AgentConfig(
    working_dir="/home/z/my-project",
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
