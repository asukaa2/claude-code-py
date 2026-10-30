"""Drive the running webui end-to-end through the Gradio client."""
import json
from gradio_client import Client

c = Client("http://127.0.0.1:7860/", verbose=False)
print("connected.")

# 1) Set permission preset to yolo so tool calls aren't blocked.
print("\n[1] set perm preset -> yolo")
out = c.predict(api_name="/h_set_perm_preset", preset="yolo")
print("  ->", out[:120])

# 2) Send a message that triggers a tool call, then a final answer.
print("\n[2] send chat: 'list files in the current working dir, then say hi'")
# h_chat_submit is a generator (streaming). client.predict returns the final value.
# h_chat_submit inputs (in order): [Message, Conversation]. Pass positional.
result = c.predict(
    "Use the list_files tool to list files in the current working directory, then briefly summarize what you see. Be concise.",
    [],
    api_name="/h_chat_submit",
)
print("  -> got result tuple of len:", len(result))
chat, tool_log, todos, status = result
print("\n[chat]:")
for m in chat:
    role = m.get("role")
    raw = m.get("content", "")
    if isinstance(raw, list):
        # gradio 6 content-blocks form: extract text from each block
        raw = " ".join(
            (b.get("text") or b.get("content") or "")
            for b in raw if isinstance(b, dict)
        )
    s = str(raw)[:300].replace("\n", " | ")
    print(f"  [{role}] {s}")
print("\n[tool_log] (first 500 chars):")
print(tool_log[:500])
print("\n[status]:", status[:120])

# 3) Test the file browser.
print("\n[3] file browser list '.'")
fb = c.predict(api_name="/h_list_files_browser", path="")
print("  ->", fb[:300])

print("\nALL GOOD.")
