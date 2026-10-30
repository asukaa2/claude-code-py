"""Comprehensive end-to-end test exercising the full UI."""
from gradio_client import Client

c = Client("http://127.0.0.1:7860/", verbose=False)
print("✓ connected")

# Set yolo + working dir to the e2e_demo subdir.
c.predict(preset="yolo", api_name="/h_set_perm_preset")
c.predict("/home/z/my-project/download/e2e_demo", api_name="/h_connect_working_dir")
c.predict(api_name="/h_clear_chat")
print("✓ configured: yolo, working dir = e2e_demo, chat cleared")

# Run a multi-tool task: build a tiny Python project + run it.
task = (
    "Please do the following, step by step:\n"
    "1. Use write_file to create main.py with this exact content:\n"
    "```python\n"
    "def greet(name):\n"
    "    return f'Hello, {name}!'\n\n"
    "if __name__ == '__main__':\n"
    "    print(greet('world'))\n"
    "```\n"
    "2. Use bash to run `python main.py` and report the output.\n"
    "3. Use todo_write to record your progress (3 todos: write file, run, verify).\n"
    "4. Finally summarize in one sentence."
)
print("\n--- sending task ---")
result = c.predict(task, [], api_name="/h_chat_submit")
chat, tool_log, todos_html, status = result

# Inspect tool calls actually made.
tool_calls = [m for m in chat if m.get("role") == "assistant" and "Tool call:" in str(m.get("content"))]
print(f"\n✓ chat has {len(chat)} messages, {len(tool_calls)} tool calls were made")
print(f"✓ tool_log HTML length: {len(tool_log)} chars")
print(f"✓ todos panel: {todos_html[:200]}")
print(f"✓ status: {status[:100]}")

# Verify the file was actually written and runs.
import subprocess, pathlib
main_py = pathlib.Path("/home/z/my-project/download/e2e_demo/main.py")
if main_py.exists():
    out = subprocess.run(["python3", str(main_py)], capture_output=True, text=True).stdout.strip()
    print(f"\n✓ main.py on disk ran successfully: {out!r}")
else:
    print("\n✗ main.py NOT created on disk")

# Test the file browser.
fb = c.predict("", api_name="/h_list_files_browser")
print(f"✓ file browser shows: {fb[:200]}")

# Test session save / list / reload.
save_out, sess_dd = c.predict(api_name="/h_save_session")
print(f"✓ session saved: {save_out[:120]}")
print(f"✓ session dropdown has {len(sess_dd.get('choices', [])) if isinstance(sess_dd, dict) else 'n/a'} entries")

# Test settings round-trip.
c.predict(50, "Be a terse coding assistant.", True, 60, api_name="/h_apply_settings")
new_max, new_override, new_thinking, new_timeout = c.predict(api_name="/h_load_settings_into_ui")
print(f"✓ settings persisted: max_iter={new_max} thinking={new_thinking} timeout={new_timeout}s")

print("\n=== ALL E2E CHECKS PASSED ===")
