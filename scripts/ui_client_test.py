"""End-to-end UI test: drive the running webui via gradio_client."""
import time
from gradio_client import Client

c = Client("http://localhost:7860/", verbose=False)
print("Connected:", c.view_api(return_format="dict") is not None)

# Find the send-button click endpoint. With a .then() chain attached, the
# gradio client exposes the root fn by endpoint name. We use the predict API
# with the component-order input.
# The send_btn.click inputs are [msg_in, chatbot]; outputs are
# [chatbot, tool_log, todos_html, status_md].
print("Endpoints:", [e.fn_name for e in c.endpoints if e.fn_name][:8])

# Trigger send. Pass a tiny prompt that should produce a tool call.
result = c.predict(
    message="Hello! Reply with one short sentence — no tools needed.",
    api_name=None,  # let client pick
) if False else None

# Use the named API: send_btn.click
api = c.get_endpoint("/send_btn_click") if hasattr(c, "get_endpoint") else None
print("send_btn api:", api)

# Fall back to the raw fn index. The first event handler with 2 inputs /
# 4 outputs is our send button.
import json
print("endpoints raw:")
for e in c.endpoints:
    if e.fn_name:
        print(" ", e.fn_name, "inputs=", e.parameters, "outputs=", e.returns)
