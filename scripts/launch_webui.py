"""Launcher that double-forks to fully detach the webui from the bash subshell."""
import os
import subprocess
import sys
import time
from pathlib import Path

env = dict(os.environ)
env["PYTHONPATH"] = "/home/z/my-project"
log = open("/tmp/webui.log", "w")
p = subprocess.Popen(
    [sys.executable, "/home/z/my-project/webui.py", "--port", "7860"],
    cwd="/home/z/my-project",
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,  # detach from this script's process group
)
print(f"Spawned PID {p.pid}, parent exiting.")
sys.exit(0)
