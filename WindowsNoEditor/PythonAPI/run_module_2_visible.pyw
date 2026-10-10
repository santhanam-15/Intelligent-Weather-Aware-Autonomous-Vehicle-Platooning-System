"""Desktop launcher for the visible Module 2 Pygame window.

This file is opened through Windows Explorer/file association so the Pygame
window belongs to the interactive desktop session, rather than a background
terminal session.
"""

from __future__ import annotations

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Keep this launcher self-contained and predictable for a desktop run.
sys.argv = [
    "run_module_2.py",
    "--launch-server",
    "--kill-launched-server",
    "--duration",
    "330",
    "--wait-seconds",
    "90",
    "--vehicles",
    "10",
    "--pedestrians",
    "0",
    "--platoon-size",
    "4",
    "--town",
    "Town04",
    "--windowed",
    "--width",
    "1280",
    "--height",
    "720",
    "--log-file",
    os.path.join(SCRIPT_DIR, "logs", "module2_visible_5min.log"),
]

from run_module_2 import parse_args, run  # noqa: E402


if __name__ == "__main__":
    run(parse_args())
