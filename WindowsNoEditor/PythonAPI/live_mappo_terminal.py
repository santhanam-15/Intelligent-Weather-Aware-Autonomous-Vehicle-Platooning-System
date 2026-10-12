#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
CARLA 0.9.9 - LIVE MAPPO TACTICAL CONTROL TERMINAL
================================================================================
Monitors, colorizes, and streams real-time MAPPO decisions, follower catch-up
headway telemetry, route waypoints, speed adjustments, and leader failovers.
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

for p in (PROJECT_ROOT, SCRIPT_DIR, os.path.join(PROJECT_ROOT, "WindowsNoEditor")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Module2.config import DECISION_LOG_FILE


def _enable_vt_mode():
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            hStdOut = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode))
            mode.value |= 4  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(hStdOut, mode)
        except Exception:
            pass


_enable_vt_mode()

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_WHITE = "\033[97m"
C_RED_BG = "\033[97;41m"
C_YELLOW_BG = "\033[90;43m"
C_CYAN_BG = "\033[97;44m"


def colorize_mappo_line(line: str) -> str:
    line = line.rstrip()
    if not line:
        return line

    if "====" in line:
        return f"{C_CYAN}{line}{C_RESET}"

    if "[MAPPO CONSOLE]" in line:
        return f"{C_BOLD}{C_GREEN}{line}{C_RESET}"

    if "[MAPPO STARTUP SUMMARY]" in line:
        return f"{C_BOLD}{C_CYAN}{line}{C_RESET}"

    if "[MAPPO FAILOVER]" in line or "FAILOVER" in line:
        return f"{C_BOLD}{C_RED_BG} {line} {C_RESET}"

    if "[MAPPO STEP" in line:
        return f"{C_BOLD}{C_CYAN}{line}{C_RESET}"

    if "CATCH_UP_ACTIVE" in line:
        return f"{C_BOLD}{C_YELLOW}{line}{C_RESET}"

    if "EXECUTED=ACCELERATE" in line or "Act=ACCELERATE" in line:
        return f"{C_GREEN}{line}{C_RESET}"

    if "EXECUTED=DECELERATE" in line or "Act=DECELERATE" in line:
        return f"{C_YELLOW}{line}{C_RESET}"

    if "CHANGE_LEFT" in line or "CHANGE_RIGHT" in line:
        return f"{C_MAGENTA}{line}{C_RESET}"

    if "CRITICAL_PROXIMITY" in line or "INSUFFICIENT_HEADWAY" in line or "COLLISION" in line:
        return f"{C_RED}{line}{C_RESET}"

    if "TRAINING UPDATE" in line:
        return f"{C_BOLD}{C_WHITE}{line}{C_RESET}"

    return line


def main():
    parser = argparse.ArgumentParser(description="Live MAPPO Tactical Terminal Monitor")
    parser.add_argument("--log-file", type=str, default=DECISION_LOG_FILE, help="Path to mappo_decisions.log")
    args = parser.parse_args()

    log_path = os.path.abspath(args.log_file)
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass

    print(f"\n{C_BOLD}{C_GREEN}[MAPPO CONSOLE] Dedicated MAPPO logging window connected.{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}[MAPPO STARTUP SUMMARY]{C_RESET}")
    print(f"  - Active Process PID:      {os.getpid()}")
    print(f"  - Log Channel:             MAPPO Tactical Control & Platoon Stream")
    print(f"  - Target Log File:         {log_path}")
    print(f"  - Logging Connection:      SUCCESS\n")

    print(f"{C_BOLD}{C_CYAN}{'=' * 80}")
    print(f"  CARLA 0.9.9 - LIVE MAPPO TACTICAL CONTROL TERMINAL")
    print(f"  Streaming: Actions, Target Speeds, Headway Gaps, Catch-Up & Failover")
    print(f"{'=' * 80}{C_RESET}\n")

    print(f"{C_YELLOW}Waiting for MAPPO tactical stream to initialize...{C_RESET}")

    # Wait for log file
    while not os.path.exists(log_path):
        time.sleep(0.2)

    print(f"{C_GREEN}[CONNECTED] Streaming live MAPPO tactical control telemetry:{C_RESET}\n")

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        # Read last 8KB if already populated
        f.seek(max(0, size - 8192), os.SEEK_SET)
        initial_lines = f.readlines()
        for line in initial_lines:
            print(colorize_mappo_line(line))

        while True:
            line = f.readline()
            if line:
                print(colorize_mappo_line(line))
            else:
                time.sleep(0.04)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}MAPPO Live Terminal closed by user.{C_RESET}")
        sys.exit(0)

