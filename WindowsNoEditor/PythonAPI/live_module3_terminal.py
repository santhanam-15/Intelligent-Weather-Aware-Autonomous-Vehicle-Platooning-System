#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
MODULE 3 TWIN-WORLD LIVE DECISION TERMINAL
================================================================================
Monitors, colorizes, and streams real-time Module 3 safety projections,
rear-threat predictions, evasion state machine transitions, candidate lane
validations, and platoon rejoining telemetry.
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

for p in (PROJECT_ROOT, SCRIPT_DIR, os.path.join(PROJECT_ROOT, "WindowsNoEditor")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Module3.config import MODULE3_LOG_FILE, MODULE3_MODE


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
C_GREEN_BG = "\033[97;42m"
C_MAGENTA_BG = "\033[97;45m"
C_CYAN_BG = "\033[97;46m"


def colorize_record(line: str) -> str:
    line = line.rstrip()
    if not line:
        return line

    if "====" in line:
        return f"{C_CYAN}{line}{C_RESET}"

    if "[MODULE 3 CONSOLE]" in line:
        return f"{C_BOLD}{C_GREEN}{line}{C_RESET}"

    if "[MODULE 3 STARTUP SUMMARY]" in line:
        return f"{C_BOLD}{C_CYAN}{line}{C_RESET}"

    if "[MODULE3 TRACE]" in line:
        return f"{C_BOLD}{C_CYAN}{line}{C_RESET}"

    if "[EMERGENCY_BRAKE]" in line or "Emergency |" in line or "EMERGENCY_NO_SAFE_ESCAPE" in line:
        return f"{C_BOLD}{C_RED_BG} {line} {C_RESET}"

    if "Risk=CRITICAL" in line:
        return f"{C_BOLD}{C_RED}>> {line}{C_RESET}"

    if "Risk=HIGH_RISK" in line:
        return f"{C_RED}{line}{C_RESET}"

    if "Risk=CAUTION" in line:
        return f"{C_YELLOW}{line}{C_RESET}"

    if "Validation=SAFE" in line:
        return f"{C_GREEN}{line}{C_RESET}"

    if "Validation=REJECTED" in line:
        return f"{C_YELLOW}{line}{C_RESET}"

    if "EVADING_LEFT" in line or "EVADING_RIGHT" in line:
        return f"{C_BOLD}{C_MAGENTA}{line}{C_RESET}"

    if "State=REJOINING" in line or "Decision=APPROVED" in line or "REJOINING" in line:
        return f"{C_BOLD}{C_CYAN}{line}{C_RESET}"

    if "State Transition" in line or "LEADER_FAILOVER" in line:
        return f"{C_BOLD}{C_WHITE}{line}{C_RESET}"

    if "Risk=NORMAL" in line:
        return f"{C_GREEN}{line}{C_RESET}"

    return line


def main():
    parser = argparse.ArgumentParser(description="Live Module 3 Decision Terminal Monitor")
    parser.add_argument("--log-file", type=str, default=MODULE3_LOG_FILE, help="Path to module3_decisions.log")
    args = parser.parse_args()

    log_path = os.path.abspath(args.log_file)
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass

    print(f"\n{C_BOLD}{C_GREEN}[MODULE 3 CONSOLE] Dedicated Module 3 logging window connected.{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}[MODULE 3 STARTUP SUMMARY]{C_RESET}")
    print(f"  - Active Process PID:      {os.getpid()}")
    print(f"  - Log Channel:             Module 3 Twin-World Safety Channel")
    print(f"  - Operating Mode:          {MODULE3_MODE.upper()}")
    print(f"  - Target Log File:         {log_path}")
    print(f"  - Logging Connection:      SUCCESS\n")

    print(f"{C_BOLD}{C_CYAN}{'=' * 80}")
    print(f"  CARLA 0.9.9 - MODULE 3 LIVE DECISION & REAR EVASION TERMINAL")
    print(f"  Operating Mode: {MODULE3_MODE.upper()}  |  Log: {log_path}")
    print(f"  Streaming: Rear-Impact Risk, Candidate Validation, Evasion & Rejoining")
    print(f"{'=' * 80}{C_RESET}\n")

    print(f"{C_YELLOW}Waiting for Module 3 decisions stream...{C_RESET}")

    # Wait for file to exist
    while not os.path.exists(log_path):
        time.sleep(0.2)

    print(f"{C_GREEN}[CONNECTED] Streaming live Module 3 decisions:{C_RESET}\n")

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        # Read last 8KB if already populated
        f.seek(max(0, size - 8192), os.SEEK_SET)
        initial_lines = f.readlines()
        for line in initial_lines:
            print(colorize_record(line))

        while True:
            line = f.readline()
            if line:
                print(colorize_record(line))
            else:
                time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}Module 3 Live Decision Terminal closed by user.{C_RESET}")
        sys.exit(0)

