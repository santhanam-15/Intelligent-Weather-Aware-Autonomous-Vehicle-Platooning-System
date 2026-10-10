#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
CARLA SIMULATION REAL-TIME LIVE LOG MONITOR
================================================================================
Monitors, colorizes, and streams real-time simulation logs, tactical MAPPO
actions, physical controls, traffic light overrides, and ghost driver intrusions.
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")

# Enable Windows 10/11 ANSI VT Escape sequences
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
C_GREEN_BG = "\033[97;42m"
C_BLUE_BG = "\033[97;44m"


def format_simulation_line(line: str) -> str:
    line = line.rstrip()
    if not line:
        return line

    if "[STOP EVENT]" in line:
        return f"{C_BOLD}{C_RED_BG} {line} {C_RESET}"
    elif "EXACT STOP REASON:" in line:
        return f"{C_BOLD}{C_RED}  {line}{C_RESET}"
    elif "[STOP RECOVERY]" in line:
        return f"{C_BOLD}{C_GREEN_BG} {line} {C_RESET}"
    elif "[AUTOPILOT]" in line:
        return f"{C_CYAN}{line}{C_RESET}"
    elif "[MAPPO]" in line:
        return f"{C_GREEN}{line}{C_RESET}"
    elif "[ACTION_MANAGER]" in line:
        return f"{C_GREEN}{line}{C_RESET}"
    elif "[CONTROL]" in line:
        if "override=RED_LIGHT_STOP" in line or "override=COLLISION_RISK" in line or "override=INSUFFICIENT_HEADWAY" in line:
            return f"{C_BOLD}{C_RED}{line}{C_RESET}"
        return f"{C_YELLOW}{line}{C_RESET}"
    elif "[GHOST SCENARIO TRIGGERED]" in line or "[GHOST SCENARIO]" in line or "[GHOST]" in line:
        return f"{C_BOLD}{C_MAGENTA}{line}{C_RESET}"
    elif "[WEATHER]" in line:
        return f"{C_BOLD}{C_BLUE}{line}{C_RESET}"
    elif "[TICK" in line:
        return f"{C_BOLD}{C_WHITE}{line}{C_RESET}"
    elif "[UPDATE]" in line or "[LOSS]" in line or "[REWARD]" in line or "[ENTROPY]" in line:
        return f"{C_CYAN}{line}{C_RESET}"
    elif "[CHECKPOINT LOAD SUCCESS]" in line or "[CHECKPOINT SAVE]" in line:
        return f"{C_BOLD}{C_BLUE_BG} {line} {C_RESET}"
    elif "MAPPO ACTION TOTALS" in line:
        return f"{C_BOLD}{C_CYAN}{line}{C_RESET}"
    elif "[MODULE 2]" in line:
        return f"{C_GREEN}{line}{C_RESET}"
    elif "WARNING" in line or "WARN" in line:
        return f"{C_YELLOW}{line}{C_RESET}"
    elif "ERROR" in line:
        return f"{C_BOLD}{C_RED}{line}{C_RESET}"

    return line


def find_active_log_file(explicit_path: str = None) -> str:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    candidates = [
        os.path.join(LOG_DIR, "uncertain_platoon.log"),
        os.path.join(LOG_DIR, "module2_platoon.log"),
    ]

    valid = [p for p in candidates if os.path.exists(p)]
    if not valid:
        default_file = candidates[0]
        os.makedirs(os.path.dirname(default_file), exist_ok=True)
        with open(default_file, "a") as f:
            f.write(f"--- Log session initialized at {datetime.datetime.now()} ---\n")
        return default_file

    # Return the file with latest modification time
    return max(valid, key=lambda p: os.path.getmtime(p))


def print_header(log_path: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 80
    print(f"{C_BOLD}{C_CYAN}{sep}")
    print("   CARLA 0.9.9 - REAL-TIME PLATOON SIMULATION LIVE LOG STREAM")
    print("   Tactical MAPPO Decisions | Controls | Overrides | Ghost Intrusions")
    print(f"{sep}{C_RESET}")
    print(f"  {C_BOLD}Active Time:{C_RESET}     {now}")
    print(f"  {C_BOLD}Target Log:{C_RESET}      {log_path}")
    print(f"  {C_BOLD}Log Size:{C_RESET}        {os.path.getsize(log_path) / 1024:.1f} KB")
    print(f"{C_CYAN}{sep}{C_RESET}\n")
    print(f"{C_YELLOW}[STREAM READY] Tailing live simulation telemetry... (Ctrl+C to exit){C_RESET}\n")


def tail_log(log_path: str, poll_interval: float = 0.15):
    last_size = 0
    last_read_time = time.time()

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        # Seek near end to catch recent events
        f.seek(0, os.SEEK_END)
        end_pos = f.tell()
        if end_pos > 0:
            f.seek(max(0, end_pos - 8192))
            recent_lines = f.readlines()
            for l in recent_lines[-35:]:
                print(format_simulation_line(l))

        while True:
            line = f.readline()
            if line:
                last_read_time = time.time()
                print(format_simulation_line(line))
            else:
                time.sleep(poll_interval)
                # If file was truncated or rotated by runner, reopen
                try:
                    curr_size = os.path.getsize(log_path)
                    if curr_size < f.tell():
                        print(f"\n{C_YELLOW}[LOG ROTATION DETECTED] Reopening {log_path}...{C_RESET}\n")
                        f.seek(0)
                except Exception:
                    pass

                # Idle watchdog indicator
                idle = time.time() - last_read_time
                if idle > 20.0:
                    sys.stdout.write(f"\r{C_YELLOW}[WAITING FOR SIMULATION TICK... (idle {int(idle)}s)]{C_RESET}  ")
                    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="CARLA Simulation Real-Time Live Log Monitor")
    parser.add_argument("--file", type=str, default=None, help="Specific log file path to follow")
    parser.add_argument("--mode", choices=("sim", "decisions", "training"), default="sim",
                        help="Log stream mode: sim (uncertain_platoon.log), decisions (mappo_decisions.log), training (mappo_training_stream.log)")
    parser.add_argument("--once", action="store_true", help="Print recent lines and exit without streaming")
    args = parser.parse_args()

    if args.mode == "decisions":
        target_file = os.path.join(LOG_DIR, "mappo_decisions.log")
    elif args.mode == "training":
        target_file = os.path.join(LOG_DIR, "mappo_training_stream.log")
    elif args.file:
        target_file = os.path.abspath(args.file)
    else:
        target_file = find_active_log_file()

    print_header(target_file)

    if args.once:
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for l in lines[-35:]:
                print(format_simulation_line(l))
        return

    try:
        tail_log(target_file)
    except KeyboardInterrupt:
        print(f"\n{C_BOLD}{C_YELLOW}[MONITOR STOPPED] Exited by user.{C_RESET}")


if __name__ == "__main__":
    main()
