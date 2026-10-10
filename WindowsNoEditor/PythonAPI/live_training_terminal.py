#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
LIVE MAPPO TRAINING TERMINAL - REAL-TIME PPO METRICS & TELEMETRY STREAM
================================================================================
Displays structured, colored live telemetry for MAPPO training updates:
  [TRAIN START] [ROLLOUT] [UPDATE] [LOSS] [REWARD] [ENTROPY] [KL]
  [GRADIENT NORM] [ACTION DIST] [CHECKPOINT SAVE] [TRAIN COMPLETE]

Supports:
- Live streaming from mappo_training_stream.log
- Status watchdog (Progressing vs Stalled)
- Verification mode (--verify-train-step)
- Automatic simulation launcher (--launch-sim)
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
MODULE2_DIR = os.path.join(PROJECT_ROOT, "WindowsNoEditor", "Module2")

for p in (PROJECT_ROOT, SCRIPT_DIR, os.path.join(PROJECT_ROOT, "WindowsNoEditor")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from Module2.config import CHECKPOINT_FILE, CHECKPOINT_ROOT_DIR, LOG_DIR
from Module2.mappo.trainer import MAPPOTrainer, compute_file_sha256

STREAM_LOG = os.path.join(SCRIPT_DIR, "mappo_training_stream.log")
SNAPSHOT_LOG = os.path.join(SCRIPT_DIR, "mappo_training.log")


# Windows ANSI Color support setup
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
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_MAGENTA = "\033[95m"
C_RED = "\033[91m"
C_WHITE_BG_BLUE = "\033[97;44m"


def format_tag_line(line: str) -> str:
    line = line.rstrip()
    if "[UPDATE]" in line:
        return f"{C_BOLD}{C_GREEN}{line}{C_RESET}"
    elif "[LOSS]" in line:
        return f"{C_MAGENTA}{line}{C_RESET}"
    elif "[REWARD]" in line:
        return f"{C_GREEN}{line}{C_RESET}"
    elif "[ENTROPY]" in line:
        return f"{C_CYAN}{line}{C_RESET}"
    elif "[KL]" in line:
        return f"{C_YELLOW}{line}{C_RESET}"
    elif "[GRADIENT NORM]" in line:
        return f"{C_CYAN}{line}{C_RESET}"
    elif "[CHECKPOINT SAVE]" in line:
        return f"{C_WHITE_BG_BLUE}{C_BOLD}{line}{C_RESET}"
    elif "[TRAIN START]" in line or "[TRAIN COMPLETE]" in line:
        return f"{C_BOLD}{C_YELLOW}{line}{C_RESET}"
    elif "[ROLLOUT]" in line:
        return f"{C_YELLOW}{line}{C_RESET}"
    elif "[ACTION DIST]" in line:
        return f"{C_CYAN}{line}{C_RESET}"
    return line


def print_banner(trainer: MAPPOTrainer):
    ckpt_path = os.path.abspath(trainer.get_checkpoint_path())
    ckpt_hash = trainer.active_checkpoint_hash or compute_file_sha256(ckpt_path)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sep = "=" * 70
    print(f"{C_BOLD}{C_CYAN}{sep}")
    print("   MAPPO LIVE TRAINING TERMINAL - VEHICLE PLATOONING SYSTEM")
    print(f"   CARLA 0.9.9  |  PyTorch {torch.__version__}  |  Python {sys.version.split()[0]}")
    print(f"{sep}{C_RESET}")
    print(f"  {C_BOLD}Started at:{C_RESET}             {now}")
    print(f"  {C_BOLD}Active Checkpoint:{C_RESET}      {ckpt_path}")
    print(f"  {C_BOLD}Checkpoint SHA-256:{C_RESET}     {ckpt_hash}")
    print(f"  {C_BOLD}Checkpoint Step:{C_RESET}        {trainer.update_step}")
    print(f"  {C_BOLD}Actor Dimensions:{C_RESET}       {trainer.actor.net[0].in_features} -> 64 -> 64 -> {trainer.actor.net[-1].out_features}")
    print(f"  {C_BOLD}Critic Dimensions:{C_RESET}      {trainer.critic.net[0].in_features} -> 128 -> 64 -> 1")
    print(f"  {C_BOLD}Device:{C_RESET}                 {trainer.device}")
    print(f"{C_CYAN}{sep}{C_RESET}\n")
    print(f"{C_YELLOW}[TRAIN START] Terminal initialized. Listening for live training updates...{C_RESET}\n")


def verify_train_step(trainer: MAPPOTrainer):
    """Run an automated synthetic rollout update to verify complete train pipeline."""
    print(f"{C_BOLD}{C_YELLOW}[TEST VERIFICATION] Executing synthetic MAPPO rollout & train step...{C_RESET}")
    import numpy as np

    num_agents = 3
    obs_dim = trainer.actor.net[0].in_features
    global_dim = trainer.critic.net[0].in_features

    # Synthetic rollout of 128 transitions
    for step in range(128):
        obs = np.random.randn(num_agents, obs_dim).astype(np.float32)
        glob_state = np.random.randn(global_dim).astype(np.float32)
        next_glob_state = np.random.randn(global_dim).astype(np.float32)
        actions, log_probs, _, _ = trainer.select_action(obs, deterministic=False, return_details=True)
        val = trainer.get_value(glob_state)
        rewards = np.array([1.2, 0.8, 1.0], dtype=np.float32)
        done = False
        trainer.buffer.store_transition(
            obs, glob_state, actions, log_probs, rewards, val, done, next_global_state=next_glob_state
        )

    metrics = trainer.train()
    print(f"{C_BOLD}{C_GREEN}[TEST VERIFICATION SUCCESS] Update step advanced to {trainer.update_step}. Saved hash: {trainer.active_checkpoint_hash[:16]}...{C_RESET}")
    return metrics


def tail_training_stream(stream_path: str, poll_interval: float = 0.25):
    """Continuously tail and format stream log in real-time."""
    if not os.path.exists(stream_path):
        # Create empty file
        try:
            with open(stream_path, "a") as f:
                f.write(f"[TRAIN START] Session initialized at {datetime.datetime.now()}\n")
        except Exception:
            pass

    last_pos = 0
    last_update_time = time.time()

    with open(stream_path, "r") as f:
        # Seek to beginning to replay session or end if large
        f.seek(0, os.SEEK_END)
        last_pos = f.tell()
        # If file has recent content, read last 20 lines
        if last_pos > 0:
            f.seek(max(0, last_pos - 4096))
            lines = f.readlines()
            for l in lines[-25:]:
                print(format_tag_line(l))

        while True:
            line = f.readline()
            if line:
                last_update_time = time.time()
                print(format_tag_line(line))
            else:
                time.sleep(poll_interval)
                # Watchdog status
                idle_sec = time.time() - last_update_time
                if idle_sec > 45.0:
                    status_line = f"\r{C_YELLOW}[STATUS: WAITING FOR ROLLOUT BUFFER (idle {int(idle_sec)}s)]{C_RESET}"
                    sys.stdout.write(status_line)
                    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Live MAPPO Training Terminal")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint file path")
    parser.add_argument("--verify-train-step", action="store_true", help="Execute synthetic train step verification")
    parser.add_argument("--launch-sim", action="store_true", help="Launch CARLA simulation in training mode alongside")
    args = parser.parse_args()

    trainer = MAPPOTrainer(
        obs_dim=32,
        action_dim=5,
        global_state_dim=128,
    )
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    else:
        trainer.load_checkpoint()

    print_banner(trainer)

    if args.verify_train_step:
        verify_train_step(trainer)
        return

    sim_proc = None
    if args.launch_sim:
        sim_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "run_module_2.py"), "--mode", "train"]
        if args.checkpoint:
            sim_cmd.extend(["--checkpoint", args.checkpoint])
        print(f"{C_CYAN}[LAUNCHER] Starting CARLA simulation in training mode: {' '.join(sim_cmd)}{C_RESET}")
        sim_proc = subprocess.Popen(sim_cmd, cwd=SCRIPT_DIR)

    try:
        tail_training_stream(STREAM_LOG)
    except KeyboardInterrupt:
        print(f"\n{C_BOLD}{C_YELLOW}[TRAIN COMPLETE] Live terminal detached by user.{C_RESET}")
        if sim_proc:
            sim_proc.terminate()


if __name__ == "__main__":
    main()
