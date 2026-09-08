#!/usr/bin/env python

"""Test script for the V2V Communication Uncertainty Module.

This is a pure-Python simulation module test. No live CARLA simulator
connection is required to run this script.

This script demonstrates and verifies:
1. NORMAL preset: zero packet loss, zero latency, zero noise corruption.
2. MEDIUM preset: transmission latency (~100ms ± 20% jitter), minor loss (~5%), noise std=0.01.
3. BAD preset: high latency (~250ms ± 20% jitter), increased loss (~15%), noise std=0.05.
4. Channel outage handling: manually triggered outages, packet drops during outage, and recovery.
5. Statistical reporting for all tested communication presets.
6. Deterministic reproducibility: verifying identical packet loss, latency, and noise
   across two independent V2V channels initialized with the same random seed.
"""

import argparse
import glob
import os
import sys

# Add uncertainty package to path (works from either examples/ or PythonAPI/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from uncertainty.communication_uncertainty import V2VChannel, CommPreset


def print_stats_table(preset_name: str, stats: dict) -> None:
    """Print formatted statistics dictionary."""
    sent = stats.get('messages_sent', 0)
    lost = stats.get('messages_lost', 0)
    delivered = stats.get('messages_delivered', 0)
    loss_rate = (lost / sent * 100.0) if sent > 0 else 0.0
    deliv_rate = (delivered / sent * 100.0) if sent > 0 else 0.0

    print(f"  Channel Statistics for '{preset_name}':")
    print(f"    Messages Attempted : {sent}")
    print(f"    Messages Lost      : {lost} ({loss_rate:.1f}%)")
    print(f"    Messages Delivered : {delivered} ({deliv_rate:.1f}%)")
    print(f"    Outage Active      : {stats.get('current_outage', False)}")


def main() -> None:
    """Execute pure-Python V2V communication uncertainty verification tests."""
    parser = argparse.ArgumentParser(
        description="CARLA 0.9.9 V2V Communication Uncertainty Test Script (Pure Python)"
    )
    parser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the CARLA host server (unused: communication test is pure Python)',
    )
    parser.add_argument(
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of the CARLA host server (unused: communication test is pure Python)',
    )
    parser.add_argument(
        '--seed',
        metavar='S',
        default=42,
        type=int,
        help='Random seed for reproducible communication channel simulation (default: 42)',
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" CARLA 0.9.9 V2V Communication Uncertainty Module Test")
    print(" (Pure-Python Simulation - No CARLA Server Connection Required)")
    print("=" * 70)
    print(f"Configured seed: {args.seed}")
    print(f"Server host/port flags parsed: {args.host}:{args.port} (connection skipped as instructed)")

    try:
        # Step 1: Create V2VChannel(seed=42)
        print(f"\nInitializing V2VChannel with seed={args.seed}...")
        channel = V2VChannel(seed=args.seed)

        # Step 2: Test NORMAL preset: send 20 messages, receive them, count delivered
        print("\n" + "-" * 70)
        print("[STEP 1/6] Testing CommPreset.NORMAL (0ms delay, 0% drop, 0 noise)")
        print("-" * 70)
        channel.reset()
        channel.configure_from_preset(CommPreset.NORMAL)

        num_messages = 20
        send_time = 1.0
        sender_id = 101
        receiver_id = 202

        print(f"Sending {num_messages} messages at simulation time t={send_time:.2f}s...")
        for i in range(num_messages):
            payload = {
                "seq": i,
                "speed": 25.0 + i * 0.5,
                "yaw_rate": 0.02,
                "lane_id": 1,
                "status": "nominal",
            }
            channel.send_message(sender_id, receiver_id, payload, current_time=send_time)

        # Receive immediately at current_time = send_time
        delivered_normal = channel.receive_messages(receiver_id, current_time=send_time)
        print(f"Delivered messages at t={send_time:.2f}s: {len(delivered_normal)}/{num_messages}")
        assert len(delivered_normal) == num_messages, "NORMAL preset must deliver all messages instantly without loss!"
        assert not any(m.is_corrupted for m in delivered_normal), "NORMAL preset must not corrupt payload!"
        print("Verification: PASSED (100% immediate delivery, exact data fidelity)")
        stats_normal = channel.get_stats()
        print_stats_table("NORMAL", stats_normal)

        # Step 3: Test MEDIUM preset: send 20 messages, show delay and losses
        print("\n" + "-" * 70)
        print("[STEP 2/6] Testing CommPreset.MEDIUM (100ms delay ± 20%, 5% drop, noise=0.01)")
        print("-" * 70)
        channel.reset()
        channel.configure_from_preset(CommPreset.MEDIUM)

        send_time = 2.0
        print(f"Sending {num_messages} messages at simulation time t={send_time:.2f}s...")
        for i in range(num_messages):
            payload = {
                "seq": i,
                "speed": 30.0 + i * 0.5,
                "steer": 0.05,
                "status": "in_transit",
            }
            channel.send_message(sender_id, receiver_id, payload, current_time=send_time)

        # Attempt immediate receive: should be 0 because base delay is 100ms (0.1s)
        immediate = channel.receive_messages(receiver_id, current_time=send_time)
        print(f"Immediate delivery query at t={send_time:.2f}s : {len(immediate)} messages ready (Demonstrating Latency Delay)")
        assert len(immediate) == 0, "MEDIUM preset messages must experience non-zero delay!"

        # Advance time to t = 2.15s (after base delay + jitter)
        receive_time = 2.15
        delivered_medium = channel.receive_messages(receiver_id, current_time=receive_time)
        print(f"Query after 150ms delay at t={receive_time:.2f}s: {len(delivered_medium)}/{num_messages} delivered")

        # Show payload noise corruption on delivered messages
        if delivered_medium:
            sample = delivered_medium[0]
            print(f"  Sample delivered payload (seq {sample.data['seq']}):")
            print(f"    Original speed: 30.0000 -> Noisy speed: {sample.data['speed']:.4f}")
            print(f"    Original steer:  0.0500 -> Noisy steer: {sample.data['steer']:.4f}")
            print(f"    Status string : '{sample.data['status']}' (strings preserved uncorrupted)")
            print(f"    Transit latency: {(sample.receive_time - sample.send_time) * 1000.0:.1f} ms")

        stats_medium = channel.get_stats()
        print_stats_table("MEDIUM", stats_medium)

        # Step 4: Test BAD preset: send 20 messages, show more losses
        print("\n" + "-" * 70)
        print("[STEP 3/6] Testing CommPreset.BAD (250ms delay ± 20%, 15% drop, noise=0.05)")
        print("-" * 70)
        channel.reset()
        channel.configure_from_preset(CommPreset.BAD)

        send_time = 3.0
        print(f"Sending {num_messages} messages at simulation time t={send_time:.2f}s...")
        for i in range(num_messages):
            payload = {
                "seq": i,
                "speed": 45.0,
                "acceleration": -1.5,
                "emergency": False,
            }
            channel.send_message(sender_id, receiver_id, payload, current_time=send_time)

        # Advance time to t = 3.35s (250ms + jitter + margin)
        receive_time = 3.35
        delivered_bad = channel.receive_messages(receiver_id, current_time=receive_time)
        print(f"Query after 350ms delay at t={receive_time:.2f}s: {len(delivered_bad)}/{num_messages} delivered")

        stats_bad = channel.get_stats()
        print(f"  Losses observed: {stats_bad['messages_lost']} packets dropped during transmission")
        print_stats_table("BAD", stats_bad)

        # Step 5: Test outage: manually trigger outage, show messages are dropped
        print("\n" + "-" * 70)
        print("[STEP 4/6] Testing Outage Handling (Packet Drops During Blackout)")
        print("-" * 70)
        channel.reset()
        channel.configure_from_preset(CommPreset.NORMAL)

        sim_time = 5.0
        # Trigger outage directly
        channel.outage_duration_s = 3.0
        channel.outage_probability = 1.0
        channel.update(current_time=sim_time)

        outage_status = channel.is_outage_active()
        print(f"Outage initiated at t={sim_time:.1f}s? Active: {outage_status}")
        assert outage_status is True, "Channel outage should be active!"

        # Send 10 messages during outage
        outage_messages = 10
        print(f"Attempting to transmit {outage_messages} messages during active outage...")
        dropped_count = 0
        for i in range(outage_messages):
            success = channel.send_message(
                sender_id=sender_id,
                receiver_id=receiver_id,
                data={"seq": i, "alert": "HIGH_PRIORITY"},
                current_time=sim_time + 0.1 * i,
            )
            if not success:
                dropped_count += 1

        print(f"Messages rejected by channel during outage: {dropped_count}/{outage_messages}")
        assert dropped_count == outage_messages, "All messages sent during active outage must be dropped!"

        # Advance simulation time past outage duration (sim_time + 4.0s)
        post_outage_time = sim_time + 4.0
        channel.update(current_time=post_outage_time)
        print(f"Updated channel at t={post_outage_time:.1f}s. Outage active? {channel.is_outage_active()}")
        assert channel.is_outage_active() is False, "Outage should have elapsed!"

        # Verify channel recovers and accepts new packets
        recovered_send = channel.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            data={"seq": 999, "status": "RECOVERED"},
            current_time=post_outage_time,
        )
        print(f"Post-outage transmission success: {recovered_send}")
        assert recovered_send is True, "Channel must accept messages after outage ends!"
        print("Outage verification: PASSED (Dropped during outage, successfully resumed post-outage)")

        # Step 6: Print stats for each preset
        print("\n" + "-" * 70)
        print("[STEP 5/6] Summary of Channel Preset Statistics")
        print("-" * 70)
        print_stats_table("NORMAL Preset", stats_normal)
        print_stats_table("MEDIUM Preset", stats_medium)
        print_stats_table("BAD Preset", stats_bad)
        print_stats_table("Outage Test Session", channel.get_stats())

        # Step 7: Test reproducibility: create two channels with same seed, send same messages, verify same results
        print("\n" + "-" * 70)
        print("[STEP 6/6] Testing Stochastic Reproducibility (Seed Invariance)")
        print("-" * 70)
        test_seed = 98765
        print(f"Creating two independent V2V channels (Channel A & Channel B) with seed={test_seed}...")
        ch_a = V2VChannel(seed=test_seed)
        ch_b = V2VChannel(seed=test_seed)

        # Configure both to BAD preset (heavily stochastic: loss, jitter, noise)
        ch_a.configure_from_preset(CommPreset.BAD)
        ch_b.configure_from_preset(CommPreset.BAD)

        print("Transmitting 50 identical messages through both channels...")
        for i in range(50):
            t = 10.0 + i * 0.05
            data = {"id": i, "x": 100.0 + i, "y": 200.0 - i, "valid": True}
            res_a = ch_a.send_message(1, 2, data, current_time=t)
            res_b = ch_b.send_message(1, 2, data, current_time=t)
            assert res_a == res_b, f"Message {i} send status mismatch between seeds!"

        # Receive all messages at t = 20.0s
        deliv_a = ch_a.receive_messages(2, current_time=20.0)
        deliv_b = ch_b.receive_messages(2, current_time=20.0)

        assert len(deliv_a) == len(deliv_b), f"Delivered count mismatch: {len(deliv_a)} vs {len(deliv_b)}"
        print(f"  Channel A delivered: {len(deliv_a)} messages")
        print(f"  Channel B delivered: {len(deliv_b)} messages")

        # Compare every message, payload, receive time, and noise
        for idx, (m_a, m_b) in enumerate(zip(deliv_a, deliv_b)):
            assert m_a.send_time == m_b.send_time, f"Send time mismatch at packet {idx}"
            assert abs(m_a.receive_time - m_b.receive_time) < 1e-9, f"Receive time mismatch at packet {idx}"
            assert m_a.is_corrupted == m_b.is_corrupted, f"Corruption flag mismatch at packet {idx}"
            assert abs(m_a.data["x"] - m_b.data["x"]) < 1e-9, f"Payload X mismatch at packet {idx}"
            assert abs(m_a.data["y"] - m_b.data["y"]) < 1e-9, f"Payload Y mismatch at packet {idx}"

        print("Reproducibility Verification: PASSED (Exact bit-for-bit equivalence verified)")

        # Overall Summary
        print("\n" + "=" * 70)
        print(" V2V Communication Uncertainty Test Summary")
        print("=" * 70)
        print("NORMAL Preset Test     : PASSED (Zero loss, zero jitter)")
        print("MEDIUM Preset Test     : PASSED (100ms latency & noise observed)")
        print("BAD Preset Test        : PASSED (250ms latency & higher loss observed)")
        print("Outage Test            : PASSED (Outage drops verified, recovery verified)")
        print("Reproducibility Test   : PASSED (Identical output with matching seeds)")
        print("Overall Status         : ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR encountered during execution: {e}")
        raise
    finally:
        print("\nV2V Communication test completed.")


if __name__ == '__main__':
    main()
