#!/usr/bin/env python
"""
run_simulation.py
One-click CARLA Uncertainty Simulation:
  1. Launch CARLA with -dx11 -quality-level=Medium
  2. Connect and wait until ready
  3. Spawn 30 vehicles + 50 walkers
  4. Apply MEDIUM uncertainty scenario
  5. Run with live status dashboard (Ctrl+C to stop)
"""

import glob
import os
import sys
import subprocess
import time
import random
import logging

# ── CARLA egg (Python 3.7 venv has it in site-packages already) ──
try:
    _egg = glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0]
    if _egg not in sys.path:
        sys.path.append(_egg)
except IndexError:
    pass  # already in venv site-packages

import carla  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uncertainty.weather_uncertainty   import WeatherPreset
from uncertainty.scenario_manager      import ScenarioLevel
from uncertainty.uncertainty_manager   import UncertaintyManager

# ─────────────────────────────────────────────
# CONFIG  ← change these if you want
# ─────────────────────────────────────────────
NUM_VEHICLES      = 30
NUM_WALKERS       = 50
SCENARIO_LEVEL    = 'medium'   # low / medium / high / extreme
SEED              = 42
HOST              = '127.0.0.1'
PORT              = 2000
TM_PORT           = 8000
CARLA_WAIT_SEC    = 20         # seconds to wait after launching CARLA
STATUS_EVERY_SEC  = 5          # live dashboard refresh interval

logging.basicConfig(level=logging.WARNING,
                    format='[%(levelname)s] %(message)s')

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def log(msg):
    print(f"  {msg}", flush=True)

def separator(char='─', width=56):
    print('  ' + char * width, flush=True)

# ─────────────────────────────────────────────
# STEP 1 — LAUNCH CARLA
# ─────────────────────────────────────────────

def launch_carla():
    carla_exe = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'CarlaUE4.exe'))

    if not os.path.exists(carla_exe):
        log(f"⚠  CarlaUE4.exe not found at {carla_exe}")
        log("   Please start CARLA manually, then press Enter.")
        input()
        return None

    log(f"🚀 Launching CARLA…")
    log(f"   {carla_exe} -dx11 -quality-level=Medium")
    proc = subprocess.Popen(
        [carla_exe, '-dx11', '-quality-level=Medium'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(carla_exe)
    )
    log(f"   PID {proc.pid} — waiting {CARLA_WAIT_SEC}s for CARLA to load…")
    for i in range(CARLA_WAIT_SEC, 0, -1):
        print(f"\r     {i}s remaining…", end='', flush=True)
        time.sleep(1)
    print()
    return proc

# ─────────────────────────────────────────────
# STEP 2 — CONNECT
# ─────────────────────────────────────────────

def connect(retries=5):
    log(f"🔌 Connecting to CARLA at {HOST}:{PORT}…")
    for attempt in range(1, retries + 1):
        try:
            client = carla.Client(HOST, PORT)
            client.set_timeout(10.0)
            world  = client.get_world()
            tm     = client.get_trafficmanager(TM_PORT)
            tm.set_global_distance_to_leading_vehicle(2.5)
            # Keep async mode (default) — do NOT call set_synchronous_mode here
            log(f"✓  Connected! Map: {world.get_map().name}")
            return client, world, tm
        except Exception as e:
            log(f"   Attempt {attempt}/{retries} failed: {e}")
            time.sleep(5)
    raise RuntimeError("Could not connect to CARLA after multiple attempts.")

# ─────────────────────────────────────────────
# STEP 3 — SPAWN VEHICLES
# ─────────────────────────────────────────────

def spawn_vehicles(world, tm, n=NUM_VEHICLES):
    log(f"🚗 Spawning {n} NPC vehicles…")
    rng       = random.Random(SEED)
    bp_lib    = world.get_blueprint_library()
    veh_bps   = [bp for bp in bp_lib.filter('vehicle.*')
                 if int(bp.get_attribute('number_of_wheels').as_int()) == 4]
    spawns    = world.get_map().get_spawn_points()
    rng.shuffle(spawns)

    vehicles = []
    for sp in spawns[:n + 10]:
        if len(vehicles) >= n:
            break
        bp = rng.choice(veh_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', rng.choice(
                bp.get_attribute('color').recommended_values))
        v = world.try_spawn_actor(bp, sp)
        if v:
            v.set_autopilot(True, tm.get_port())
            vehicles.append(v)

    log(f"✓  Spawned {len(vehicles)}/{n} vehicles")

    # Tick the world a few times so TM registers all vehicles and they start moving
    log("   Warming up Traffic Manager (5 ticks)…")
    for _ in range(5):
        world.wait_for_tick(seconds=5.0)
    log("   ✓ Vehicles should be moving now")
    return vehicles

# ─────────────────────────────────────────────
# STEP 4 — SPAWN WALKERS
# ─────────────────────────────────────────────

def spawn_walkers(world, n=NUM_WALKERS):
    log(f"🚶 Spawning {n} walkers…")
    rng          = random.Random(SEED + 1)
    bp_lib       = world.get_blueprint_library()
    walker_bps   = bp_lib.filter('walker.pedestrian.*')
    ctrl_bp      = bp_lib.find('controller.ai.walker')

    walkers      = []
    controllers  = []

    for _ in range(n * 2):          # try up to 2× to reach target
        if len(walkers) >= n:
            break
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        sp = carla.Transform(loc)
        bp = rng.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        w = world.try_spawn_actor(bp, sp)
        if w is None:
            continue
        c = world.try_spawn_actor(ctrl_bp, carla.Transform(), w)
        if c:
            c.start()
            dest = world.get_random_location_from_navigation()
            if dest:
                c.go_to_location(dest)
            c.set_max_speed(0.8 + rng.random() * 1.8)   # 0.8–2.6 m/s
            controllers.append(c)
        walkers.append(w)

    log(f"✓  Spawned {len(walkers)}/{n} walkers")
    return walkers, controllers

# ─────────────────────────────────────────────
# STEP 5 — APPLY UNCERTAINTY
# ─────────────────────────────────────────────

def apply_uncertainty(world, vehicles, tm):
    log(f"⚡ Applying {SCENARIO_LEVEL.upper()} uncertainty scenario (seed={SEED})…")
    mgr = UncertaintyManager(seed=SEED)
    mgr.load_scenario(SCENARIO_LEVEL)
    mgr.setup(world, ego_vehicle=None, npc_vehicles=vehicles,
              traffic_manager=tm)
    log(f"✓  Uncertainty active!")

    state = mgr.get_state()
    weather = state.get('weather', {})
    if weather:
        log(f"   Weather : rain={weather.get('rain_intensity',0):.0f}%  "
            f"fog={weather.get('fog_intensity',0):.0f}%  "
            f"vis={weather.get('visibility',100):.0f}%")
    log(f"   Modules : {', '.join(k for k,v in state['enabled_modules'].items() if v)}")
    return mgr

# ─────────────────────────────────────────────
# STEP 6 — LIVE STATUS LOOP
# ─────────────────────────────────────────────

def run_loop(world, mgr, vehicles, walkers):
    separator()
    log("📡 Simulation running — press Ctrl+C to stop")
    separator()

    start      = time.time()
    last_print = start
    total_events = 0

    try:
        while True:
            try:
                snap  = world.wait_for_tick(seconds=5.0)
            except Exception:
                time.sleep(0.05)
                continue

            delta    = snap.timestamp.delta_seconds
            sim_time = snap.timestamp.elapsed_seconds
            result   = mgr.tick(delta, sim_time)

            # ── print triggered events immediately ──
            for ev in result.get('triggered_events', []):
                total_events += 1
                print(f"\n  ⚡ EVENT [{sim_time:.1f}s] "
                      f"{ev.event_type.value.upper()}  "
                      f"severity={ev.severity}  dur={ev.duration:.0f}s",
                      flush=True)

            # ── periodic status ──
            now = time.time()
            if now - last_print >= STATUS_EVERY_SEC:
                last_print = now
                elapsed    = now - start
                w_info     = result.get('weather_info') or {}
                comm       = result.get('communication_stats') or {}
                active_ev  = result.get('active_events', [])
                jams       = len(mgr.traffic.get_active_jams())
                accidents  = len(mgr.accident.get_active_events())

                print(f"\n  ┌── {elapsed:>5.0f}s elapsed │ sim {sim_time:.1f}s "
                      + '─' * 22)
                print(f"  │  Weather  rain={w_info.get('rain_intensity',0):4.0f}%  "
                      f"fog={w_info.get('fog_intensity',0):4.0f}%  "
                      f"vis={w_info.get('visibility',100):4.0f}%")
                print(f"  │  Vehicles {len(vehicles)} NPCs  │  "
                      f"Walkers {len(walkers)}")
                print(f"  │  Events   {total_events} total  │  "
                      f"{len(active_ev)} active")
                if jams or accidents:
                    print(f"  │  Hazards  {jams} jams  {accidents} accidents  ⚠")
                if comm:
                    print(f"  │  V2V      sent={comm.get('messages_sent',0)}  "
                          f"lost={comm.get('messages_lost',0)}  "
                          f"outage={'YES ⚠' if comm.get('current_outage') else 'no'}")
                active_names = [e.event_type.value for e in active_ev]
                if active_names:
                    print(f"  │  Active   {', '.join(active_names[:4])}")
                print(f"  └" + '─' * 52, flush=True)

    except KeyboardInterrupt:
        print()
        separator()
        elapsed = time.time() - start
        log(f"🛑 Stopped after {elapsed:.0f}s — {total_events} events triggered")

# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────

def cleanup(mgr, vehicles, walkers, controllers, carla_proc):
    log("🧹 Cleaning up…")
    try:
        mgr.cleanup()
    except Exception:
        pass
    for c in controllers:
        try: c.stop()
        except Exception: pass
    for a in controllers + walkers + vehicles:
        try: a.destroy()
        except Exception: pass
    log("✓  Done.")
    if carla_proc:
        log("   CARLA is still running — close the window manually.")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print()
    separator('═')
    log("CARLA 0.9.9 — Uncertainty Simulation")
    log(f"Config: {NUM_VEHICLES} vehicles  {NUM_WALKERS} walkers  "
        f"scenario={SCENARIO_LEVEL.upper()}  seed={SEED}")
    separator('═')
    print()

    carla_proc  = None
    vehicles    = []
    walkers     = []
    controllers = []
    mgr         = None

    try:
        # 1. Launch CARLA
        carla_proc = launch_carla()
        print()

        # 2. Connect
        client, world, tm = connect()
        print()

        # 3. Spawn vehicles
        vehicles = spawn_vehicles(world, tm)

        # 4. Spawn walkers
        walkers, controllers = spawn_walkers(world)
        print()

        # 5. Apply uncertainty
        mgr = apply_uncertainty(world, vehicles, tm)
        print()

        # 6. Live loop
        run_loop(world, mgr, vehicles, walkers)

    except Exception as e:
        print(f"\n  ✗ Fatal: {e}")
        raise

    finally:
        if mgr:
            cleanup(mgr, vehicles, walkers, controllers, carla_proc)


if __name__ == '__main__':
    main()

