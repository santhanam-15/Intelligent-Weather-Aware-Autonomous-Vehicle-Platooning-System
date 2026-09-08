import carla
import sys

print("[1/4] Attempting to connect to 127.0.0.1:2000...")
try:
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    print("[2/4] RPC Server connected!")
    
    print("[3/4] Requesting World State on Port 2001...")
    world = client.get_world()
    print(f"[4/4] SUCCESS! Connected to map: {world.get_map().name}")

except Exception as e:
    print(f"\n[ERROR FAILED AT STEP]: {e}")
    sys.exit(1)