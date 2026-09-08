import socket
import json
import time

def send_command(cmd_type, params=None):
    # CARLA's default RPC port is 2000
    host = "127.0.0.1"
    port = 2000
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((host, port))
        
        # Format the basic raw instruction payload
        payload = {"command": cmd_type, "params": params or {}}
        s.sendall(json.dumps(payload).encode('utf-8'))
        
        response = s.recv(4096).decode('utf-8')
        s.close()
        return response
    except Exception as e:
        return f"Connection Error: {e}"

def main():
    print("Connecting directly to CARLA engine server...")
    
    # Send a request to query the map assets
    map_info = send_command("get_map_info")
    print(f"Server response: {map_info}")
    
    # Request a vehicle spawn on an open intersection coordinate
    spawn_params = {
        "blueprint": "vehicle.tesla.model3",
        "location": {"x": 0.0, "y": 0.0, "z": 2.0}
    }
    
    result = send_command("spawn_actor", spawn_params)
    print(f"Spawn command status: {result}")

if __name__ == "__main__":
    main()
