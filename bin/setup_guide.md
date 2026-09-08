# CARLA 0.9.9 Setup Guide

This guide covers the setup, launching, and verification of CARLA 0.9.9 for the Weather-Aware Platooning project.

## Prerequisites
- Python 3.7 (Required for CARLA 0.9.9 Python API compatibility)
- Packages: `pygame`, `numpy`

## Python Virtual Environment Setup

1. **Create a virtual environment (Python 3.7):**
   ```cmd
   python -m venv carla_env
   ```

2. **Activate the environment:**
   ```cmd
   carla_env\Scripts\activate
   ```

3. **Install required dependencies:**
   ```cmd
   pip install pygame numpy
   ```

## Launching CARLA

To run CARLA smoothly on hardware with 4GB VRAM (like an RTX 3050), use the provided batch script `launch_carla_lowend.bat`.

It launches CARLA with the following optimized parameters:
- `-quality-level=Low`: Reduces graphical fidelity to save VRAM and processing power.
- `-windowed`: Runs the simulation in a window.
- `-ResX=800 -ResY=600`: Sets the resolution to 800x600.
- `-carla-rpc-port=2000`: Ensures the server listens on the default port.

To start, simply double-click `launch_carla_lowend.bat` or run it from the command prompt.

## Verifying the Setup

After launching CARLA and waiting for the map to load, verify your setup by running the verification script:

```cmd
cd PythonAPI
python verify_carla.py
```

This script will connect to the simulator, set up synchronous mode, spawn a Tesla Model 3, tick the simulation for 10 frames, and clean up.

## Troubleshooting

- **Python Version Mismatch (No module named carla):**
  Ensure you are using a Python 3.7 environment. CARLA 0.9.9 provides the egg `carla-0.9.9-py3.7-win-amd64.egg`. Using Python 3.8+ will result in import errors.

- **Connection Refused:**
  Ensure CARLA is running and has fully loaded the map before running scripts. Check if a firewall is blocking port 2000.

- **Out of Video Memory (VRAM):**
  If CARLA crashes due to VRAM limits, make sure you are using the `launch_carla_lowend.bat` script. Avoid spawning more than 10-15 vehicles if you have 4GB VRAM.

## Hardware Limits
- **Max Vehicles:** Keep the number of spawned actors (vehicles + pedestrians) below 20 to maintain stable performance in synchronous mode.
- **Resolution:** Keep resolution at 800x600 for the best framerates, especially when running logic-heavy scripts alongside the simulator.
