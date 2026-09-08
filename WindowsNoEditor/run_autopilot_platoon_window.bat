@echo off
REM Standalone CARLA autopilot platoon Pygame window launcher.
REM Usage:
REM   run_autopilot_platoon_window.bat
REM   run_autopilot_platoon_window.bat --town Town01 --vehicles 32 --pedestrians 20
REM   run_autopilot_platoon_window.bat --launch-server

cd /d "%~dp0WindowsNoEditor\PythonAPI"
py -3.7 autopilot_platoon_window.py %*
