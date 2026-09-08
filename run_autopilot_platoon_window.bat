@echo off
REM Standalone CARLA autopilot platoon Pygame window launcher.
REM Usage:
REM   run_autopilot_platoon_window.bat
REM   run_autopilot_platoon_window.bat --town Town01 --vehicles 32 --pedestrians 20
REM   run_autopilot_platoon_window.bat --launch-server

setlocal
set "API_DIR=%~dp0WindowsNoEditor\PythonAPI"
set "LOG_DIR=%API_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\autopilot_platoon_window.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
break > "%LOG_FILE%"
start "CARLA Platoon Logs" powershell -NoExit -Command "$host.UI.RawUI.WindowTitle='CARLA Platoon Live Logs'; Get-Content -LiteralPath '%LOG_FILE%' -Wait"

cd /d "%API_DIR%"
py -3.7 autopilot_platoon_window.py --log-file "%LOG_FILE%" %*
