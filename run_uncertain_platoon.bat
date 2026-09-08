@echo off
REM ============================================================================
REM   CARLA 0.9.9 - UNCERTAIN PLATOON SIMULATION (UNIFIED LAUNCHER)
REM   Integrates Platoon System + Ghost Drivers + Uncertainty World
REM ============================================================================

setlocal
set "API_DIR=%~dp0WindowsNoEditor\PythonAPI"
set "LOG_DIR=%API_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\uncertain_platoon.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
break > "%LOG_FILE%"
start "CARLA Uncertain Platoon Logs" powershell -NoExit -Command "$host.UI.RawUI.WindowTitle='CARLA Uncertain Platoon Live Logs'; Get-Content -LiteralPath '%LOG_FILE%' -Wait"

cd /d "%API_DIR%"
py -3.7 uncertain_platoon_main.py --platoon-size 3 --log-file "%LOG_FILE%" %*

