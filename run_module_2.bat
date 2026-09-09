@echo off
REM ============================================================================
REM   CARLA 0.9.9 - MODULE 2 MAPPO PLATOON SIMULATION (UNIFIED LAUNCHER)
REM ============================================================================

setlocal
set "API_DIR=%~dp0WindowsNoEditor\PythonAPI"
set "LOG_DIR=%API_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\module2_platoon.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
break > "%LOG_FILE%"
start "CARLA Module 2 Logs" powershell -NoExit -Command "$host.UI.RawUI.WindowTitle='CARLA Module 2 Live Logs'; Get-Content -LiteralPath '%LOG_FILE%' -Wait"
start "MAPPO Telemetry" cmd /c "%~dp0show_mappo_telemetry.bat"

cd /d "%API_DIR%"
py -3.7 run_module_2.py --log-file "%LOG_FILE%" %*

