@echo off
REM ============================================================================
REM   CARLA 0.9.9 - MODULE 2 MAPPO PLATOON SIMULATION (UNIFIED LAUNCHER)
REM ============================================================================
REM Launches:
REM   1. Dedicated MAPPO Tactical Control Console
REM   2. Dedicated Module 3 Twin-World Safety Console
REM   3. CARLA Simulation Process & Pygame Telemetry Window
REM ============================================================================

setlocal
set "PROJ_ROOT=%~dp0"
if "%PROJ_ROOT:~-1%"=="\" set "PROJ_ROOT=%PROJ_ROOT:~0,-1%"
set "API_DIR=%PROJ_ROOT%\WindowsNoEditor\PythonAPI"
set "LOG_DIR=%API_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\uncertain_platoon.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo ============================================================================
echo   LAUNCHING MULTI-CONSOLE ARCHITECTURE (3 WINDOWS)
echo   1. CARLA Simulation Process Window
echo   2. Dedicated MAPPO Tactical Control Console
echo   3. Dedicated Module 3 Twin-World Safety Console
echo ============================================================================
echo.

set "CARLA_MULTI_CONSOLE_ACTIVE=1"

start "CARLA MAPPO Platoon Tactical Control Terminal" cmd /K "title CARLA MAPPO Platoon Tactical Control Terminal && cd /d "%PROJ_ROOT%" && py -3.7 WindowsNoEditor\PythonAPI\live_mappo_terminal.py"

start "CARLA Module 3 Twin-World Safety Terminal" cmd /K "title CARLA Module 3 Twin-World Safety Terminal && cd /d "%PROJ_ROOT%" && py -3.7 WindowsNoEditor\PythonAPI\live_module3_terminal.py"

cd /d "%API_DIR%"
py -3.7 run_module_2.py --log-file "%LOG_FILE%" %*
