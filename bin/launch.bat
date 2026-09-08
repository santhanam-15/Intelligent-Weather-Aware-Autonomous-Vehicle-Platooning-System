@echo off
REM ============================================================
REM  One-click CARLA launcher:
REM    - Starts CARLA with -dx11 -medium
REM    - Spawns 30 vehicles + 50 walkers
REM    - Applies MEDIUM uncertainty scenario
REM ============================================================

SET VENV_PYTHON=%~dp0..\Scripts\python.exe
SET SCRIPT=%~dp0run_simulation.py

echo.
echo  Starting CARLA Uncertainty Simulation...
echo  ==========================================
echo.

IF NOT EXIST "%VENV_PYTHON%" (
    echo  [ERROR] Python 3.7 venv not found at:
    echo          %VENV_PYTHON%
    pause
    exit /b 1
)

"%VENV_PYTHON%" "%SCRIPT%"

pause

