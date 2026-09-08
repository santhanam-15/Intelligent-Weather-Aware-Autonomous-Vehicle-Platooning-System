    @echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM   CARLA 0.9.9 — Real-World Uncertainty Simulation Launcher
REM   Uses scenario_runner-0.9.9 to test safety-critical driving situations:
REM     * Emergency Braking & Obstacle Dodging
REM     * Aggressive Lane Cut-Ins
REM     * Pedestrian / Jaywalker Hazards
REM     * Slippery Road / Hydroplaning (Friction Uncertainty)
REM     * Red-Light Violations & Wrong-Way Overtakers
REM     * Unsignalized Blind Intersections
REM ============================================================================

title CARLA 0.9.9 - Real-World Uncertainty Testbench

set CARLA_ROOT=%~dp0
REM Remove trailing backslash if present
if "%CARLA_ROOT:~-1%"=="\" set CARLA_ROOT=%CARLA_ROOT:~0,-1%

set VENV_PYTHON=%CARLA_ROOT%\Scripts\python.exe
set SCENARIO_DIR=%CARLA_ROOT%\scenario_runner-0.9.9
set EGG_FILE=%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.9-py3.7-win-amd64.egg
set CARLA_API=%CARLA_ROOT%\PythonAPI\carla
set CARLA_AGENTS=%CARLA_ROOT%\PythonAPI\carla\agents

REM Check Python Virtual Environment
if not exist "%VENV_PYTHON%" (
    echo.
    echo  [ERROR] Python 3.7 virtual environment was not found at:
    echo          "%VENV_PYTHON%"
    echo.
    echo  Please ensure your Python 3.7 venv is installed in the Scripts directory.
    echo.
    pause
    exit /b 1
)

REM Check ScenarioRunner Directory
if not exist "%SCENARIO_DIR%\scenario_runner.py" (
    echo.
    echo  [ERROR] scenario_runner.py was not found at:
    echo          "%SCENARIO_DIR%"
    echo.
    pause
    exit /b 1
)

REM Set Environment Variables
set PYTHONPATH=%CARLA_API%;%CARLA_AGENTS%;%EGG_FILE%;%SCENARIO_DIR%;%PYTHONPATH%

echo.
echo  ============================================================================
echo        CARLA 0.9.9  -  REAL-WORLD UNCERTAINTY RUNNER (ScenarioRunner)
echo  ============================================================================
echo.

REM Check if CARLA UE4 Simulator is running
tasklist /FI "IMAGENAME eq CarlaUE4.exe" 2>NUL | find /I /N "CarlaUE4.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    echo  [OK] CARLA Simulator is currently running.
) else (
    echo  [!] CARLA Simulator (CarlaUE4.exe) is NOT running.
    echo.
    echo      [1] Start CARLA with Low-End Settings (800x600, Low Quality, RTX 3050 safe)
    echo      [2] I will start CARLA manually
    echo      [3] Exit
    echo.
    set /p CARLA_CHOICE="  Select option [1]: "
    if "!CARLA_CHOICE!"=="" set CARLA_CHOICE=1

    if "!CARLA_CHOICE!"=="1" (
        echo.
        echo  [INFO] Launching CarlaUE4.exe with low-end GPU settings...
        start "CARLA Simulator" "%CARLA_ROOT%\CarlaUE4.exe" -quality-level=Low -windowed -ResX=800 -ResY=600 -carla-rpc-port=2000
        echo  [INFO] Waiting 15 seconds for CARLA to load...
        timeout /t 15 /nobreak >nul 2>&1 || ping 127.0.0.1 -n 16 >nul
        echo  [OK] CARLA should now be initialized.
    ) else if "!CARLA_CHOICE!"=="3" (
        exit /b 0
    )
)

echo.
echo  [INFO] Launching Interactive Real-World Uncertainty Testbench...
echo.

cd /d "%SCENARIO_DIR%"
"%VENV_PYTHON%" "%SCENARIO_DIR%\run_uncertainty.py"

echo.
echo  ============================================================================
echo    Simulation session ended.
echo  ============================================================================
echo.
pause
