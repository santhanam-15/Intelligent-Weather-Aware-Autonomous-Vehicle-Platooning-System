@echo off
REM ============================================================================
REM   CARLA 0.9.9 - 1-CLICK LIVING REAL-WORLD UNCERTAINTY SIMULATION
REM   Ghost Driver Behavioral Model - Fully Working Launcher
REM ============================================================================

title CARLA 0.9.9 - Ghost Driver Simulation
color 0B

SET CARLA_ROOT=%~dp0
if "%CARLA_ROOT:~-1%"=="\" SET CARLA_ROOT=%CARLA_ROOT:~0,-1%

IF "%~1"=="" (SET TARGET_TOWN=auto) ELSE (SET TARGET_TOWN=%~1)

SET VENV_PYTHON=%CARLA_ROOT%\Scripts\python.exe
SET SIM_SCRIPT=%CARLA_ROOT%\PythonAPI\real_world_simulation.py
SET EGG_PATH=%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.9-py3.7-win-amd64.egg
SET PYTHONPATH=%CARLA_ROOT%\PythonAPI;%CARLA_ROOT%\PythonAPI\carla;%CARLA_ROOT%\PythonAPI\carla\agents;%EGG_PATH%;%PYTHONPATH%

echo.
echo  ============================================================================
echo        CARLA 0.9.9  -  GHOST DRIVER SIMULATION (DAVI Model)
echo  ============================================================================
echo.

IF NOT EXIST "%VENV_PYTHON%" (
    echo  [ERROR] Python 3.7 venv not found: %VENV_PYTHON%
    pause
    exit /b 1
)

IF NOT EXIST "%SIM_SCRIPT%" (
    echo  [ERROR] Script not found: %SIM_SCRIPT%
    pause
    exit /b 1
)

REM Check if CARLA already running
echo  [1/3] Checking CARLA on 127.0.0.1:2000 ...
"%VENV_PYTHON%" -c "import socket,sys;s=socket.socket();s.settimeout(1.5);r=s.connect_ex(('127.0.0.1',2000));s.close();sys.exit(0 if r==0 else 1)"
IF %ERRORLEVEL% EQU 0 (
    echo        [OK] CARLA already running!
    goto :LAUNCH_SIM
)

REM Auto-launch CARLA
SET CARLA_EXE=%CARLA_ROOT%\CarlaUE4.exe
IF NOT EXIST "%CARLA_EXE%" (
    echo  [ERROR] CarlaUE4.exe not found at: %CARLA_EXE%
    pause
    exit /b 1
)

echo        [AUTO-LAUNCH] Starting CarlaUE4.exe (Low Quality, RTX 3050 mode)...
start "" "%CARLA_EXE%" -quality-level=Low -windowed -ResX=800 -ResY=600 -dx11 -carla-rpc-port=2000

echo  [2/3] Waiting for CARLA to start (up to 90s) ...
SET /A ATTEMPTS=0
:WAIT_LOOP
ping 127.0.0.1 -n 4 >nul
SET /A ATTEMPTS+=1
"%VENV_PYTHON%" -c "import socket,sys;s=socket.socket();s.settimeout(1.5);r=s.connect_ex(('127.0.0.1',2000));s.close();sys.exit(0 if r==0 else 1)"
IF %ERRORLEVEL% EQU 0 (
    echo        [OK] CARLA is READY! Starting simulation in 3s...
    ping 127.0.0.1 -n 4 >nul
    goto :LAUNCH_SIM
)
IF %ATTEMPTS% GEQ 25 (
    echo        [INFO] Timeout - Python will retry internally...
    goto :LAUNCH_SIM
)
echo        ... still starting (%ATTEMPTS%/25) ...
goto :WAIT_LOOP

:LAUNCH_SIM
echo.
echo  [3/3] Launching Ghost Driver Simulation...
echo        Map: %TARGET_TOWN% / 30 NPC Vehicles / Ghost 50m bubble
echo  ============================================================================
echo.
"%VENV_PYTHON%" "%SIM_SCRIPT%" --town %TARGET_TOWN% --vehicles 30

echo.
echo  Simulation ended. Press any key to close.
pause
