@echo off
REM ============================================================
REM  Run any CARLA example script with the correct Python 3.7
REM
REM  Usage (drag any .py onto this bat, or from terminal):
REM    run_example.bat manual_control.py
REM    run_example.bat spawn_npc.py -n 50
REM    run_example.bat dynamic_weather.py
REM ============================================================

SET VENV_PYTHON=%~dp0..\..\Scripts\python.exe

IF "%~1"=="" (
    echo.
    echo  Usage: run_example.bat ^<script.py^> [args...]
    echo.
    echo  Common examples:
    echo    run_example.bat manual_control.py
    echo    run_example.bat spawn_npc.py -n 50
    echo    run_example.bat dynamic_weather.py
    echo    run_example.bat synchronous_mode.py
    echo.
    pause
    exit /b 0
)

IF NOT EXIST "%VENV_PYTHON%" (
    echo  [ERROR] Python 3.7 venv not found at: %VENV_PYTHON%
    pause
    exit /b 1
)

echo  Using Python: %VENV_PYTHON%
echo  Running: %*
echo.
"%VENV_PYTHON%" %*

