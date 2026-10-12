@echo off
REM ============================================================================
REM   CARLA 0.9.9 - SIMULATION REAL-TIME LIVE LOG TERMINAL
REM ============================================================================
title CARLA Simulation Real-Time Live Logs
color 0F
setlocal

cd /d "%~dp0"
set "API_DIR=%~dp0WindowsNoEditor\PythonAPI"

py -3.7 WindowsNoEditor\PythonAPI\live_log_monitor.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Log monitor exited with code %ERRORLEVEL%.
    pause
)

