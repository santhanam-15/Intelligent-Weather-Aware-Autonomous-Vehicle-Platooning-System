@echo off
REM ============================================================================
REM   CARLA 0.9.9 - DEDICATED LIVE MAPPO TACTICAL CONTROL TERMINAL LAUNCHER
REM ============================================================================
title CARLA MAPPO Platoon Tactical Control Terminal
color 0B
setlocal
set "PROJ_ROOT=%~dp0"
if "%PROJ_ROOT:~-1%"=="\" set "PROJ_ROOT=%PROJ_ROOT:~0,-1%"
cd /d "%PROJ_ROOT%"
set "API_DIR=%PROJ_ROOT%\WindowsNoEditor\PythonAPI"
set "LOG_DIR=%API_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo ============================================================================
echo   STARTING DEDICATED LIVE MAPPO TACTICAL CONTROL TERMINAL
echo   Streaming Actions, Headway Gaps, Speed Catch-Up, and Leader Failovers
echo ============================================================================
echo.

py -3.7 WindowsNoEditor\PythonAPI\live_mappo_terminal.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] MAPPO Terminal exited with code %ERRORLEVEL%.
    pause
)

