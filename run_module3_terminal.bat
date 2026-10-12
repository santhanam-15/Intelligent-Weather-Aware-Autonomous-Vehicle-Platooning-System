@echo off
REM ============================================================================
REM   CARLA 0.9.9 - MODULE 3 TWIN-WORLD LIVE DECISION TERMINAL LAUNCHER
REM ============================================================================
title Module 3 Live Decision Terminal - Rear-Impact Evasion & Platoon Rejoining
color 0D
setlocal
set "PROJ_ROOT=%~dp0"
if "%PROJ_ROOT:~-1%"=="\" set "PROJ_ROOT=%PROJ_ROOT:~0,-1%"
cd /d "%PROJ_ROOT%"
set "API_DIR=%PROJ_ROOT%\WindowsNoEditor\PythonAPI"
set "LOG_DIR=%API_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo ============================================================================
echo   MODULE 3 TWIN-WORLD LIVE DECISION TERMINAL
echo   Streaming Real-Time Rear-Threat Projections, Evasion & Rejoining Records
echo ============================================================================
echo.

py -3.7 WindowsNoEditor\PythonAPI\live_module3_terminal.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Module 3 Live Decision Terminal exited with code %ERRORLEVEL%.
    pause
)

