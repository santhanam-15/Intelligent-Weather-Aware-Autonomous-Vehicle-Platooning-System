@echo off
REM ============================================================================
REM   CARLA 0.9.9 - LIVE MAPPO TRAINING TERMINAL (DEDICATED MONITOR & RUNNER)
REM ============================================================================
title MAPPO Live Training Terminal - Intelligent Vehicle Platooning
color 0B
setlocal

cd /d "%~dp0"
set "API_DIR=%~dp0WindowsNoEditor\PythonAPI"
set "LOG_DIR=%API_DIR%\logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo ============================================================================
echo   STARTING LIVE MAPPO TRAINING TERMINAL
echo   CARLA 0.9.9  ^|  Python 3.7.9  ^|  PyTorch 1.13.1
echo ============================================================================
echo.

py -3.7 WindowsNoEditor\PythonAPI\live_training_terminal.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training terminal exited with code %ERRORLEVEL%.
    pause
)

