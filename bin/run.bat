@echo off
REM ============================================================
REM  CARLA 0.9.9 Uncertainty Toolkit — Launcher
REM  Run this file to start the interactive simulation menu.
REM ============================================================

SET VENV_PYTHON=%~dp0..\Scripts\python.exe
SET SCRIPT=%~dp0start.py

echo.
echo   CARLA 0.9.9 - Uncertainty Toolkit
echo   ===================================

REM Check if venv Python exists
IF NOT EXIST "%VENV_PYTHON%" (
    echo   [ERROR] Could not find venv Python at:
    echo           %VENV_PYTHON%
    echo   Please make sure the virtualenv is set up correctly.
    pause
    exit /b 1
)

echo   Using Python: %VENV_PYTHON%
echo   Starting start.py...
echo.

"%VENV_PYTHON%" "%SCRIPT%"

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [ERROR] Script exited with error code %ERRORLEVEL%
    pause
)

