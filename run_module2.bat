@echo off
REM ============================================================================
REM Root Launcher for Module 2: Weather-Aware MAPPO Cooperative Decision Core
REM ============================================================================

setlocal
set "MODULE2_DIR=%~dp0WindowsNoEditor\module_2"
set "SCRIPT=%MODULE2_DIR%\run_module2.py"

cd /d "%MODULE2_DIR%"

echo.
echo ============================================================================
echo   MODULE 2: WEATHER-AWARE MAPPO MULTI-AGENT PLATOONING SYSTEM
echo   Centralized Training and Decentralized Execution (CTDE)
echo ============================================================================
echo   Entry Point: %SCRIPT%
echo.
echo   Options:
echo     run_module2.bat --mode test
echo     run_module2.bat --mode evaluate
echo     run_module2.bat --mode train --steps 1000
echo     run_module2.bat --mode simulate --town Town01 --friction 0.8
echo ============================================================================
echo.

py -3.7 "%SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Module 2 failed with exit code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%

