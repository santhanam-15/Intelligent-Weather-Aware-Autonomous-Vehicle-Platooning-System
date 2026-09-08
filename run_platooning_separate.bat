@echo off
REM Standalone CARLA platooning launcher using the separate uncertain-world platooning file.
REM Usage:
REM   run_platooning_separate.bat
REM   run_platooning_separate.bat --town Town01 --vehicles 32 --pedestrians 20
REM   run_platooning_separate.bat --town Town05 --platoon-size 5 --platoon-speed 35
REM   run_platooning_separate.bat --windowed --width 1280 --height 720
REM   run_platooning_separate.bat --no-ghost
REM   run_platooning_separate.bat --help

setlocal
set "API_DIR=%~dp0WindowsNoEditor\PythonAPI"
set "LOG_DIR=%API_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\platooning_separate.log"
set "SIM_SCRIPT=%API_DIR%\platooning_separate.py"

if /I "%~1"=="--help" goto :RUN_WITHOUT_LOG_WINDOW
if /I "%~1"=="-h" goto :RUN_WITHOUT_LOG_WINDOW

echo.
echo  ============================================================================
echo       CARLA 0.9.9 - SEPARATE PLATOONING + UNCERTAIN WORLD
echo       Ghost Driver vehicle uncertainty enabled by default
echo  ============================================================================
echo.
echo  Script:
echo    %SIM_SCRIPT%
echo.
echo  Log file:
echo    %LOG_FILE%
echo.
echo  Runtime:
echo    Default Pygame window: fullscreen
echo    Use --windowed --width 1280 --height 720 for a normal window
echo    Auto-launch CARLA if server is not running
echo    Command-line arguments are forwarded to platooning_separate.py
echo.
echo  Examples:
echo    run_platooning_separate.bat --town Town01 --vehicles 32 --pedestrians 20
echo    run_platooning_separate.bat --windowed --width 1280 --height 720
echo    run_platooning_separate.bat --ghost-min-cooldown 1 --ghost-max-cooldown 20
echo.
echo  ============================================================================
echo.

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
break > "%LOG_FILE%"
start "CARLA Separate Platooning Logs" powershell -NoExit -Command "$host.UI.RawUI.WindowTitle='CARLA Separate Platooning Live Logs'; Get-Content -LiteralPath '%LOG_FILE%' -Wait"

cd /d "%API_DIR%"
py -3.7 "%SIM_SCRIPT%" --launch-server --log-file "%LOG_FILE%" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Platooning launcher failed with exit code %EXIT_CODE%.
    echo Check the log file:
    echo %LOG_FILE%
    pause
)
exit /b %EXIT_CODE%

:RUN_WITHOUT_LOG_WINDOW
cd /d "%API_DIR%"
py -3.7 "%SIM_SCRIPT%" %*
exit /b %ERRORLEVEL%
