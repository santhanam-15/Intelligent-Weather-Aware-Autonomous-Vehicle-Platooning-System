@echo off
REM ============================================================================
REM Root Launcher: Open Visual Pygame Platoon Display & Live Telemetry Stream
REM ============================================================================

setlocal
set "MODULE2_DIR=%~dp0WindowsNoEditor\module_2"
cd /d "%MODULE2_DIR%"

echo.
echo ============================================================================
echo   MODULE 2: OPENING PLATOON VISUALIZER AND LIVE TELEMETRY STREAM
echo ============================================================================
echo   1. Pygame Window (1280x720): Displays all platoon cars in real-time
echo   2. Separate Terminal Window: Streams every single data field live
echo ============================================================================
echo.

call "%MODULE2_DIR%\view_platoon_cars.bat" %*
