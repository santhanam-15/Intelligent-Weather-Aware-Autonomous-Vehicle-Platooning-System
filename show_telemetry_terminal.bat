@echo off
REM ============================================================================
REM Root Launcher: Open Separate Live Telemetry Stream Terminal
REM ============================================================================

setlocal
set "MODULE2_DIR=%~dp0WindowsNoEditor\module_2"
call "%MODULE2_DIR%\launch_telemetry_terminal.bat"

