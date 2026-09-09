@echo off
title MAPPO Telemetry Dashboard
echo Waiting for MAPPO telemetry...
:loop
cls
if exist "WindowsNoEditor\PythonAPI\mappo_telemetry.log" (
    type "WindowsNoEditor\PythonAPI\mappo_telemetry.log"
) else (
    echo MAPPO Telemetry file not created yet...
)
timeout /t 1 >nul
goto loop

