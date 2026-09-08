@echo off
REM ============================================================================
REM   CARLA MODULE 2 - MAPPO DRL TRAINING MONITOR
REM ============================================================================

set "LOG_FILE=%~dp0WindowsNoEditor\PythonAPI\logs\mappo_training.log"

if not exist "%LOG_FILE%" (
    echo [INFO] Training log file not found. It will be created once MAPPO finishes its first 256-tick rollout...
    echo Waiting for %LOG_FILE%...
    :wait_loop
    timeout /t 2 /nobreak >nul
    if not exist "%LOG_FILE%" goto wait_loop
)

echo [INFO] Found MAPPO training log. Tailing live metrics...
echo ============================================================================
powershell -NoExit -Command "$host.UI.RawUI.WindowTitle='MAPPO Live Training Metrics'; Get-Content -LiteralPath '%LOG_FILE%' -Wait -Tail 10"

