@echo off
REM ============================================================
REM  Activates the CARLA Python 3.7 venv and opens a terminal.
REM  Run this ONCE per session, then use "python" normally.
REM
REM  Or use it directly:
REM    carla_python.bat manual_control.py
REM    carla_python.bat spawn_npc.py -n 30
REM ============================================================

SET VENV=d:\I FILES\Studies\sem7\MiniProj\WindowsNoEditor
SET VENV_PYTHON=%VENV%\Scripts\python.exe
SET VENV_ACTIVATE=%VENV%\Scripts\activate.bat

IF NOT EXIST "%VENV_PYTHON%" (
    echo [ERROR] Venv Python not found at: %VENV_PYTHON%
    pause
    exit /b 1
)

REM If a script was passed, run it directly
IF NOT "%~1"=="" (
    "%VENV_PYTHON%" %*
    exit /b %ERRORLEVEL%
)

REM Otherwise open an interactive shell with venv activated
echo.
echo  CARLA Python 3.7 environment
echo  Type 'python manual_control.py' etc.
echo  Type 'exit' to close.
echo.
CALL "%VENV_ACTIVATE%"
cmd /k "cd /d d:\I FILES\Studies\sem7\MiniProj\WindowsNoEditor\PythonAPI\examples"

