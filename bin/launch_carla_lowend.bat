@echo off
echo ========================================================
echo Launching CARLA 0.9.9 with Low Quality Settings
echo Optimized for 4GB VRAM (e.g., RTX 3050)
echo ========================================================
echo Starting CarlaUE4.exe...
"d:\I FILES\Studies\sem7\MiniProj\WindowsNoEditor\CarlaUE4.exe" -quality-level=Medium -windowed -ResX=800 -ResY=600 -carla-rpc-port=2000 Town05
echo CARLA has been launched.
pause
