@echo off
REM Drives the platform and target in Gazebo
wsl.exe -d Ubuntu-24.04 -- bash -c "cd /mnt/d/Aiming/gazebo/scripts && python3 00_live_demo_drive.py"
pause
