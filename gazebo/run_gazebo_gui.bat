@echo off
REM Launches the aiming_test.sdf world with the Gazebo GUI (via WSLg)
wsl.exe -d Ubuntu-24.04 -- bash -c "cd /mnt/d/Aiming/gazebo && gz sim -r worlds/aiming_test.sdf"
pause
