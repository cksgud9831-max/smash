@echo off
REM Run this in a SECOND window, only after run_gazebo_gui.bat's Gazebo
REM window is already open. Drives the platform (shaking) and target
REM (moving) forever so you can watch it live in the GUI window.
wsl.exe -d Ubuntu-24.04 -- bash -c "cd /mnt/c/Users/cksgu/Desktop/Aiming/gazebo/scripts && python3 00_live_demo_drive.py"
pause
