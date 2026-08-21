@echo off
REM Launches the aiming_test.sdf world with the Gazebo GUI (via WSLg) so you
REM can watch it directly. Leave this window open. Then, in a SECOND window,
REM run run_live_demo_drive.bat to actually make the platform/target move --
REM this window alone just shows the static starting scene.
wsl.exe -d Ubuntu-24.04 -- bash -c "cd /mnt/c/Users/cksgu/Desktop/Aiming/gazebo && gz sim -r worlds/aiming_test.sdf"
pause
