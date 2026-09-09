@echo off
setlocal enabledelayedexpansion
title SMASH FCS CIWS 시뮬레이션 런처

set "WSL_DISTRO=Ubuntu-24.04"

if defined WSL_DISTRO (set "WSL_D=-d %WSL_DISTRO%") else (set "WSL_D=")
set "WSL_CMD=wsl.exe %WSL_D% bash -lc"

set "SIM_DIR=%~dp0"
set "SIM_DIR=%SIM_DIR:~0,-1%"

pushd "%SIM_DIR%\.."
set "PROJECT_ROOT=%CD%"
popd

set "CORE_PATH_WSL="
for /f "usebackq delims=" %%i in (`wsl.exe %WSL_D% wslpath -a "%PROJECT_ROOT%" 2^>nul`) do set "CORE_PATH_WSL=%%i"
if not defined CORE_PATH_WSL set "CORE_PATH_WSL=/mnt/d/Aiming"
set "SETUP_SCRIPT=%CORE_PATH_WSL%/scripts/wsl_setup_smash_fcs.sh"
set "VIEWER_SCRIPT=%PROJECT_ROOT%\scripts\win_scope_viewer.py"
set "HEADLESS=true"

:menu
set "ENVPFX=SMASH_CORE_PATH='%CORE_PATH_WSL%' SMASH_HEADLESS=%HEADLESS%"
cls
echo ============================================================
echo   SMASH FCS Gazebo 시뮬레이터 WSL2 런처
echo ============================================================
echo   프로젝트 루트 (WSL): %CORE_PATH_WSL%
echo   Gazebo GUI 모드: 헤드리스=%HEADLESS%
echo.
echo   1. 환경 진단 (check)
echo   2. 워크스페이스 링크 (link)
echo   3. colcon 빌드 (build)
echo   4. 시뮬레이션 실행 (비사격, 1/2단계)
echo   5. [수동] 시뮬레이션 및 스마트 스코프 뷰어 연동 (수동 격발)
echo   6. [자동] 시뮬레이션 및 스마트 스코프 뷰어 연동 (READY 자동 격발)
echo   7. 3단계 격발 판정 오프라인 검증 (ROS2/Gazebo 불필요)
echo   8. 스마트 스코프 뷰어만 단독 실행 (Windows 네이티브)
echo   9. Gazebo 3D GUI 창 토글 (현재 헤드리스=%HEADLESS%)
echo   0. 종료
echo ============================================================
echo.
set "choice="
set /p choice=번호를 입력하세요 (0~9): 

if not defined choice goto menu
if "%choice%"=="1" goto do_check
if "%choice%"=="2" goto do_link
if "%choice%"=="3" goto do_build
if "%choice%"=="4" goto do_run_plain
if "%choice%"=="5" goto do_run_fire_manual
if "%choice%"=="6" goto do_run_fire_auto
if "%choice%"=="7" goto do_verify
if "%choice%"=="8" goto do_viewer
if "%choice%"=="9" goto do_toggle
if "%choice%"=="0" goto end
echo.
echo   잘못된 입력입니다: %choice%
pause
goto menu

:do_toggle
if "!HEADLESS!"=="true" (set "HEADLESS=false") else (set "HEADLESS=true")
goto menu

:do_check
%WSL_CMD% "%ENVPFX% bash '%SETUP_SCRIPT%' check"
goto after

:do_link
%WSL_CMD% "%ENVPFX% bash '%SETUP_SCRIPT%' link"
goto after

:do_build
%WSL_CMD% "%ENVPFX% bash '%SETUP_SCRIPT%' build"
goto after

:do_run_plain
echo   비사격 모드로 시뮬레이션을 구동합니다.
%WSL_CMD% "%ENVPFX% SMASH_ENABLE_FIRE_CONTROL=false bash '%SETUP_SCRIPT%' run"
goto after

:do_run_fire_manual
echo.
echo   ============================================================
echo   [5번] 격발 제어 시뮬레이션 및 스마트 스코프 뷰어를 구동합니다.
echo   Windows 화면에 스코프 창이 팝업됩니다.
echo   웹 뷰어 주소: http://localhost:9999
echo   ============================================================
echo.
start "" python "%VIEWER_SCRIPT%"
%WSL_CMD% "%ENVPFX% SMASH_ENABLE_FIRE_CONTROL=true SMASH_FIRE_MODE=manual SMASH_LAUNCH_VIEWER=true bash '%SETUP_SCRIPT%' run"
goto after

:do_run_fire_auto
echo.
echo   ============================================================
echo   [6번] 격발 제어 시뮬레이션 및 스마트 스코프 뷰어를 구동합니다.
echo   READY 상태 충족 시 자동으로 격발 신호가 발생합니다.
echo   웹 뷰어 주소: http://localhost:9999
echo   ============================================================
echo.
start "" python "%VIEWER_SCRIPT%"
%WSL_CMD% "%ENVPFX% SMASH_ENABLE_FIRE_CONTROL=true SMASH_FIRE_MODE=auto SMASH_LAUNCH_VIEWER=true bash '%SETUP_SCRIPT%' run"
goto after

:do_verify
%WSL_CMD% "%ENVPFX% bash '%SETUP_SCRIPT%' verify_fire_control"
goto after

:do_viewer
echo   SMASH Windows 네이티브 스코프 뷰어를 단독 실행합니다.
python "%VIEWER_SCRIPT%"
goto after

:after
echo.
pause
goto menu

:end
endlocal
exit /b 0
