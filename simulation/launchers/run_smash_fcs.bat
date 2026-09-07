@echo off
setlocal enabledelayedexpansion
title SMASH FCS 대드론 사격 통제 시뮬레이터 런처

set "WSL_DISTRO=Ubuntu-24.04"
set "CORE_PATH_WSL=/mnt/d/Aiming"
set "SETUP_SCRIPT=%CORE_PATH_WSL%/scripts/wsl_setup_smash_fcs.sh"

if defined WSL_DISTRO (
    set "WSL_CMD=wsl.exe -d %WSL_DISTRO% bash -lc"
) else (
    set "WSL_CMD=wsl.exe bash -lc"
)

:menu
cls
echo ============================================================
echo   SMASH FCS (CIWS 포탑 통합) WSL2 실행 런처
echo ============================================================
echo   코어 경로 (WSL 기준): %CORE_PATH_WSL%
echo.
echo   1. 환경 진단만 (check)          : 아무것도 바꾸지 않음
echo   2. 워크스페이스 연결 (link)     : 최초 1회만 필요
echo   3. colcon 빌드 (build)
echo   4. 시뮬레이션 실행 (조준만, 1/2단계)
echo   5. 시뮬레이션 실행 (격발 포함, 수동 트리거, 3단계)
echo   6. 시뮬레이션 실행 (격발 포함, READY 자동격발, 3단계)
echo   7. 3단계 격발/판정 로직 오라클 검증 (ROS2/Gazebo 불필요)
echo   8. 대화형 스코프 뷰어 (마우스 좌클릭/스페이스바 격발 창)
echo   0. 종료
echo ============================================================
echo.
set /p choice=번호를 입력하세요: 

if "%choice%"=="1" goto do_check
if "%choice%"=="2" goto do_link
if "%choice%"=="3" goto do_build
if "%choice%"=="4" goto do_run_plain
if "%choice%"=="5" goto do_run_fire_manual
if "%choice%"=="6" goto do_run_fire_auto
if "%choice%"=="7" goto do_verify
if "%choice%"=="8" goto do_viewer
if "%choice%"=="0" goto end
echo.
echo   잘못된 입력입니다.
pause
goto menu

:do_check
%WSL_CMD% "bash '%SETUP_SCRIPT%' check"
goto after

:do_link
%WSL_CMD% "bash '%SETUP_SCRIPT%' link"
goto after

:do_build
%WSL_CMD% "bash '%SETUP_SCRIPT%' build"
goto after

:do_run_plain
echo   조준만 실행합니다 (격발 없음). rqt_image_view 또는 8번 뷰어로 확인.
%WSL_CMD% "SMASH_ENABLE_FIRE_CONTROL=false bash '%SETUP_SCRIPT%' run"
goto after

:do_run_fire_manual
echo   격발 포함, manual 모드(마우스 클릭 / 스페이스바로 격발)로 실행합니다.
echo   새 창에서 8번(대화형 스코프 뷰어)을 실행하여 마우스 좌클릭/스페이스바로 격발하십시오.
%WSL_CMD% "SMASH_ENABLE_FIRE_CONTROL=true SMASH_FIRE_MODE=manual bash '%SETUP_SCRIPT%' run"
goto after

:do_run_fire_auto
echo   격발 포함, auto 모드(READY 진입 시 자동 격발)로 실행합니다.
%WSL_CMD% "SMASH_ENABLE_FIRE_CONTROL=true SMASH_FIRE_MODE=auto bash '%SETUP_SCRIPT%' run"
goto after

:do_verify
%WSL_CMD% "bash '%SETUP_SCRIPT%' verify_fire_control"
goto after

:do_viewer
echo   SMASH 대화형 스코프 뷰어를 실행합니다. (마우스 좌클릭 또는 스페이스바로 격발)
%WSL_CMD% "bash '%SETUP_SCRIPT%' viewer"
goto after

:after
echo.
pause
goto menu

:end
endlocal
exit /b 0
