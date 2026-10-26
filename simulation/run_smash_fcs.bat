@echo off
@chcp 65001 >nul
rem 주의: UTF-8(chcp 65001)에서 cmd 는 한글 등 비ASCII 문자로 끝나는 줄을 읽은 뒤 위치를 잘못 계산해 (cmd 버그)
rem 다음 줄 중간부터 실행한다. 모든 줄은 ASCII 문자로 끝내고 줄바꿈은 CRLF 로 유지한다 (tests/test_launcher_bat.py).
setlocal enabledelayedexpansion
title SMASH FCS CIWS 시뮬레이션 런처 (WSL2)

set "WSL_DISTRO=Ubuntu-24.04"

if defined WSL_DISTRO (set "WSL_D=-d %WSL_DISTRO%") else (set "WSL_D=")
set "WSL_CMD=wsl.exe %WSL_D% bash -lc"

set "SIM_DIR=%~dp0"
set "SIM_DIR=%SIM_DIR:~0,-1%"

pushd "%SIM_DIR%\.."
set "PROJECT_ROOT=%CD%"
popd

set "CORE_PATH_WSL="
for /f "usebackq delims=" %%i in (`wsl.exe %WSL_D% wslpath -a "%PROJECT_ROOT%" 2^>nul ^<nul`) do set "CORE_PATH_WSL=%%i"
if not defined CORE_PATH_WSL set "CORE_PATH_WSL=/mnt/d/Aiming"
set "SETUP_SCRIPT=%CORE_PATH_WSL%/scripts/wsl_setup_smash_fcs.sh"
set "VIEWER_SCRIPT=%PROJECT_ROOT%\scripts\win_scope_viewer.py"
set "PLAYER_SCRIPT=%PROJECT_ROOT%\scripts\recording_player.py"
set "LAST_RESULT=Ready"
set "HEADLESS=true"
set "TARGET_MOTION=none"
set "TARGET_KMH=30"
set "TARGET_NAME=8자 기동"
set "TARGET_HALF_LENGTH=3.0"
set "CAMERA_HFOV=0.4"
rem Turret follow limits (rad). Tested 2026-10-18 at 10 km/h and 0.12 rad: 0.03 / 0.5 followed the target.
rem Override before launching, e.g.  set AIM_JUMP=0.05  (script defaults without this launcher: 0.012 / 0.08)
if not defined AIM_JUMP set "AIM_JUMP=0.03"
if not defined AIM_ENVELOPE set "AIM_ENVELOPE=0.5"
rem Operator emulation: the turret follows the target ground-truth position (pointing only). Override: set OPERATOR_FOLLOW=false
if not defined OPERATOR_FOLLOW set "OPERATOR_FOLLOW=true"
rem FCS measurement-lag term (s). With operator emulation the target is kept centered, so the yaml value 0.17 over-leads
rem (2026-10-18, 10 km/h: 1%% hits at 0.17 vs 92%% at 0.02). Override: set MOTION_LAG_S=0.17 (yaml and code defaults are unchanged)
if not defined MOTION_LAG_S if /i "%OPERATOR_FOLLOW%"=="true" set "MOTION_LAG_S=0.02"
if not defined MOTION_LAG_S set "MOTION_LAG_S=0.17"
rem Fire trigger. manual = one shot per mouse click or Space when READY; auto = READY fires by itself. Override: set FIRE_TRIGGER=auto
if not defined FIRE_TRIGGER set "FIRE_TRIGGER=manual"
rem Kill effect: a HIT destroys the target (it falls, then a new one appears). On by default in manual mode, off in auto mode. Override: set DEMO_KILL=false
if not defined DEMO_KILL if /i "%FIRE_TRIGGER%"=="manual" set "DEMO_KILL=true"
if not defined DEMO_KILL set "DEMO_KILL=false"
rem READY gate: max disagreement (m/s) between the long-window and recent-6-frame velocity. Figure-8 (maneuver) turns constantly, so
rem the default 0.75 blocks READY ~60%% of the time; 1.5 doubled READY but cut the hit rate from ~77%% to ~68%% (2026-10-22, simulation only).
rem Other motions keep 0.75. Override for every motion before launching, e.g.  set VEL_DISAGREE_MPS=0.75  (yaml and code defaults are unchanged)
set "VEL_DISAGREE_USER=%VEL_DISAGREE_MPS%"
set "CAMERA_FOV_LABEL=광각 0.4 rad - 22.9도, 탐색용"

:menu
call :set_vel_gate
set "ENVPFX=SMASH_CORE_PATH='%CORE_PATH_WSL%' SMASH_HEADLESS=%HEADLESS% SMASH_TARGET_MOTION=%TARGET_MOTION% SMASH_TARGET_SPEED_KMH=%TARGET_KMH% SMASH_TARGET_HALF_LENGTH=%TARGET_HALF_LENGTH% SMASH_CAMERA_HFOV=%CAMERA_HFOV% SMASH_AIM_MAX_GOAL_JUMP=%AIM_JUMP% SMASH_AIM_ACQUIRE_ENVELOPE=%AIM_ENVELOPE% SMASH_OPERATOR_FOLLOW=%OPERATOR_FOLLOW% SMASH_MOTION_LAG_S=%MOTION_LAG_S% SMASH_FIRE_TRIGGER=%FIRE_TRIGGER% SMASH_DEMO_KILL=%DEMO_KILL% SMASH_MOTION_VEL_DISAGREE_MPS=%VEL_GATE%"
if "%TARGET_MOTION%"=="none" (set "TARGET_LABEL=정지 호버링, 10 m - 실행 중 뷰어 0~4 키로 기동 변경") else (set "TARGET_LABEL=%TARGET_NAME% %TARGET_KMH% km/h, 35 m, 표적을 잡으면 출발")
cls
echo ============================================================
echo   SMASH FCS Gazebo 시뮬레이터 런처 (WSL2)
echo ============================================================
echo   프로젝트 루트 (WSL): %CORE_PATH_WSL%
echo   Gazebo GUI 모드: 헤드리스=%HEADLESS%
echo   표적: %TARGET_LABEL%
echo   카메라 화각: %CAMERA_FOV_LABEL%
echo.
echo   마지막 작업: !LAST_RESULT!
echo.
echo   [환경 준비]
echo   1. 환경 진단 (check)
echo   2. 워크스페이스 링크 (link)
echo   3. colcon 빌드 (build)
echo.
echo   [실행 및 재생]
echo   5. 녹화 재생기 (로컬 파일, 시뮬레이터 불필요)
echo   4. [비사격] 시뮬레이션 + 스코프 뷰어 (탐지·조준만, 격발 없음)
echo   6. [자동 격발] 시뮬레이션 + 스코프 뷰어 (리드점 추종, READY 되면 자동 격발)
echo   7. 3단계 격발 판정 오프라인 검증 (ROS2/Gazebo 불필요)
echo   8. 스코프 뷰어 창만 다시 열기 (4~6번 실행 중 창을 닫았을 때)
echo   9. Gazebo 3D GUI 창 토글 (현재 헤드리스=%HEADLESS%)
echo.
echo   O. 녹화 폴더 열기 .
echo   L. ROS 실행 로그 폴더 열기 (WSL)
echo   0. 종료 (exit)
echo ============================================================
echo.
set "choice="
set /p choice=번호 또는 O/L을 입력하세요:

if not defined choice goto menu
if "%choice%"=="1" goto do_check
if "%choice%"=="2" goto do_link
if "%choice%"=="3" goto do_build
if "%choice%"=="5" goto do_player
if /i "%choice%"=="O" goto do_recordings
if /i "%choice%"=="L" goto do_logs
if "%choice%"=="4" goto do_run_plain
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
echo.
echo   ============================================================
echo   [4번] 비사격 시뮬레이션 및 스마트 스코프 뷰어를 구동합니다.
echo   탐지·조준 계산과 리드점 추종 보조만 동작하며 격발 제어는 꺼집니다.
echo   웹 뷰어 주소: http://127.0.0.1:9999
echo   ============================================================
echo.
start "" python "%VIEWER_SCRIPT%"
%WSL_CMD% "%ENVPFX% SMASH_ENABLE_FIRE_CONTROL=false SMASH_LAUNCH_VIEWER=true bash '%SETUP_SCRIPT%' run"
goto after

:do_run_fire_auto
echo.
echo   [6번] 시뮬레이션 조건을 선택합니다.
call :select_target_motion
call :select_camera_fov
call :set_vel_gate
set "ENVPFX=SMASH_CORE_PATH='%CORE_PATH_WSL%' SMASH_HEADLESS=%HEADLESS% SMASH_TARGET_MOTION=%TARGET_MOTION% SMASH_TARGET_SPEED_KMH=%TARGET_KMH% SMASH_TARGET_HALF_LENGTH=%TARGET_HALF_LENGTH% SMASH_CAMERA_HFOV=%CAMERA_HFOV% SMASH_AIM_MAX_GOAL_JUMP=%AIM_JUMP% SMASH_AIM_ACQUIRE_ENVELOPE=%AIM_ENVELOPE% SMASH_OPERATOR_FOLLOW=%OPERATOR_FOLLOW% SMASH_MOTION_LAG_S=%MOTION_LAG_S% SMASH_FIRE_TRIGGER=%FIRE_TRIGGER% SMASH_DEMO_KILL=%DEMO_KILL% SMASH_MOTION_VEL_DISAGREE_MPS=%VEL_GATE%"
echo.
echo   ============================================================
echo   [6번] 격발 제어 시뮬레이션 및 스마트 스코프 뷰어를 구동합니다.
echo   시뮬레이터 Pan/Tilt가 리드점을 추종하고, READY 가 되면 격발이 자동으로 발생합니다.
echo   격발 방식: !FIRE_TRIGGER! (manual 이면 READY 일 때 마우스 왼쪽 클릭 또는 스페이스바로 한 번에 한 발, auto 이면 자동)
echo   실제 제품의 사수 조작을 시뮬레이터 조준 보조가 대신하며, FCS의 READY 게이트는 그대로 유지됩니다.
echo   이동 표적이 출발하기 전에는 격발하지 않습니다. 격추/재출현 연출은 수동 격발이면 켜지고 자동이면 꺼집니다(!DEMO_KILL!).
echo   표적: !TARGET_NAME! !TARGET_KMH! km/h
echo   화각: !CAMERA_FOV_LABEL!
echo   웹 뷰어 주소: http://127.0.0.1:9999
echo   ============================================================
echo.
start "" python "%VIEWER_SCRIPT%"
%WSL_CMD% "%ENVPFX% SMASH_ENABLE_FIRE_CONTROL=true SMASH_LAUNCH_VIEWER=true bash '%SETUP_SCRIPT%' run"
goto after

:do_verify
%WSL_CMD% "%ENVPFX% bash '%SETUP_SCRIPT%' verify_fire_control"
goto after

:do_viewer
echo   스코프 뷰어 창만 다시 엽니다. 시뮬레이터는 켜지 않습니다.
echo   4~6번이 다른 런처 창에서 실행 중이어야 화면이 연결됩니다.
python "%VIEWER_SCRIPT%"
goto after


:do_player
python "%PLAYER_SCRIPT%"
goto after

:do_recordings
if not exist "%PROJECT_ROOT%\recordings" mkdir "%PROJECT_ROOT%\recordings"
start "" explorer.exe "%PROJECT_ROOT%\recordings"
goto menu

:do_logs
wsl.exe %WSL_D% --exec sh -c "mkdir -p ~/.ros/log; cd ~/.ros/log; explorer.exe ."
goto after

:after
set "TASK_EXIT=!errorlevel!"
set "LAST_RESULT=Exit code !TASK_EXIT!"
echo.
if "!TASK_EXIT!"=="0" (
    echo   [완료] 작업이 정상 종료되었습니다.
) else (
    echo   [실패] 종료 코드 !TASK_EXIT! - 위 오류 메시지와 L 메뉴의 로그를 확인하세요.
)
pause
goto menu

:end
endlocal
exit /b 0

:set_vel_gate
if defined VEL_DISAGREE_USER (
    set "VEL_GATE=!VEL_DISAGREE_USER!"
) else if /i "!TARGET_MOTION!"=="maneuver" (
    set "VEL_GATE=1.5"
) else (
    set "VEL_GATE=0.75"
)
exit /b 0

:select_camera_fov
echo.
echo   카메라 수평 화각을 선택하세요.
echo   선택한 비행 방식: !TARGET_NAME!
echo   자동 적용한 권장 화각: !CAMERA_FOV_LABEL!
echo.
echo     [1] 광각 0.4 rad (수평 22.9도)
echo         35 m에서 UAV 긴 변 약 22 px, 화면 가로폭 약 14.2 m
echo         넓은 영역에서 UAV를 찾는 탐색 시험용입니다.
echo         UAV가 작게 보여 광류 특징점이 부족하며 전체 파이프라인은 불안정할 수 있습니다.
echo.
echo     [2] 중간 0.2 rad (수평 11.5도)
echo         35 m에서 UAV 긴 변 약 44 px, 화면 가로폭 약 7.0 m
echo         탐지와 추적이 안정화되는 경계 조건을 시험할 때 사용합니다.
echo         이전 15 km/h 왕복 시험에서는 STABLE 약 40%%로 안정 추적에는 부족했습니다.
echo.
echo     [3] 협각 0.12 rad (수평 6.9도)
echo         35 m에서 UAV 긴 변 약 73 px, 화면 가로폭 약 4.2 m
echo         탐지부터 추적, STABLE, 조준점 생성까지 안정적으로 확인하는 시험용입니다.
echo         10 km/h, 왕복 반폭 1.5 m 시험에서 STABLE 100%%를 3회 확인했습니다.
echo.
echo     Enter. 현재 선택 유지 (!CAMERA_HFOV! rad)
set "fov_in="
set /p fov_in=화각 (1~3, Enter=현재값):
if defined fov_in set "fov_in=!fov_in: =!"
if not defined fov_in goto camera_fov_selected
if "!fov_in!"=="1" (
    set "CAMERA_HFOV=0.4"
    set "CAMERA_FOV_LABEL=광각 0.4 rad - 22.9도, 탐색용"
) else if "!fov_in!"=="2" (
    set "CAMERA_HFOV=0.2"
    set "CAMERA_FOV_LABEL=중간 0.2 rad - 11.5도, 경계 시험"
) else if "!fov_in!"=="3" (
    set "CAMERA_HFOV=0.12"
    set "CAMERA_FOV_LABEL=협각 0.12 rad - 6.9도, 안정 추적"
) else (
    echo   잘못된 입력입니다: !fov_in!
    goto select_camera_fov
)
:camera_fov_selected
echo   선택됨: !CAMERA_FOV_LABEL!
if "!TARGET_MOTION!"=="none" goto camera_fov_done
if "!CAMERA_HFOV!"=="0.4" goto camera_fov_warn_wide
if "!TARGET_MOTION!"=="maneuver" if "!CAMERA_HFOV!"=="0.12" goto camera_fov_warn_figure8
goto camera_fov_done

:camera_fov_warn_wide
echo.
echo   [경고] 35 m UAV가 약 22 px로 보여 REL HOLD가 지속될 가능성이 높습니다.
echo   STABLE 잠금이 되지 않으면 UAV는 WAIT LOCK 상태로 호버링하고 조준점도 생성되지 않습니다.
echo   기동을 강제로 시작하려면 뷰어에서 M을 누르되, 전체 파이프라인 안정성은 보장되지 않습니다.
goto camera_fov_done

:camera_fov_warn_figure8
echo.
echo   [경고] 협각에서는 좌우 3 m, 상하 1.8 m 8자 궤적이 화면 가장자리를 벗어날 수 있습니다.
echo   8자 기동은 중간 0.2 rad 화각을 권장합니다.

:camera_fov_done
exit /b 0

:select_target_motion
echo.
echo   UAV 비행 방식을 선택하세요. 기동 표적은 35 m에서 시작하며 FCS가 잡으면 출발합니다.
echo.
echo     [0] 정지 호버링 (none)
echo         UAV가 포탑 전방 약 10 m에서 움직이지 않습니다.
echo         정지 표적 탐지, 초기 정렬, 탄도 기준점 확인에 사용합니다.
echo.
echo     [1] 직선 통과 (straight)
echo         UAV가 중심에서 출발해 시야를 좌우로 등속 횡단합니다.
echo         중심 기준 반폭 15 m 끝에 도달하면 반대편에서 다시 진입합니다.
echo         화면 밖으로 나가므로 고속 횡단 표적의 재탐지와 재추종 시험용입니다.
echo.
echo     [2] 좌우 왕복 - 안정 추적 시험 권장 (cross)
echo         UAV가 중심 기준 좌우 30 m 구간을 등속으로 왕복합니다.
echo         양 끝에서는 곡선 선회 없이 이동 방향이 즉시 반전됩니다.
echo         사수 에뮬레이션이 포탑을 표적 방향으로 돌려 화면 중심에 유지합니다. 끄면 표적이 화면을 벗어납니다.
echo.
echo     [3] 8자 기동 (maneuver)
echo         UAV가 좌우 3 m, 상하 1.8 m 크기의 8자 곡선을 속력 일정으로 비행합니다.
echo         방향과 고도가 계속 변해 곡선 기동 추적과 예측 조준 시험에 사용합니다.
echo         협각에서는 화면 가장자리를 벗어날 수 있어 중간 0.2 rad 화각을 권장합니다.
echo.
echo     [4] 정면 접근 (approach)
echo         UAV가 포탑 정면 시선축을 따라 약 50 m에서 20 m까지 등속 접근합니다.
echo         가까운 지점에 도달하면 먼 지점으로 돌아가 접근을 반복합니다.
echo         화면 중앙을 유지하면서 거리 추정과 접근 표적 탄도 계산을 시험합니다.
set "pat_in="
set /p pat_in=방식 (0~4):
if defined pat_in set "pat_in=!pat_in: =!"
if "!pat_in!"=="0" (
    set "TARGET_MOTION=none"
    set "TARGET_NAME=정지 호버링"
    set "TARGET_HALF_LENGTH=3.0"
    set "CAMERA_HFOV=0.4"
    set "CAMERA_FOV_LABEL=광각 0.4 rad - 22.9도, 정지 표적 기본값"
    exit /b 0
)
if "!pat_in!"=="1" (
    set "new_motion=straight"
    set "new_name=직선 통과"
    set "new_half=15.0"
    set "CAMERA_HFOV=0.12"
    set "CAMERA_FOV_LABEL=협각 0.12 rad - 직선 표적 잠금 권장값"
) else if "!pat_in!"=="2" (
    set "new_motion=cross"
    set "new_name=왕복"
    set "new_half=30.0"
    set "CAMERA_HFOV=0.12"
    set "CAMERA_FOV_LABEL=협각 0.12 rad - 안정 추적 실측값"
) else if "!pat_in!"=="3" (
    set "new_motion=maneuver"
    set "new_name=8자 기동"
    set "new_half=3.0"
    set "CAMERA_HFOV=0.2"
    set "CAMERA_FOV_LABEL=중간 0.2 rad - 8자 궤적 화면 유지 권장값"
) else if "!pat_in!"=="4" (
    set "new_motion=approach"
    set "new_name=정면 접근"
    set "new_half=15.0"
    set "CAMERA_HFOV=0.12"
    set "CAMERA_FOV_LABEL=협각 0.12 rad - 접근 표적 추적 권장값"
) else (
    echo   잘못된 입력입니다: !pat_in!
    goto select_target_motion
)

:select_target_speed
echo.
echo   UAV 속도를 직접 입력하세요. 허용 범위는 1~150 km/h이며 소수도 가능합니다. 예: 3, 12.5, 10km/h
echo   Enter만 누르면 현재 값 !TARGET_KMH! km/h를 유지합니다.
echo   권장: 저속 확인 1~3 km/h, 안정 추적 확인 10 km/h, 경계 시험 15 km/h, 고속 기동 시험 30 km/h
set "kmh_in="
set /p kmh_in=속도 km/h:
if defined kmh_in set "kmh_in=!kmh_in: =!"
rem 단위를 붙여 입력해도 받는다 (10km/h, 10KM/H, 10kmh). 치환은 대소문자를 구분하지 않는다.
if defined kmh_in set "kmh_in=!kmh_in:km/h=!"
if defined kmh_in set "kmh_in=!kmh_in:kmh=!"
if not defined kmh_in set "kmh_in=!TARGET_KMH!"
echo !kmh_in!| findstr /r "^[0-9][0-9]*$ ^[0-9][0-9]*\.[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo   숫자가 아닙니다: !kmh_in!
    goto select_target_speed
)
python -c "import sys; v=float(sys.argv[1]); sys.exit(0 if min(150.0,max(1.0,v))==v else 1)" "!kmh_in!" >nul 2>nul
if errorlevel 1 (
    echo   허용 범위를 벗어났습니다. 1~150 km/h 사이로 입력하세요: !kmh_in!
    goto select_target_speed
)
set "TARGET_KMH=!kmh_in!"
set "TARGET_MOTION=!new_motion!"
set "TARGET_NAME=!new_name!"
set "TARGET_HALF_LENGTH=!new_half!"
echo   선택됨: !TARGET_NAME! !TARGET_KMH! km/h
echo   이 비행 방식의 권장 화각을 기본 선택했습니다: !CAMERA_FOV_LABEL!
exit /b 0
