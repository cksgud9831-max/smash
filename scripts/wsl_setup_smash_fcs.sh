#!/usr/bin/env bash
# wsl_setup_smash_fcs.sh — WSL2 우분투에서 SMASH FCS 를 빌드하고 실행하기 위한 준비 스크립트.
#
# ★ 이 스크립트는 Windows PowerShell 이 아니라 WSL2 우분투 안에서 실행해야 한다.
#   PowerShell 에서 먼저 wsl 을 입력해 우분투 셸로 들어간 다음 실행할 것.
#
# 사용법
#   bash /mnt/d/Aiming/scripts/wsl_setup_smash_fcs.sh check                # 환경 진단만 (아무것도 바꾸지 않음)
#   bash /mnt/d/Aiming/scripts/wsl_setup_smash_fcs.sh link                 # D 드라이브 소스를 워크스페이스에 연결
#   bash /mnt/d/Aiming/scripts/wsl_setup_smash_fcs.sh build                # colcon 빌드
#   bash /mnt/d/Aiming/scripts/wsl_setup_smash_fcs.sh run                  # 시뮬레이션 실행
#   bash /mnt/d/Aiming/scripts/wsl_setup_smash_fcs.sh verify_fire_control  # 3단계 격발/판정 오라클 검증 (ROS/Gazebo 불필요)
#
# 환경변수로 덮어쓸 수 있는 값
#   SMASH_CORE_PATH        기본값 /mnt/d/Aiming
#   SMASH_MODEL_PATH       추적기 가중치. 기본값은 우리가 GA 튜닝으로 직접 학습시킨
#                          <core>/detector+tracker/ga_results/yolo11s_ga_final-3/weights/best.pt 이다.
#                          (학습 → 탐지 → 추적 → 조준 전 파이프라인을 한 모델로 관통시키기 위함.
#                           다른 가중치로 비교 실험할 때만 이 값을 지정한다.)
#   ROS2_WS                기본값 $HOME/ros2_ws
#   SMASH_DEVICE           기본값 cpu  (CUDA 가 WSL 에서 동작하면 0)
#   SMASH_ENABLE_FIRE_CONTROL  기본값 true (READY 에서 자동 격발 + HIT/MISS 판정). false 면 탐지·조준만 한다
#   SMASH_DEMO_KILL            기본값 false. true면 명중 시 낙하/재출현 시연을 켠다.
#   SMASH_HEADLESS             기본값 false. true 면 Gazebo GUI 없이 서버만 실행
#   SMASH_GT_POSE_TOPIC        기본값 /model/drone/pose (Gazebo 실측 위치 gz 토픽.
#                              `gz topic -l` 로 확인 후 다르면 이 값을 바꿀 것)
#   SMASH_TARGET_MOTION        기본값 none (정지 호버링). straight(좌우 직선 통과) | cross(좌우 왕복) |
#                              maneuver(8자 기동) | approach(정면 접근).
#                              이동 표적이면 표적을 35 m (33, 0, 12.5) 에 둔다. 아래 값으로 조절한다.
#                              어느 경우든 실행 중 스코프 뷰어 키로 방식(0~4)·속도(+/-)·정지(M)를 바꿀 수 있다
#                              (target_control:=true). SMASH_DEMO_KILL=true일 때만 격추/재출현 연출을 한다.
#   SMASH_TARGET_SPEED_KMH     기본값 30 (기동 속력 km/h, 소수 가능)
#   SMASH_TARGET_HALF_LENGTH   기본값 straight/approach 15, cross/maneuver 3.0 (반폭 m)   SMASH_TARGET_HALF_HEIGHT 기본값 1.8 (상하 반폭 m)
#   SMASH_CAMERA_HFOV          기본값 0.4 rad. 런처 F/11 메뉴에서 0.4 / 0.2 / 0.12 를 선택한다.
#                              이동 표적 시험 권장값은 0.12 rad: 35 m 표적이 약 73 px 로 보여
#                              전체 파이프라인이 안정적으로 동작했다(2026-09-23, 3회 STABLE 100 %).
#                              0.4 에서는 35 m 표적이 약 20 px 라
#                              추적기가 3 프레임마다 low_features 재탐지(=복구 이벤트)를 내 신뢰도가
#                              HOLD 에서 못 벗어난다(2026-09-22 실측). 0.2 에서 8자 3.0 x 1.8 m 가 화면에 든다.
#   SMASH_AIM_MAX_GOAL_JUMP    기본값 0.012 rad. 검출기 갱신 한 번에 허용하는 포 목표각 변화 한도(추종 한계 속도를 정한다)
#   SMASH_AIM_ACQUIRE_ENVELOPE 기본값 0.08 rad. 잠금(READY 1초 연속) 확정 전에 시작 자세에서 허용하는 최대 이탈각
#   SMASH_OPERATOR_FOLLOW      기본값 false. true 면 사수 에뮬레이션: 표적 실측 위치로 포를 돌려 표적을 화면 중심 부근에 둔다(지향에만 사용, 탐지·추정·격발은 FCS)
#   SMASH_OPERATOR_LATERAL_M   기본값 0.21 m. 사수 에뮬레이션 좌우 평행 보정(역기구학 상수 0.21 과 URDF 0 의 차이 상쇄)
#   SMASH_MOTION_LAG_S         기본값 0.17 s(smash_fcs.yaml 과 같음). FCS 가 이동 표적 리드에 더하는 영상/추적 위치 지연. 시뮬 비교용
#   SMASH_MOTION_FIT_RMSE_M    기본값 0.20 m(smash_fcs.yaml 과 같음). 이동 표적 궤적 적합 잔차가 이 값을 넘으면 READY 를 막는다
#                              ("MOTION FIT"). 8자처럼 곡선 기동 시험 비교용이며 실장비 값이 아니다
#   SMASH_MOTION_VEL_DISAGREE_MPS  기본값 0.75 m/s(smash_fcs.yaml 과 같음). 장기 OLS 속도와 최근 6프레임 속도의 차가 이 값을 넘으면
#                              READY 를 막는다("MOTION FIT"). 곡선 기동 시험 비교용이며 실장비 값이 아니다
#   SMASH_FIRE_TRIGGER         기본값 auto. manual 이면 READY 일 때 뷰어에서 마우스 왼쪽 클릭·스페이스바를 누른 한 번에 한 발만 나간다
#   SMASH_OPERATOR_FCS_LEAD    기본값 true. 사수 에뮬레이션이 조준선을 FCS 리드점(HUD LEAD)에 올린다. false 면 표적 중심
#   SMASH_OPERATOR_LEAD_S      기본값 0.16 s. 사수 에뮬레이션이 표적 앞을 겨냥하는 시간(포탑 지연을 상쇄해 표적을 화면 중심에 둔다)
#   SMASH_MANEUVER_RESET       기본값 true (방향 전환 시 오래된 운동 이력을 비워 리드 지연 방지)
#   SMASH_TARGET_START_DELAY   기본값 -1 (FCS 가 표적을 잡으면(STABLE 1초) 출발. 그전에 M 을 누르면 바로
#                              출발. 다시 누르면 정지/재개). 0 이상이면 기동 뒤 그 모의 초만큼 지나 자동 출발
#   SMASH_TARGET_START_ON_LOCK 기본값 true (위 "잡으면 출발". 실행 중 0~4 키로 방식을 바꿀 때도 같다)
#   SMASH_ENABLE_LIDAR         미지정 시 정지 근거리 표적은 true, 35 m 기동 표적은 false.
#                              현재 LiDAR 최대 유효거리는 15 m라 35 m에서는 끄는 것이 정상이다.
#   SMASH_GL_BACKEND           미지정 시 d3d12. 단, CUDA 추론과 GPU LiDAR를 동시에 켜면
#                              확인된 WSL 정지 현상을 피하기 위해 software를 선택한다.
#   MESA_D3D12_DEFAULT_ADAPTER_NAME  기본값 NVIDIA (내장 Intel GPU 대신 외장 GPU 선택)

set +u
COLCON_TRACE="${COLCON_TRACE:-}"

# ROS 2 Jazzy 환경 로드
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "/opt/ros/jazzy/setup.bash"
fi

# PyTorch / Ultralytics 가상환경 경로 자동 등록
for venv_dir in \
    "$HOME/venvs/smash/lib/python3.12/site-packages" \
    "$HOME/venvs/smash/lib/python3.11/site-packages" \
    "$HOME/.local/lib/python3.12/site-packages" \
    "$HOME/.local/lib/python3.11/site-packages"; do
    if [ -d "$venv_dir" ]; then
        case ":${PYTHONPATH:-}:" in
            *:"$venv_dir":*) ;;
            *) export PYTHONPATH="$venv_dir:${PYTHONPATH:-}" ;;
        esac
    fi
done

SMASH_CORE_PATH="${SMASH_CORE_PATH:-/mnt/d/Aiming}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
SMASH_DEVICE="${SMASH_DEVICE:-cuda:0}"
SMASH_ENABLE_FIRE_CONTROL="${SMASH_ENABLE_FIRE_CONTROL:-true}"
# 헤더 주석에 기본값이 적혀 있었으나 실제로는 설정되지 않아, 빈 문자열이
# ground_truth_pose_topic:= 로 넘어가 런치가 거부되던 문제를 바로잡는다.
SMASH_GT_POSE_TOPIC="${SMASH_GT_POSE_TOPIC:-/model/drone/pose}"
# true 면 Gazebo GUI 없이 서버만 띄운다. WSLg 처럼 GUI 렌더링이 안 되는 환경에서는
# 화면을 8번 스코프 뷰어(OpenCV 창)로 확인한다.
SMASH_HEADLESS="${SMASH_HEADLESS:-false}"
SMASH_TARGET_MOTION="${SMASH_TARGET_MOTION:-none}"
if [ -d "$SMASH_CORE_PATH/simulation/ciws_turret_aerial_object_detection-main" ]; then
    REPO_DIR="$SMASH_CORE_PATH/simulation/ciws_turret_aerial_object_detection-main"
else
    REPO_DIR="$SMASH_CORE_PATH/ciws_turret_aerial_object_detection-main"
fi
SRC_DIR="$ROS2_WS/src"

# 추적기 가중치. GA 튜닝으로 직접 학습시킨 결과물을 저장소 안 경로에서 그대로 쓴다.
#   시뮬레이터의 목적이 "학습 → 탐지 → 추적 → 조준" 전 파이프라인 검증이므로,
#   별도 파인튜닝본(models/best_finetuned.pt)이 아니라 학습 산출물 원본을 기본으로 한다.
#   models/ 는 .gitignore 대상이라 새로 clone 한 저장소에는 존재하지도 않는다.
SMASH_REPO_WEIGHTS="$SMASH_CORE_PATH/detector+tracker/ga_results/yolo11s_ga_final-3/weights/best.pt"
SMASH_MODEL_PATH="${SMASH_MODEL_PATH:-$SMASH_REPO_WEIGHTS}"

ok()   { printf '  [정상] %s\n' "$1"; }
bad()  { printf '  [문제] %s\n' "$1"; }
warn() { printf '  [주의] %s\n' "$1"; }
head_() { printf '\n== %s ==\n' "$1"; }

check() {
    local problems=0

    head_ "1. 셸 확인"
    if [ -n "${WSL_DISTRO_NAME:-}" ]; then
        ok "WSL2 배포판: $WSL_DISTRO_NAME"
    else
        warn "WSL_DISTRO_NAME 이 비어 있다. WSL 안이 아닐 수 있다."
    fi
    ok "현재 셸: $(readlink /proc/$$/exe 2>/dev/null || echo bash)"

    head_ "2. ROS 2 확인"
    if command -v ros2 >/dev/null 2>&1; then
        ok "ros2 명령 있음: $(command -v ros2)"
        ok "ROS_DISTRO=${ROS_DISTRO:-미설정}"
        if [ "${ROS_DISTRO:-}" != "jazzy" ]; then
            warn "ROS_DISTRO 가 jazzy 가 아니다. 프로젝트 기준은 Jazzy 다."
        fi
        if ! ros2 pkg list 2>/dev/null | grep "controller_manager" >/dev/null; then
            bad "controller_manager 패키지가 없다. 아래 명령으로 설치 필요:"
            echo "        sudo apt update && sudo apt install -y ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-gz-ros2-control ros-jazzy-controller-manager"
            problems=$((problems+1))
        else
            ok "ros2_control / controller_manager 확인됨"
        fi
    else
        bad "ros2 명령이 없다. 아래 중 하나가 필요하다."
        echo "        source /opt/ros/jazzy/setup.bash"
        echo "        또는 ROS 2 Jazzy 자체를 아직 설치하지 않았다."
        problems=$((problems+1))
    fi

    head_ "3. Gazebo Harmonic 확인"
    if command -v gz >/dev/null 2>&1; then
        ok "gz 명령 있음: $(gz sim --versions 2>/dev/null | head -1 || echo 버전확인실패)"
    else
        bad "gz 명령이 없다. Gazebo Harmonic 설치가 필요하다."
        problems=$((problems+1))
    fi

    head_ "4. SMASH 조준 코어 확인"
    if [ -f "$SMASH_CORE_PATH/aiming_engine/__init__.py" ] && [ -f "$SMASH_CORE_PATH/bridge/__init__.py" ]; then
        ok "조준 코어 확인: $SMASH_CORE_PATH"
    else
        bad "조준 코어를 찾지 못했다: $SMASH_CORE_PATH"
        echo "        aiming_engine/__init__.py 와 bridge/__init__.py 가 있어야 한다."
        problems=$((problems+1))
    fi
    if [ -f "$SMASH_MODEL_PATH" ]; then
        ok "추적기 가중치 확인 (GA 튜닝 학습 결과): $SMASH_MODEL_PATH"
    else
        bad "가중치 파일이 없다: $SMASH_MODEL_PATH"
        echo "        기본 경로: $SMASH_REPO_WEIGHTS"
        echo "        저장소에 커밋된 파일이다. git clone 이 정상적으로 끝났는지 확인할 것."
        problems=$((problems+1))
    fi

    head_ "5. 신규 FCS 소스 확인 (D 드라이브 원본)"
    local need=(
        "$REPO_DIR/drone_sim/drone_sim/smash_fcs_node.py"
        "$REPO_DIR/drone_sim/drone_sim/smash_fcs/turret_kinematics.py"
        "$REPO_DIR/drone_sim/drone_sim/smash_fcs/fire_control.py"
        "$REPO_DIR/drone_sim/config/smash_fcs.yaml"
        "$REPO_DIR/drone_sim/launch/smash_scene.launch.py"
        "$REPO_DIR/drone_sim/models/drone/model.sdf"
        "$REPO_DIR/drone_sim/verify_stage3_fire_control.py"
        "$REPO_DIR/ciws_turret/urdf/ciws_turret.urdf.xacro"
    )
    for f in "${need[@]}"; do
        if [ -f "$f" ]; then ok "$(basename "$f")"; else bad "없음: $f"; problems=$((problems+1)); fi
    done
    if grep -q "turret_boresight_laser" "$REPO_DIR/ciws_turret/urdf/ciws_turret.urdf.xacro" 2>/dev/null; then
        ok "URDF 에 광각 레이저 센서 포함됨"
    else
        bad "URDF 에 광각 레이저 센서가 없다. 갱신된 xacro 가 아니다."
        problems=$((problems+1))
    fi

    head_ "6. ROS 워크스페이스 확인 (가장 중요)"
    if [ -d "$SRC_DIR" ]; then
        ok "워크스페이스 소스 폴더: $SRC_DIR"
        echo "     아래는 $SRC_DIR 안에서 발견된 ciws_turret / drone_sim 패키지다."
        local found=0
        while IFS= read -r pkg; do
            found=$((found+1))
            local d; d="$(dirname "$pkg")"
            if [ -L "$d" ]; then
                echo "       [심볼릭링크] $d -> $(readlink -f "$d")"
            else
                echo "       [실제복사본] $d"
            fi
        done < <(find "$SRC_DIR" -maxdepth 4 -name package.xml 2>/dev/null \
                 | xargs -r grep -l -E '<name>(ciws_turret|drone_sim)</name>' 2>/dev/null)
        if [ "$found" -eq 0 ]; then
            warn "워크스페이스에 ciws_turret / drone_sim 이 아직 없다. link 단계가 필요하다."
        fi
    else
        warn "워크스페이스가 없다: $SRC_DIR (link 단계에서 생성한다)"
    fi

    head_ "7. 파이썬 의존성 확인"
    for m in numpy cv2 yaml ultralytics torch; do
        if python3 -c "import $m" >/dev/null 2>&1; then
            ok "python3 -c 'import $m'"
        else
            bad "python3 에서 $m 을 임포트할 수 없다"
            problems=$((problems+1))
        fi
    done
    if python3 -c "import cv_bridge" >/dev/null 2>&1; then
        ok "python3 -c 'import cv_bridge'"
    else
        warn "cv_bridge 임포트 실패. ROS 환경을 source 한 뒤 다시 확인할 것."
    fi

    head_ "결과"
    if [ "$problems" -eq 0 ]; then
        echo "  문제 없음. 다음 단계: bash $0 link"
    else
        echo "  문제 $problems 건. 위 [문제] 항목을 먼저 해결할 것."
    fi
    return 0
}

link() {
    head_ "워크스페이스 연결"
    mkdir -p "$SRC_DIR" || { bad "워크스페이스 생성 실패: $SRC_DIR"; return 1; }

    # 이름이 같은 기존 패키지가 있으면 colcon 이 중복으로 실패한다. 먼저 확인한다.
    local dup=0
    while IFS= read -r pkg; do
        local d; d="$(dirname "$pkg")"
        local real; real="$(readlink -f "$d")"
        if [ "$real" != "$(readlink -f "$REPO_DIR/ciws_turret")" ] && \
           [ "$real" != "$(readlink -f "$REPO_DIR/drone_sim")" ]; then
            warn "다른 위치의 동일 패키지 발견: $d"
            dup=$((dup+1))
        fi
    done < <(find "$SRC_DIR" -maxdepth 4 -name package.xml 2>/dev/null \
             | xargs -r grep -l -E '<name>(ciws_turret|drone_sim)</name>' 2>/dev/null)

    if [ "$dup" -gt 0 ]; then
        echo
        echo "  ★ 워크스페이스에 이미 별도 복사본이 있다. 이대로 두면 colcon 이"
        echo "    같은 이름의 패키지를 두 번 찾아 빌드에 실패하거나, D 드라이브의"
        echo "    새 코드가 아니라 옛 복사본을 빌드하게 된다."
        echo
        echo "    해결 방법 두 가지 중 하나를 고를 것."
        echo
        echo "    (가) 옛 복사본을 옆으로 치우고 D 드라이브를 단일 원본으로 삼는다 (권장)"
        echo "         mkdir -p $SRC_DIR/_backup_$(date +%Y%m%d)"
        echo "         mv <위에 표시된 폴더> $SRC_DIR/_backup_$(date +%Y%m%d)/"
        echo "         그 뒤 이 스크립트의 link 를 다시 실행."
        echo
        echo "    (나) 옛 복사본 위에 새 파일을 덮어쓴다"
        echo "         rsync -av $REPO_DIR/ciws_turret/  <옛 ciws_turret 경로>/"
        echo "         rsync -av $REPO_DIR/drone_sim/    <옛 drone_sim 경로>/"
        echo "         이 방식은 원본이 둘로 갈라지므로 이후 수정 때마다 다시 복사해야 한다."
        echo
        return 1
    fi

    for pkg in ciws_turret drone_sim; do
        local target="$SRC_DIR/$pkg"
        if [ -L "$target" ]; then
            rm "$target"
        elif [ -e "$target" ]; then
            bad "$target 이 이미 존재한다(링크가 아님). 직접 확인 후 옮길 것."
            return 1
        fi
        ln -s "$REPO_DIR/$pkg" "$target" || { bad "링크 실패: $target"; return 1; }
        ok "$target -> $REPO_DIR/$pkg"
    done

    echo
    echo "  다음 단계: bash $0 build"
}

build() {
    head_ "colcon 빌드"
    if ! command -v colcon >/dev/null 2>&1; then
        bad "colcon 이 없다. sudo apt install python3-colcon-common-extensions"
        return 1
    fi
    cd "$ROS2_WS" || { bad "워크스페이스로 이동 실패: $ROS2_WS"; return 1; }
    echo "  작업 폴더: $ROS2_WS"
    # symlink-install 은 쓰지 않는다. 소스가 /mnt/d 의 심볼릭링크라 이중 링크가 되어
    # 런타임에 경로 해석이 꼬이는 경우가 있다.
    colcon build --packages-select ciws_turret drone_sim
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        bad "빌드 실패 (종료코드 $rc). 위 로그를 확인할 것."
        return "$rc"
    fi
    ok "빌드 완료"
    echo
    echo "  다음 단계: bash $0 run"
}

# Match only this repository's simulation launch tree. Do not use broad
# patterns such as "pkill -f ros2", "pkill -f gz", or "killall python3":
# another ROS/Gazebo project may be running in the same WSL distribution.
# drone_sim/demo_director, drone_sim/target_mover, set_pose 브리지는 2026-09-22 에 추가했다. 빠져 있어서
# 런처 프로세스가 먼저 죽으면 고아로 남아 다음 실행의 시뮬레이터에 붙어 표적을 움직였다.
SMASH_SIM_PROCESS_PATTERN='ros2 launch drone_sim smash_scene\.launch\.py|gz sim .*turret_world\.sdf|drone_sim/smash_fcs|drone_sim/smash_scope_viewer|drone_sim/demo_director|drone_sim/target_mover|drone_sim/turret_command|ros_gz_bridge.*parameter_bridge.*(turret_camera|model/drone/pose|/clock@|set_pose)|controller_manager/spawner.*(joint_state_broadcaster|turret_controller)|ros_gz_sim.*create.*(ciws_turret|drone)'

# 부모 런처가 죽어 고아가 된 robot_state_publisher 를 정리한다. 명령줄이 흔한 형태라 패턴으로 잡으면
# 다른 프로젝트 것까지 죽이므로, 부모가 init 인 것(=고아)만 대상으로 한다. WSL2 는 세션마다 /init 이
# 고아를 넘겨받으므로 부모가 PID 1 이 아닐 수 있다(2026-09-22: 고아가 살아남아 다음 실행의 TF 에 섞였다).
reap_orphan_rsp() {
    local pid ppid pargs
    # 실행 파일(argv[0])이 robot_state_publisher 인 것만. 명령줄 어딘가에 문자열이 든 셸 등은 제외한다.
    for pid in $(ps -eo pid=,args= | awk '$2 ~ /\/robot_state_publisher\/robot_state_publisher$/ {print $1}'); do
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -z "$ppid" ] && continue
        # WSL2 에서 고아의 부모는 세션의 /init (comm 은 "Relay(N)", args 는 "/init") 이다.
        pargs=$(ps -o args= -p "$ppid" 2>/dev/null)
        if [ "$ppid" = 1 ] || [ "$pargs" = /init ]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
}

cleanup_smash_sim() {
    head_ "SMASH simulation cleanup"
    # 어느 경로로 끝나든 고아 정리를 거친다. 런처가 죽은 뒤에야 고아가 되므로 맨 끝에서 한다.
    trap 'sleep 0.3; reap_orphan_rsp' RETURN

    # `killall -9 gz_sim` cannot match the observed command line:
    # `gz sim ... turret_world.sdf`. Stop the owning launch process first,
    # then terminate only remaining SMASH-specific processes.
    pkill -INT -f 'ros2 launch drone_sim smash_scene\.launch\.py' 2>/dev/null || true
    sleep 0.5
    pkill -TERM -f "$SMASH_SIM_PROCESS_PATTERN" 2>/dev/null || true

    local attempt
    for attempt in {1..15}; do
        if ! pgrep -f "$SMASH_SIM_PROCESS_PATTERN" >/dev/null 2>&1; then
            ok "SMASH simulation processes stopped"
            return 0
        fi
        sleep 0.1
    done

    warn "SMASH processes still alive after 2 seconds; using SIGKILL fallback"
    pkill -KILL -f "$SMASH_SIM_PROCESS_PATTERN" 2>/dev/null || true
    sleep 0.2

    if pgrep -f "$SMASH_SIM_PROCESS_PATTERN" >/dev/null 2>&1; then
        warn "some SMASH processes remain; inspect with:"
        echo "        pgrep -af '$SMASH_SIM_PROCESS_PATTERN'"
        return 1
    fi
    ok "remaining SMASH simulation processes stopped"
}

run() {
    export QT_X11_NO_MITSHM=1
    # WSL2 의 기본 OpenGL 은 llvmpipe(CPU) 로 떨어진다. Mesa D3D12 드라이버로 NVIDIA GPU 를
    # 지정해 Gazebo(ogre2) 카메라 / gpu_lidar 센서 렌더링을 GPU 에서 돌린다.
    # 헤드리스 측정 (GTX 1050 Ti, 4코어):
    #   월드만:        카메라 12.2 -> 18.0 Hz, 레이저 18.5 -> 28.6 Hz, gz 서버 CPU 200% -> 120%
    #   FCS 포함 전체: 카메라 5.6 -> 13.1 Hz, FCS 처리율 3.8 -> 10.2 Hz
    # 35 m 기동 시험에서는 최대 15 m인 LiDAR가 항상 무반사이면서 GPU 렌더링만
    # 소모하므로 끈다. 10 m 정지 표적은 레이저 거리 보정을 위해 유지한다.
    if [ -z "${SMASH_ENABLE_LIDAR:-}" ]; then
        if [ "$SMASH_TARGET_MOTION" = "none" ]; then
            SMASH_ENABLE_LIDAR=true
        else
            SMASH_ENABLE_LIDAR=false
        fi
    fi
    # CUDA + D3D12 자체는 LiDAR를 끈 3회 시험에서 안정적이었다. CUDA 추론과
    # GPU LiDAR를 함께 켠 경우에만 /clock 정지가 재현되어 software로 우회한다.
    if [ -z "${SMASH_GL_BACKEND:-}" ]; then
        if [ "$SMASH_DEVICE" != "cpu" ] && [ "$SMASH_ENABLE_LIDAR" = "true" ]; then
            SMASH_GL_BACKEND=software
        else
            SMASH_GL_BACKEND=d3d12
        fi
    fi
    if [ "$SMASH_GL_BACKEND" = "software" ]; then
        export LIBGL_ALWAYS_SOFTWARE=1
        unset GALLIUM_DRIVER MESA_D3D12_DEFAULT_ADAPTER_NAME
    else
        unset LIBGL_ALWAYS_SOFTWARE
        export GALLIUM_DRIVER=d3d12
        export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
    fi
    head_ "실행"
    # 이전 잔류 프로세스 자동 청소 (충돌 방지)
    cleanup_smash_sim || return 1
    # 강제 종료된 노드가 남긴 Fast DDS 공유메모리 파일 정리. 640 영상용 프로파일
    # (ciws_turret/config/fastdds_large_messages.xml)은 참가자마다 16 MB 를 잡으므로
    # 남겨 두면 /dev/shm 에 쌓인다. 사용 중인 파일은 건드리지 않는다.
    command -v fastdds >/dev/null 2>&1 && fastdds shm clean >/dev/null 2>&1 || true
    sleep 0.5
    if [ ! -f "$ROS2_WS/install/setup.bash" ]; then
        bad "$ROS2_WS/install/setup.bash 가 없다. 먼저 build 를 실행할 것."
        return 1
    fi
    set +u
    # shellcheck disable=SC1091
    source "$ROS2_WS/install/setup.bash"
    export SMASH_CORE_PATH
    export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
    export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-}"
    echo "  SMASH_CORE_PATH=$SMASH_CORE_PATH"
    echo "  model_path=$SMASH_MODEL_PATH"
    echo "  device=$SMASH_DEVICE   headless=$SMASH_HEADLESS"
    echo "  gl_backend=$SMASH_GL_BACKEND   lidar=$SMASH_ENABLE_LIDAR"
    echo "  enable_fire_control=$SMASH_ENABLE_FIRE_CONTROL"
    if [ "$SMASH_ENABLE_FIRE_CONTROL" = "true" ]; then
        echo "  ground_truth_pose_topic=$SMASH_GT_POSE_TOPIC"
        echo "  (주의) 이 토픽이 실제로 발행되는지 다른 터미널에서 'gz topic -l' 로"
        echo "  확인할 것. 확인 전에는 Hit/Kill 판정이 전부 UNVERIFIED 로 남는다."
    fi
    local target_args=()
    if [ "$SMASH_TARGET_MOTION" != "none" ]; then
        local default_half=3.0
        case "$SMASH_TARGET_MOTION" in straight|approach) default_half=15.0 ;; esac
        target_args=(
            target_motion:="$SMASH_TARGET_MOTION"
            target_speed_kmh:="${SMASH_TARGET_SPEED_KMH:-30}"
            target_half_length:="${SMASH_TARGET_HALF_LENGTH:-$default_half}"
            target_half_height:="${SMASH_TARGET_HALF_HEIGHT:-1.8}"
            maneuver_reset:="${SMASH_MANEUVER_RESET:-true}"
            target_start_delay:="${SMASH_TARGET_START_DELAY:--1}"
            target_x:=33.0 target_y:=0.0 target_z:=12.5
        )

        echo "  표적: $SMASH_TARGET_MOTION ${SMASH_TARGET_SPEED_KMH:-30} km/h, FCS 가 잡으면 출발 (M: 바로 출발/정지)"
        echo "  (표적 중심 거리는 화각별 포착 가능 거리 안으로 제한된다. 해제: SMASH_ENFORCE_ENVELOPE=false)"
    else
        target_args=(target_speed_kmh:="${SMASH_TARGET_SPEED_KMH:-30}")
        echo "  표적: 정지 호버링"
    fi
    # 표적 시작 위치 "x,y,z"(world, m). 거리별 포착 시험용. 비우면 위 기본값(정지 호버링은 launch 기본값)
    if [ -n "${SMASH_TARGET_POS:-}" ]; then
        IFS=, read -r _tx _ty _tz <<< "$SMASH_TARGET_POS"
        target_args+=(target_x:="$_tx" target_y:="$_ty" target_z:="$_tz")
        echo "  표적 시작 위치: $_tx $_ty $_tz"
    fi
    # 이동 표적은 시작 거리(약 35 m)와 비슷한 곳에 다시 나타나게 한다. launch 기본 목록(17/13/16/9 m)은 광각 0.4 rad 용이라
    # 화각 0.12 rad 에서는 9 m 표적이 화면에서 너무 커서(약 300 px) 탐지기가 못 잡아 4번째 재출현부터 멈췄다(2026-10-19 시험).
    # 화각이 넓어 범위를 넘으면 enforce_envelope 가 시작 시 가까이 당겨 준다.
    if [ -z "${SMASH_RESPAWN_POSITIONS:-}" ] && [ "$SMASH_TARGET_MOTION" != "none" ]; then
        SMASH_RESPAWN_POSITIONS="[33.0, 0.0, 12.5, 30.0, 3.0, 11.5, 34.0, -3.0, 13.0, 32.0, 1.0, 12.0]"
    fi
    # 재출현 위치 목록 [x,y,z, ...] (world). 거리별 포착 시험용. 비우면 launch 기본값
    if [ -n "${SMASH_RESPAWN_POSITIONS:-}" ]; then
        target_args+=(respawn_positions:="$SMASH_RESPAWN_POSITIONS")
        echo "  재출현 위치: $SMASH_RESPAWN_POSITIONS"
    fi
    echo "  실행 중 스코프 뷰어 키: 0~4 기동 방식(0 호버링 1 직선 2 왕복 3 8자 4 접근), +/- 속도, M 정지/재개"
    target_args+=(
        camera_hfov:="${SMASH_CAMERA_HFOV:-0.4}"
        target_control:=true
        target_start_on_lock:="${SMASH_TARGET_START_ON_LOCK:-true}"
    )
    echo "  카메라 수평 화각: ${SMASH_CAMERA_HFOV:-0.4} rad"
    echo
    # ros2 launch normally tears down its children on Ctrl+C. The EXIT trap
    # also removes a detached Gazebo / ros2_control tree if that is incomplete.
    # cleanup_smash_sim is idempotent, so startup and shutdown may both call it.
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'cleanup_smash_sim' EXIT
    ros2 launch drone_sim smash_scene.launch.py \
        core_path:="$SMASH_CORE_PATH" \
        headless:="$SMASH_HEADLESS" \
        model_path:="$SMASH_MODEL_PATH" \
        device:="$SMASH_DEVICE" \
        enable_fire_control:="$SMASH_ENABLE_FIRE_CONTROL" \
        aim_max_goal_jump_rad:="${SMASH_AIM_MAX_GOAL_JUMP:-0.012}" \
        aim_acquire_envelope_rad:="${SMASH_AIM_ACQUIRE_ENVELOPE:-0.08}" \
        operator_follow:="${SMASH_OPERATOR_FOLLOW:-false}" \
        operator_lead_s:="${SMASH_OPERATOR_LEAD_S:-0.16}" \
        operator_lateral_m:="${SMASH_OPERATOR_LATERAL_M:-0.21}" \
        operator_fcs_lead:="${SMASH_OPERATOR_FCS_LEAD:-true}" \
        trigger_mode:="${SMASH_FIRE_TRIGGER:-auto}" \
        motion_measurement_lag_s:="${SMASH_MOTION_LAG_S:-0.17}" \
        motion_ready_max_fit_rmse_m:="${SMASH_MOTION_FIT_RMSE_M:-0.20}" \
        motion_ready_max_velocity_disagreement_mps:="${SMASH_MOTION_VEL_DISAGREE_MPS:-0.75}" \
        aim_follow_mode:="${SMASH_AIM_FOLLOW_MODE:-lead}" \
        aim_offset_smoothing:="${SMASH_AIM_OFFSET_SMOOTHING:-0.8}" \
        aim_follower_rate_smoothing:="${SMASH_AIM_RATE_SMOOTHING:-0.0}" \
        demo_kill:="${SMASH_DEMO_KILL:-false}" \
        enable_lidar:="$SMASH_ENABLE_LIDAR" \
        ground_truth_pose_topic:="$SMASH_GT_POSE_TOPIC" \
        enforce_envelope:="${SMASH_ENFORCE_ENVELOPE:-true}"         launch_viewer:="${SMASH_LAUNCH_VIEWER:-false}"         "${target_args[@]}"
    local rc=$?
    trap - INT TERM EXIT
    cleanup_smash_sim || true
    return "$rc"
}

verify_fire_control() {
    head_ "3단계 격발/판정 로직 오라클 검증 (ROS/Gazebo 불필요, 순수 파이썬)"
    export SMASH_CORE_PATH
    python3 "$REPO_DIR/drone_sim/verify_stage3_fire_control.py"
}

viewer() {
    head_ "SMASH 스코프 영상 중계 서버 실행"
    if [ ! -f "$ROS2_WS/install/setup.bash" ]; then
        bad "$ROS2_WS/install/setup.bash 가 없다. 먼저 build 를 실행할 것."
        return 1
    fi
    set +u
    # shellcheck disable=SC1091
    source "$ROS2_WS/install/setup.bash"
    ros2 run drone_sim smash_scope_viewer
}

case "${1:-check}" in
    check)  check ;;
    link)   link ;;
    build)  build ;;
    run)    run ;;
    cleanup) cleanup_smash_sim ;;
    viewer) viewer ;;
    verify_fire_control) verify_fire_control ;;
    *)
        echo "사용법: bash $0 {check|link|build|run|cleanup|viewer|verify_fire_control}"
        exit 2
        ;;
esac
