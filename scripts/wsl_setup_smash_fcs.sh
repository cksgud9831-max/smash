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
#   ROS2_WS                기본값 $HOME/ros2_ws
#   SMASH_DEVICE           기본값 cpu  (CUDA 가 WSL 에서 동작하면 0)
#   SMASH_ENABLE_FIRE_CONTROL  기본값 false (로드맵 3단계: 가상 탄환 격발/판정)
#   SMASH_FIRE_MODE            기본값 manual ("manual" 또는 "auto")
#   SMASH_GT_POSE_TOPIC        기본값 /model/drone/pose (Gazebo 실측 위치 gz 토픽.
#                              `gz topic -l` 로 확인 후 다르면 이 값을 바꿀 것)

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
SMASH_DEVICE="${SMASH_DEVICE:-cpu}"
SMASH_ENABLE_FIRE_CONTROL="${SMASH_ENABLE_FIRE_CONTROL:-false}"
SMASH_FIRE_MODE="${SMASH_FIRE_MODE:-manual}"
if [ -d "$SMASH_CORE_PATH/simulation/ciws_turret_aerial_object_detection-main" ]; then
    REPO_DIR="$SMASH_CORE_PATH/simulation/ciws_turret_aerial_object_detection-main"
else
    REPO_DIR="$SMASH_CORE_PATH/ciws_turret_aerial_object_detection-main"
fi
SRC_DIR="$ROS2_WS/src"

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
    if [ -f "$SMASH_CORE_PATH/models/best_finetuned.pt" ]; then
        ok "파인튜닝 가중치 확인: $SMASH_CORE_PATH/models/best_finetuned.pt"
    else
        bad "가중치 파일이 없다: $SMASH_CORE_PATH/models/best_finetuned.pt"
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

run() {
    head_ "실행"
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
    echo "  device=$SMASH_DEVICE"
    echo "  enable_fire_control=$SMASH_ENABLE_FIRE_CONTROL  fire_mode=$SMASH_FIRE_MODE"
    if [ "$SMASH_ENABLE_FIRE_CONTROL" = "true" ]; then
        echo "  ground_truth_pose_topic=$SMASH_GT_POSE_TOPIC"
        echo "  (주의) 이 토픽이 실제로 발행되는지 다른 터미널에서 'gz topic -l' 로"
        echo "  확인할 것. 확인 전에는 Hit/Kill 판정이 전부 UNVERIFIED 로 남는다."
    fi
    echo
    ros2 launch drone_sim smash_scene.launch.py \
        core_path:="$SMASH_CORE_PATH" \
        model_path:="$SMASH_CORE_PATH/models/best_finetuned.pt" \
        device:="$SMASH_DEVICE" \
        enable_fire_control:="$SMASH_ENABLE_FIRE_CONTROL" \
        fire_mode:="$SMASH_FIRE_MODE" \
        ground_truth_pose_topic:="$SMASH_GT_POSE_TOPIC"
}

verify_fire_control() {
    head_ "3단계 격발/판정 로직 오라클 검증 (ROS/Gazebo 불필요, 순수 파이썬)"
    export SMASH_CORE_PATH
    python3 "$REPO_DIR/drone_sim/verify_stage3_fire_control.py"
}

viewer() {
    head_ "SMASH 대화형 스코프 뷰어 실행 (마우스 좌클릭 및 스페이스바 격발)"
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
    viewer) viewer ;;
    verify_fire_control) verify_fire_control ;;
    *)
        echo "사용법: bash $0 {check|link|build|run|viewer|verify_fire_control}"
        exit 2
        ;;
esac
