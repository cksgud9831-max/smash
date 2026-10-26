#!/usr/bin/env bash
# SMASH 시뮬레이터 측정 1회: 헤드리스 시뮬레이터 기동 → (계측 FCS 교체) → 지연 기록 → 자동 조준 측정 → 정리.
# WSL2 Ubuntu 에서 실행한다. 이 저장소의 시뮬레이터 프로세스만 정리하므로(런처 cleanup 과 같은 패턴)
# 같은 WSL 의 다른 ROS/Gazebo 작업은 건드리지 않지만, 실행 중인 SMASH 시뮬레이터는 종료된다.
#
# 사용법: run_case.sh <label> <x> <y> <z> [window_s]
#   x y z     표적 호버링 위치 (m, 포탑 기준 월드 좌표)
#   window_s  측정 구간 길이, 기본 60
#
# 환경변수 (모두 선택):
#   CAMERA_PX=640        카메라 해상도 (정사각)
#   CAMERA_HFOV=         카메라 수평 화각(rad). 비우면 런치 기본값(0.4)
#   DEMO_KILL=false      격추 연출(낙하·재출현). 측정 중 표적이 옮겨지면 안 되므로 기본 끔
#   PROFILE=1            1 이면 FCS 를 fcs_timed.py 로 교체해 단계별 시간·지연을 기록
#   NOCLOCK=0            1 이면 clock_logger 를 띄우지 않는다 (지연 분석 불가)
#   NOIMG=1              1 이면 프로브가 영상 토픽을 구독하지 않는다 (측정 부하 최소화)
#   EXTRA_LAUNCH=        smash_scene.launch.py 에 그대로 넘길 추가 인자
#                        (이동 표적 예: "target_motion:=cross target_speed:=5.0 target_start_delay:=20.0")
#   PROBE_EXTRA=         auto_aim_probe.py 에 그대로 넘길 추가 인자 (이동 표적 예: "--track")
#   SIM_RUNS_DIR=<repo>/results/sim_runs   결과 폴더
#   SMASH_CORE_PATH=<이 스크립트 기준 저장소 루트>
#   ROS2_WS=$HOME/ros2_ws
#   ENABLE_LIDAR=false    35 m 시험에서는 광선이 닿지 않으므로 기본 끔. 근거리 시험은 true로 지정
#
# 결과 (SIM_RUNS_DIR/<label>.*):
#   .json            프로브 결과 (포착·정렬·상태 분포·격발·명중·빗나감)
#   .top.txt         측정 중간 CPU 스냅샷
#   .prof.log        단계별 시간 표 (PROFILE=1)
#   .lat_fcs.json    프레임·격발 시각 (PROFILE=1)
#   .track_diag.jsonl 프레임별 추적기 박스와 실측 표적 투영 화소 (PROFILE=1, track_diag.py 로 요약)
#   .lat_clock.json  모의↔실제 시각 대응 (NOCLOCK=0)
#   .preflight.txt   시작 전 남아 있던 프로세스 (있으면 측정하지 않는다)
#   .topics.txt      /clock, 표적 실측 토픽의 발행자 수 (1 이어야 한다)
#   .launch.log / .probe.log / .launchnotes.txt
set -u

if [ $# -lt 4 ]; then
    sed -n '2,30p' "$0"
    exit 2
fi
LABEL="$1"; X="$2"; Y="$3"; Z="$4"; WIN="${5:-60}"

HERE="$(cd "$(dirname "$0")" && pwd)"
SMASH_CORE_PATH="${SMASH_CORE_PATH:-$(cd "$HERE/../.." && pwd)}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
OUTDIR="${SIM_RUNS_DIR:-$SMASH_CORE_PATH/results/sim_runs}"
mkdir -p "$OUTDIR"
LAUNCHER="$SMASH_CORE_PATH/scripts/wsl_setup_smash_fcs.sh"
# 이 폴더의 측정 도구 프로세스 (런처 cleanup 패턴에는 없다)
TOOL_PATTERN='benchmarks/sim_measurement/(fcs_timed|clock_logger|auto_aim_probe)\.py'

cleanup() {
    pkill -INT -f "$TOOL_PATTERN" 2>/dev/null
    bash "$LAUNCHER" cleanup >/dev/null 2>&1
    sleep 0.5
    pkill -KILL -f "$TOOL_PATTERN" 2>/dev/null
    command -v fastdds >/dev/null 2>&1 && fastdds shm clean >/dev/null 2>&1
    return 0
}

cleanup
# 이전 실행의 시뮬레이터가 남아 있으면 Gazebo 토픽(/clock, 표적 실측)이 섞인다. Gazebo 통신에는
# ROS_DOMAIN_ID 같은 격리가 없다. 정리 뒤에도 남아 있으면 측정하지 않는다.
LEFTOVER_PATTERN='gz sim|parameter_bridge|drone_sim/|robot_state_publisher|controller_manager/spawner|ros2 launch drone_sim'
sleep 1
LEFT=$(pgrep -af "$LEFTOVER_PATTERN" | grep -v -E 'pgrep|run_case\.sh')
if [ -n "$LEFT" ]; then
    echo "$LEFT" >"$OUTDIR/$LABEL.preflight.txt"
    echo "[$(date +%T)] FAIL $LABEL: 이전 시뮬레이터 프로세스가 남아 있음 ($OUTDIR/$LABEL.preflight.txt)"
    echo "$LEFT"
    exit 1
fi
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$ROS2_WS/install/setup.bash"
set -u
for venv_dir in "$HOME/venvs/smash/lib/python3.12/site-packages" "$HOME/venvs/smash/lib/python3.11/site-packages"; do
    [ -d "$venv_dir" ] && export PYTHONPATH="$venv_dir:${PYTHONPATH:-}" && break
done
export SMASH_CORE_PATH
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-}"
ENABLE_LIDAR="${ENABLE_LIDAR:-false}"
if [ -z "${SMASH_GL_BACKEND:-}" ]; then
    # CUDA 추론과 D3D12는 LiDAR를 함께 렌더링할 때만 정지 현상이 재현됐다.
    if [ "${SMASH_DEVICE:-cuda:0}" != "cpu" ] && [ "$ENABLE_LIDAR" = "true" ]; then
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
# 대화형 세션과 DDS 를 섞지 않는다.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-41}"
# 측정 도구도 런치와 같은 DDS 프로파일을 써야 640 영상이 공유메모리로 온다.
export FASTRTPS_DEFAULT_PROFILES_FILE="${DDS_XML:-$ROS2_WS/install/ciws_turret/share/ciws_turret/config/fastdds_large_messages.xml}"

MODEL="${SMASH_MODEL_PATH:-$SMASH_CORE_PATH/detector+tracker/ga_results/yolo11s_ga_final-3/weights/best.pt}"
LOG="$OUTDIR/$LABEL.launch.log"
HFOV_ARG=()
[ -n "${CAMERA_HFOV:-}" ] && HFOV_ARG=(camera_hfov:="$CAMERA_HFOV")

echo "[$(date +%T)] start $LABEL target=($X, $Y, $Z) px=${CAMERA_PX:-640} hfov=${CAMERA_HFOV:-default} window=${WIN}s"
# shellcheck disable=SC2086
ros2 launch drone_sim smash_scene.launch.py core_path:="$SMASH_CORE_PATH" headless:=true \
    model_path:="$MODEL" device:="${SMASH_DEVICE:-cuda:0}" \
    enable_fire_control:=true demo_kill:="${DEMO_KILL:-false}" \
    enable_lidar:="$ENABLE_LIDAR" \
    camera_px:="${CAMERA_PX:-640}" "${HFOV_ARG[@]}" ${EXTRA_LAUNCH:-} \
    target_x:="$X" target_y:="$Y" target_z:="$Z" >"$LOG" 2>&1 &

wait_ready() {  # $1 = 로그 파일, $2 = 제한 시간(초)
    local i
    for i in $(seq 1 "$2"); do
        grep -q '기동 완료' "$1" 2>/dev/null && return 0
        sleep 1
    done
    return 1
}

if ! wait_ready "$LOG" 120; then
    echo "[$(date +%T)] FAIL $LABEL: FCS 가 120초 안에 기동하지 않음 ($LOG)"
    cleanup
    exit 1
fi

if [ "${PROFILE:-1}" = 1 ]; then
    sleep 3
    ARGS=$(pgrep -af 'lib/drone_sim/smash_fcs' | head -1 | sed 's/.*--ros-args/--ros-args/')
    pkill -KILL -f 'lib/drone_sim/smash_fcs'
    sleep 1
    # shellcheck disable=SC2086
    LAT_OUT="$OUTDIR/$LABEL.lat_fcs.json" TRACK_DIAG_OUT="$OUTDIR/$LABEL.track_diag.jsonl"         python3 "$HERE/fcs_timed.py" $ARGS >"$OUTDIR/$LABEL.prof.log" 2>&1 &
    if ! wait_ready "$OUTDIR/$LABEL.prof.log" 60; then
        echo "[$(date +%T)] FAIL $LABEL: 계측 FCS 기동 실패 ($OUTDIR/$LABEL.prof.log)"
        cleanup
        exit 1
    fi
fi

CL=""
if [ "${NOCLOCK:-0}" != 1 ]; then
    python3 "$HERE/clock_logger.py" "$OUTDIR/$LABEL.lat_clock.json" >"$OUTDIR/$LABEL.clocklog.log" 2>&1 &
    CL=$!
fi

# 측정 중 시계·실측 토픽의 발행자 수를 기록한다. 2 이상이면 다른 시뮬레이터가 섞인 것이다.
(sleep 20; for t in /clock /model/drone/pose; do
    echo "== $t"; ros2 topic info -v "$t" 2>&1 | grep -E "Publisher count|Node name|Subscription count"
done >"$OUTDIR/$LABEL.topics.txt") &

PROBE_ARGS=(--window "$WIN")
[ "${NOIMG:-1}" = 1 ] && PROBE_ARGS+=(--no-images)
# shellcheck disable=SC2206
[ -n "${PROBE_EXTRA:-}" ] && PROBE_ARGS+=(${PROBE_EXTRA})
python3 "$HERE/auto_aim_probe.py" "$OUTDIR/$LABEL.json" "$X" "$Y" "$Z" "${PROBE_ARGS[@]}" >"$OUTDIR/$LABEL.probe.log" 2>&1
RC=$?

[ -n "$CL" ] && kill -INT "$CL" 2>/dev/null && sleep 2
# PROFILE=1 이면 런치의 FCS 를 일부러 KILL 했으므로 그 사망 기록(exit code -9)은 뺀다.
grep -E '카메라 내부파라미터|spawner|died|Error|error' "$LOG"     | { [ "${PROFILE:-1}" = 1 ] && grep -v 'smash_fcs.*exit code -9' || cat; } | head -8 >"$OUTDIR/$LABEL.launchnotes.txt"
cleanup
sleep 1
LEFT=$(pgrep -af "$LEFTOVER_PATTERN" | grep -v -E 'pgrep|run_case\.sh')
if [ -n "$LEFT" ]; then
    echo "$LEFT" >"$OUTDIR/$LABEL.postflight.txt"
    echo "[$(date +%T)] WARN $LABEL: 정리 뒤에도 남은 프로세스가 있다 ($OUTDIR/$LABEL.postflight.txt)"
fi
echo "[$(date +%T)] done $LABEL (probe rc=$RC) -> $OUTDIR/$LABEL.json"
exit $RC
