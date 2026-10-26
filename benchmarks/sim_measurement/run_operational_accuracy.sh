#!/usr/bin/env bash
# 35 m / 3 km/h 실제 자동 추종 격발 1회와 분기 보고서 생성.
set -uo pipefail
ROOT="${SMASH_CORE_PATH:-$(cd "$(dirname "$0")/../.." && pwd)}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
DURATION="${DURATION:-180}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="09_auto_cross3_${STAMP}"
RUN_DIR="$ROOT/results/fps_diag/runs"; REPORT_DIR="$ROOT/results/fps_diag/reports"
RESULT="$RUN_DIR/$RUN_ID.json"; SIM_LOG="$RUN_DIR/$RUN_ID.sim.log"; PROBE_LOG="$RUN_DIR/$RUN_ID.probe.log"
REPORT="$REPORT_DIR/$RUN_ID.md"; POST="$RUN_DIR/$RUN_ID.postflight.txt"
WINDOW_MARKER="$RUN_DIR/$RUN_ID.window_started"
mkdir -p "$RUN_DIR" "$REPORT_DIR"
rm -f "$WINDOW_MARKER"

set +u
source /opt/ros/jazzy/setup.bash
source "$ROS2_WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-45}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${DDS_XML:-$ROS2_WS/install/ciws_turret/share/ciws_turret/config/fastdds_large_messages.xml}"

SIM_PID=""; PROBE_PID=""
cleanup() {
    [ -n "$PROBE_PID" ] && kill -INT "$PROBE_PID" 2>/dev/null || true
    [ -n "$SIM_PID" ] && kill -INT "$SIM_PID" 2>/dev/null || true
    sleep 1
    # 백그라운드 bash는 SIGINT를 무시할 수 있으므로 저장소 범위 cleanup으로
    # 자식 ROS/Gazebo 트리를 먼저 끝낸 뒤 wrapper를 회수한다.
    timeout 15s bash "$ROOT/scripts/wsl_setup_smash_fcs.sh" cleanup >/dev/null 2>&1 || true
    [ -n "$SIM_PID" ] && kill -TERM "$SIM_PID" 2>/dev/null || true
    sleep 1
    [ -n "$SIM_PID" ] && kill -KILL "$SIM_PID" 2>/dev/null || true
    wait "$PROBE_PID" 2>/dev/null || true; wait "$SIM_PID" 2>/dev/null || true
    # launch wrapper가 종료되면서 늦게 고아가 된 자식까지 한 번 더 회수한다.
    timeout 15s bash "$ROOT/scripts/wsl_setup_smash_fcs.sh" cleanup >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

COMMAND="SMASH_TARGET_MOTION=cross SMASH_TARGET_SPEED_KMH=3 SMASH_TARGET_HALF_LENGTH=1.5 SMASH_CAMERA_HFOV=0.2 SMASH_ENABLE_LIDAR=false SMASH_GL_BACKEND=d3d12 SMASH_DEVICE=cuda:0 SMASH_LAUNCH_VIEWER=true bash scripts/wsl_setup_smash_fcs.sh run"
echo "run_id=$RUN_ID" | tee "$SIM_LOG"
echo "$COMMAND" | tee -a "$SIM_LOG"
SMASH_CORE_PATH="$ROOT" SMASH_HEADLESS=true SMASH_TARGET_MOTION=cross SMASH_TARGET_SPEED_KMH=3 \
SMASH_TARGET_HALF_LENGTH=1.5 SMASH_CAMERA_HFOV=0.2 SMASH_ENABLE_LIDAR=false SMASH_GL_BACKEND=d3d12 \
SMASH_DEVICE=cuda:0 SMASH_ENABLE_FIRE_CONTROL=true SMASH_LAUNCH_VIEWER=true \
SMASH_DEMO_KILL=false SMASH_MANEUVER_RESET=true SMASH_TARGET_START_ON_LOCK=false SMASH_TARGET_START_DELAY=-1 \
bash "$ROOT/scripts/wsl_setup_smash_fcs.sh" run >>"$SIM_LOG" 2>&1 & SIM_PID=$!

ready=false
for _ in $(seq 1 180); do
    grep -q '기동 완료' "$SIM_LOG" 2>/dev/null && ready=true && break
    kill -0 "$SIM_PID" 2>/dev/null || break
    sleep 1
done
if [ "$ready" != true ]; then
    printf '# 분기 보고서 — %s\n\n- 판정: **INVALID**\n- 원인: FCS가 180초 안에 기동하지 않음\n- 로그: [%s](%s)\n' "$RUN_ID" "$(basename "$SIM_LOG")" "$SIM_LOG" >"$REPORT"
    exit 2
fi

python3 "$ROOT/benchmarks/sim_measurement/auto_aim_probe.py" "$RESULT" 33.0 0.0 12.5 \
    --window "$DURATION" --no-images --skip-align --start-motion 2 --acquire-ready-hold 2 \
    --window-marker "$WINDOW_MARKER" \
    >"$PROBE_LOG" 2>&1 & PROBE_PID=$!
wait "$PROBE_PID"; PROBE_RC=$?; PROBE_PID=""
cleanup; trap - EXIT INT TERM; SIM_PID=""
sleep 1
PATTERN='gz sim|parameter_bridge|drone_sim/|robot_state_publisher|controller_manager/spawner|ros2 launch drone_sim'
if pgrep -af "$PATTERN" | grep -v -E 'pgrep|run_operational_accuracy\.sh' >"$POST"; then CLEANUP="FAIL: $POST"; else CLEANUP=CLEAN; rm -f "$POST"; fi
if [ -f "$RESULT" ]; then
    python3 "$ROOT/benchmarks/sim_measurement/render_operational_report.py" "$RESULT" "$REPORT" \
        --run-id "$RUN_ID" --command "$COMMAND" --sim-log "$SIM_LOG" --probe-log "$PROBE_LOG" \
        --cleanup "$CLEANUP" || true
else
    printf '# 분기 보고서 — %s\n\n- 판정: **INVALID**\n- 원인: 프로브 결과 JSON 없음(rc=%s)\n- 로그: [%s](%s)\n' "$RUN_ID" "$PROBE_RC" "$(basename "$PROBE_LOG")" "$PROBE_LOG" >"$REPORT"
fi
echo "result=$RESULT"; echo "report=$REPORT"
exit "$PROBE_RC"
