#!/usr/bin/env bash
# P0-1: FCS/YOLO/HUD/viewer 없이 Gazebo raw camera 생성·전달률을 측정한다.
set -u

ROOT="${SMASH_CORE_PATH:-$(cd "$(dirname "$0")/../.." && pwd)}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
BACKEND="${SMASH_GL_BACKEND:-software}"
CAMERA_PX="${CAMERA_PX:-640}"
CAMERA_HFOV="${CAMERA_HFOV:-0.2}"
DURATION="${DURATION:-60}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="01_camera_only_${BACKEND}_${STAMP}"
RUN_DIR="$ROOT/results/fps_diag/runs"
REPORT_DIR="$ROOT/results/fps_diag/reports"
JSON="$RUN_DIR/$RUN_ID.json"
LOG="$RUN_DIR/$RUN_ID.log"
REPORT="$REPORT_DIR/$RUN_ID.md"
mkdir -p "$RUN_DIR" "$REPORT_DIR"

set +u
source /opt/ros/jazzy/setup.bash
source "$ROS2_WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-43}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${DDS_XML:-$ROS2_WS/install/ciws_turret/share/ciws_turret/config/fastdds_large_messages.xml}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-}"
if [ "$BACKEND" = software ]; then
    export LIBGL_ALWAYS_SOFTWARE=1
    unset GALLIUM_DRIVER MESA_D3D12_DEFAULT_ADAPTER_NAME
else
    unset LIBGL_ALWAYS_SOFTWARE
    export GALLIUM_DRIVER=d3d12
    export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
fi

PATTERN='ros2 launch ciws_turret view\.launch\.py|gz sim .*turret_world\.sdf'
if pgrep -af "$PATTERN" >"$RUN_DIR/$RUN_ID.preflight.txt"; then
    echo "preflight failed: an existing turret simulation is running" | tee "$LOG"
    rm -f "$RUN_DIR/$RUN_ID.preflight.txt"
    exit 2
fi
rm -f "$RUN_DIR/$RUN_ID.preflight.txt"

LAUNCH_PID=""
cleanup() {
    if [ -n "$LAUNCH_PID" ]; then
        kill -INT "$LAUNCH_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$LAUNCH_PID" 2>/dev/null || break
            sleep 0.25
        done
        if kill -0 "$LAUNCH_PID" 2>/dev/null; then
            kill -TERM "$LAUNCH_PID" 2>/dev/null || true
            pkill -TERM -f 'gz sim .*turret_world\.sdf' 2>/dev/null || true
        fi
        wait "$LAUNCH_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

COMMAND="ros2 launch ciws_turret view.launch.py headless:=true camera_px:=$CAMERA_PX camera_hfov:=$CAMERA_HFOV"
echo "run_id=$RUN_ID" | tee "$LOG"
echo "$COMMAND" | tee -a "$LOG"
# shellcheck disable=SC2086
$COMMAND >>"$LOG" 2>&1 &
LAUNCH_PID=$!

for _ in $(seq 1 120); do
    ros2 topic info /turret_camera/image_raw 2>/dev/null | grep -q 'Publisher count: 1' && break
    sleep 1
done
if ! ros2 topic info /turret_camera/image_raw 2>/dev/null | grep -q 'Publisher count: 1'; then
    echo "camera publisher did not become ready" | tee -a "$LOG"
fi

set +e
python3 "$ROOT/benchmarks/sim_measurement/camera_only_probe.py" \
    --output "$JSON" --duration "$DURATION" --run-id "$RUN_ID" --backend "$BACKEND" \
    --camera-px "$CAMERA_PX" --camera-hfov "$CAMERA_HFOV" --log-path "$LOG" --command "$COMMAND" \
    2>&1 | tee -a "$LOG"
PROBE_RC=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT INT TERM
LAUNCH_PID=""
sleep 1
if pgrep -af "$PATTERN" >"$RUN_DIR/$RUN_ID.postflight.txt"; then
    CLEANUP="FAIL: see $RUN_DIR/$RUN_ID.postflight.txt"
else
    CLEANUP="CLEAN"
    rm -f "$RUN_DIR/$RUN_ID.postflight.txt"
fi

if [ -f "$JSON" ]; then
    python3 "$ROOT/benchmarks/sim_measurement/camera_only_probe.py" \
        --render-report "$JSON" "$REPORT" "$CLEANUP"
else
    echo "probe JSON missing; report generation failed" | tee -a "$LOG"
fi
echo "result=$JSON"
echo "report=$REPORT"
exit "$PROBE_RC"
