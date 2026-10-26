#!/usr/bin/env bash
# P1-1: CUDA FCS/YOLO는 켜고 annotated/HUD/viewer는 끈 상태의 raw 입력·처리율 측정.
set -u
ROOT="${SMASH_CORE_PATH:-$(cd "$(dirname "$0")/../.." && pwd)}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
BACKEND="${SMASH_GL_BACKEND:-software}"
CAMERA_PX="${CAMERA_PX:-640}"; CAMERA_HFOV="${CAMERA_HFOV:-0.2}"; DURATION="${DURATION:-60}"
DEVICE="${SMASH_DEVICE:-cuda:0}"
LAUNCH_FCS="${LAUNCH_FCS:-true}"
SPAWN_TARGET="${SPAWN_TARGET:-true}"
LAUNCH_TURRET_COMMAND="${LAUNCH_TURRET_COMMAND:-true}"
TURRET_COMMAND_RATE="${TURRET_COMMAND_RATE:-60.0}"
PAN_DITHER_RAD="${PAN_DITHER_RAD:-0.0005}"
ENABLE_LIDAR="${ENABLE_LIDAR:-true}"
PUBLISH_ANNOTATED="${PUBLISH_ANNOTATED:-false}"
LAUNCH_VIEWER="${LAUNCH_VIEWER:-false}"
MODEL="${SMASH_MODEL_PATH:-$ROOT/detector+tracker/ga_results/yolo11s_ga_final-3/weights/best.pt}"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [ "$LAUNCH_FCS" = true ] && [ "$LAUNCH_VIEWER" = true ]; then
    RUN_ID="08_fcs_viewer_${BACKEND}_${STAMP}"; MODE="fcs-viewer"
elif [ "$LAUNCH_FCS" = true ] && [ "$PUBLISH_ANNOTATED" = true ]; then
    RUN_ID="07_fcs_annotated_${BACKEND}_${STAMP}"; MODE="fcs-annotated"
elif [ "$LAUNCH_FCS" = true ]; then
    RUN_ID="02_fcs_raw_${BACKEND}_${STAMP}"; MODE="fcs-raw"
elif [ "$SPAWN_TARGET" = true ] && [ "$LAUNCH_TURRET_COMMAND" = true ]; then
    RATE_TAG="${TURRET_COMMAND_RATE//./p}"
    RUN_ID="06_fullscene_tc${RATE_TAG}_${BACKEND}_${STAMP}"; MODE="scene-only"
elif [ "$SPAWN_TARGET" = false ] && [ "$LAUNCH_TURRET_COMMAND" = true ]; then
    RUN_ID="04_turret_command_only_${BACKEND}_${STAMP}"; MODE="turret-command-only"
elif [ "$SPAWN_TARGET" = true ] && [ "$LAUNCH_TURRET_COMMAND" = false ]; then
    RUN_ID="05_drone_only_${BACKEND}_${STAMP}"; MODE="drone-only"
else
    RUN_ID="06_scene_minimal_${BACKEND}_${STAMP}"; MODE="scene-only"
fi
RUN_DIR="$ROOT/results/fps_diag/runs"; REPORT_DIR="$ROOT/results/fps_diag/reports"
JSON="$RUN_DIR/$RUN_ID.json"; LOG="$RUN_DIR/$RUN_ID.log"; REPORT="$REPORT_DIR/$RUN_ID.md"
RESOURCE="$RUN_DIR/$RUN_ID.resources.json"; RESOURCE_LOG="$RUN_DIR/$RUN_ID.resources.log"
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
for venv_dir in "$HOME/venvs/smash/lib/python3.12/site-packages" "$HOME/venvs/smash/lib/python3.11/site-packages"; do
    [ -d "$venv_dir" ] && export PYTHONPATH="$venv_dir:${PYTHONPATH:-}" && break
done

PATTERN='ros2 launch drone_sim smash_scene\.launch\.py|gz sim .*turret_world\.sdf'
if pgrep -af "$PATTERN" >/dev/null; then echo "existing simulation" >"$LOG"; exit 2; fi
LAUNCH_PID=""
RESOURCE_PID=""
cleanup() {
    [ -n "$RESOURCE_PID" ] && kill -TERM "$RESOURCE_PID" 2>/dev/null || true
    if [ -n "$LAUNCH_PID" ]; then
        kill -INT "$LAUNCH_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do kill -0 "$LAUNCH_PID" 2>/dev/null || break; sleep 0.25; done
        if kill -0 "$LAUNCH_PID" 2>/dev/null; then
            kill -TERM "$LAUNCH_PID" 2>/dev/null || true
        fi
        wait "$LAUNCH_PID" 2>/dev/null || true
    fi
    # ros2 launch가 먼저 종료돼 고아가 된 Gazebo 서버도 단계적으로 정리한다.
    pkill -INT -f 'gz sim .*turret_world\.sdf' 2>/dev/null || true
    for _ in $(seq 1 20); do
        pgrep -f 'gz sim .*turret_world\.sdf' >/dev/null || break
        sleep 0.25
    done
    pkill -TERM -f 'gz sim .*turret_world\.sdf' 2>/dev/null || true
    sleep 0.5
    pkill -KILL -f 'gz sim .*turret_world\.sdf' 2>/dev/null || true
}
trap cleanup EXIT INT TERM

COMMAND="ros2 launch drone_sim smash_scene.launch.py core_path:=$ROOT headless:=true model_path:=$MODEL device:=$DEVICE camera_px:=$CAMERA_PX camera_hfov:=$CAMERA_HFOV enable_lidar:=$ENABLE_LIDAR publish_annotated:=$PUBLISH_ANNOTATED launch_fcs:=$LAUNCH_FCS spawn_target:=$SPAWN_TARGET launch_turret_command:=$LAUNCH_TURRET_COMMAND turret_command_rate:=$TURRET_COMMAND_RATE pan_dither_rad:=$PAN_DITHER_RAD launch_viewer:=$LAUNCH_VIEWER target_motion:=none target_control:=false"
echo "run_id=$RUN_ID" | tee "$LOG"; echo "$COMMAND" | tee -a "$LOG"
# shellcheck disable=SC2086
$COMMAND >>"$LOG" 2>&1 & LAUNCH_PID=$!
for _ in $(seq 1 180); do
    if [ "$LAUNCH_FCS" = true ]; then
        ros2 topic info /smash_fcs/state 2>/dev/null | grep -q 'Publisher count: 1' && break
    else
        ros2 topic info /turret_camera/image_raw 2>/dev/null | grep -q 'Publisher count: 1' && break
    fi
    sleep 1
done

python3 "$ROOT/benchmarks/sim_measurement/resource_sampler.py" "$RESOURCE" --duration "$DURATION" \
    >"$RESOURCE_LOG" 2>&1 &
RESOURCE_PID=$!
set +e
python3 "$ROOT/benchmarks/sim_measurement/camera_only_probe.py" --output "$JSON" --duration "$DURATION" \
    --run-id "$RUN_ID" --backend "$BACKEND" --camera-px "$CAMERA_PX" --camera-hfov "$CAMERA_HFOV" \
    --log-path "$LOG" --command "$COMMAND" --mode "$MODE" --resource-path "$RESOURCE" 2>&1 | tee -a "$LOG"
PROBE_RC=${PIPESTATUS[0]}
set -e
wait "$RESOURCE_PID" 2>/dev/null || true
RESOURCE_PID=""
cleanup; trap - EXIT INT TERM; LAUNCH_PID=""; sleep 1
if pgrep -af "$PATTERN" >"$RUN_DIR/$RUN_ID.postflight.txt"; then
    CLEANUP="FAIL: see $RUN_DIR/$RUN_ID.postflight.txt"
else
    CLEANUP="CLEAN"; rm -f "$RUN_DIR/$RUN_ID.postflight.txt"
fi
if [ -f "$JSON" ]; then
    python3 "$ROOT/benchmarks/sim_measurement/camera_only_probe.py" --render-report "$JSON" "$REPORT" "$CLEANUP"
fi
echo "result=$JSON"; echo "report=$REPORT"; exit "$PROBE_RC"
