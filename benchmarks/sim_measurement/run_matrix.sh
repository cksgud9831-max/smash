#!/usr/bin/env bash
# 화각 × 거리 × 반복 측정 매트릭스. 조건마다 run_case.sh 를 한 번씩 부른다.
# 2회차 이상은 거리 순서를 뒤집어 시간에 따른 PC 부하 변화가 한쪽 거리로 몰리지 않게 한다.
#
# 사용법: run_matrix.sh [prefix]
# 환경변수:
#   FOVS="0.12"           화각 목록(rad). 0.12=6.9°, 0.2=11.5°, 0.4=23°
#   DISTS="10 20 35 60 100"   거리 목록(m). 아래 POS 표에 있는 값만
#   REPS=2                반복 횟수
#   WINDOW=60             조건당 측정 구간(초)
#   그 밖의 변수는 run_case.sh 로 그대로 전달된다.
#
# 결과 라벨: <prefix>f<fov>_<dist>m_r<rep>  → summarize_runs.py 로 표를 만든다.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${1:-}"
FOVS="${FOVS:-0.12}"
DISTS="${DISTS:-10 20 35 60 100}"
REPS="${REPS:-2}"
WINDOW="${WINDOW:-60}"
SMASH_CORE_PATH="${SMASH_CORE_PATH:-$(cd "$HERE/../.." && pwd)}"
OUTDIR="${SIM_RUNS_DIR:-$SMASH_CORE_PATH/results/sim_runs}"
mkdir -p "$OUTDIR"
PROGRESS="$OUTDIR/${PREFIX}matrix_progress.log"

# 앙각 약 20도로 맞춘 정지 호버링 위치 (x y z). 10 m 는 기본 표적 위치(경사거리 10.8 m).
declare -A POS=(
    [10]="10.0 0.0 4.2"
    [20]="18.8 0.0 7.5"
    [35]="33.0 0.0 12.5"
    [60]="56.4 0.0 21.2"
    [100]="94.0 0.0 34.9"
)

for rep in $(seq 1 "$REPS"); do
    ds="$DISTS"
    [ $((rep % 2)) = 0 ] && ds="$(echo "$DISTS" | tr ' ' '\n' | tac | tr '\n' ' ')"
    for fov in $FOVS; do
        for d in $ds; do
            if [ -z "${POS[$d]:-}" ]; then
                echo "$(date +%T) skip unknown distance $d" | tee -a "$PROGRESS"
                continue
            fi
            label="${PREFIX}f${fov}_${d}m_r${rep}"
            echo "$(date +%T) start $label load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$PROGRESS"
            # shellcheck disable=SC2086
            CAMERA_HFOV="$fov" bash "$HERE/run_case.sh" "$label" ${POS[$d]} "$WINDOW" >>"$PROGRESS" 2>&1 \
                || echo "$(date +%T) FAILED $label" | tee -a "$PROGRESS"
        done
    done
done
echo "$(date +%T) MATRIX_DONE" | tee -a "$PROGRESS"
