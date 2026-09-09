"""SMASH 로드맵 2단계 리드각 정밀도 정량 검증 (등속 직선 비행, 1~2 m/s)

known_issues.md 1.5절이 지적한 "이중 미분" 노이즈 문제를 target_state.py의
velocity_smoothing_alpha, history_length 두 파라미터에 대해 실제로 정량 측정한다.
Level 1 오라클 검증(aim_oracle_validation.html)과 같은 방법론을 쓴다: 실제 3D 궤적을
직접 정의해 두고(drone_motion_models.compute_drone_linear_position, 정답을 아는 상태),
카메라/YOLO/OpticalFlowTracker 전체 비전 파이프라인은 거치지 않고 aiming_engine의
TargetState만 단독으로 떼어내 검증한다. 이렇게 하면 "비전 파이프라인의 노이즈"와
"이중 미분 자체가 만드는 노이즈"를 분리해서 볼 수 있다.

측정 대상 오차:
    위치오차(m): 예측한 미래 표적 위치와, 그 미래 시각의 실제(정답) 위치 사이 거리.
    각도오차(deg): 원점 플랫폼(06번 인터랙티브 시뮬레이터와 동일하게 (0,0,1.2)에 고정된
        사수 위치)에서 봤을 때, 예측 방향과 실제 방향 사이의 각도.

측정용 위치 노이즈(가정치, 실측 아님): Level 1 검증 보고서(level1_level2_validation_report.md)의
가정과 동일하게 픽셀 오차 시그마 2px를 기준으로 삼되, 06번 인터랙티브 시뮬레이터의 카메라
초점거리(fov_rad=0.85, cam_w=640이면 fx는 약 700)와 드론 사거리(35m)로 역산해 횡방향 위치오차
시그마를 약 0.1m로 두었다. 레이저 방향(사거리) 노이즈는 TF02 Pro 데이터시트 스펙(0.1~5m 구간
정확도 5cm 수준, known_issues.md/sensor_integration_gap_analysis.md 참고)에 맞춰 시그마 0.05m로 두었다.
둘 다 실측치가 아닌 가정된 조건이며, 실 하드웨어 연결 후에는 실측 노이즈로 교체해야 한다
(known_issues.md, level1_level2_validation_report.md Level 3 항목과 동일한 주의사항).

실행: python 07_stage2_lead_accuracy_validation.py
출력: 콘솔 요약 테이블 + <project_root>/results/stage2_lead_accuracy_results.json
"""

import json
import math
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
core_dir = os.path.join(project_root, "smash_core")
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

from aiming_engine.config import load_config
from aiming_engine.target_state import ConstantVelocityEstimator
from aiming_engine.types import Vector3
from drone_motion_models import compute_drone_linear_position, DRONE_PATH_HALF_LENGTH, DRONE_SPEED_MPS

FPS = 45.0
DURATION_S = 40.0  # 등속 왕복 주기(2*14/1.5 약 18.7초)를 넉넉히 두 번 이상 포함
PLATFORM_POS = np.array([0.0, 0.0, 1.2], dtype=np.float64)

# 측정용 가정 위치 노이즈(실측 아님, 위 docstring 근거 참고)
LATERAL_NOISE_STD_M = 0.10  # y, z (카메라 시선에 거의 수직인 성분)
RANGE_NOISE_STD_M = 0.05    # x (사거리 방향, 레이저 스펙 기준)

HISTORY_LENGTHS = [10, 20, 30]
SMOOTHING_ALPHAS = [1.0, 0.9, 0.7, 0.5, 0.3]

TURN_MARGIN_S = 0.5  # 방향 반전 지점 앞뒤 이 시간(초) 이내 샘플은 "선회 근접"으로 별도 집계


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def distance_to_leg_boundary(t):
    """가장 가까운 방향 반전 시점까지 남은 시간(초)."""
    leg_duration = (2.0 * DRONE_PATH_HALF_LENGTH) / DRONE_SPEED_MPS
    phase = math.fmod(t, leg_duration)
    return min(phase, leg_duration - phase)


def run_one_config(history_length, alpha, muzzle_velocity, rng):
    estimator = ConstantVelocityEstimator(history_length=history_length, min_dt=1e-3, velocity_smoothing_alpha=alpha)

    steady_pos_err, steady_ang_err = [], []
    turn_pos_err, turn_ang_err = [], []

    n_frames = int(DURATION_S * FPS)
    for i in range(n_frames):
        t = i / FPS
        true_x, true_y, true_z, _ = compute_drone_linear_position(t)
        noisy = np.array([
            true_x + rng.normal(0.0, RANGE_NOISE_STD_M),
            true_y + rng.normal(0.0, LATERAL_NOISE_STD_M),
            true_z + rng.normal(0.0, LATERAL_NOISE_STD_M),
        ])
        estimator.update(Vector3(*noisy), t)

        # 대략적인 비행시간(TOF) 추정: 중력/리드 각도를 무시한 단순 사거리/탄속 근사.
        # 이 스크립트의 목적은 뉴턴솔버 전체가 아니라 TargetState 예측 자체의 정밀도이므로
        # 이 정도 근사면 충분하다 (aiming_engine/lead_prediction.py의 1차 근사와 같은 발상).
        range_now = math.sqrt((true_x - PLATFORM_POS[0]) ** 2 + (true_y - PLATFORM_POS[1]) ** 2 + (true_z - PLATFORM_POS[2]) ** 2)
        tof = range_now / muzzle_velocity
        t_future = t + tof

        predicted = estimator.predict(t_future)
        pred_pos = predicted.position.to_array()

        gt_x, gt_y, gt_z, _ = compute_drone_linear_position(t_future)
        gt_pos = np.array([gt_x, gt_y, gt_z])

        pos_err = float(np.linalg.norm(pred_pos - gt_pos))
        ang_err = angle_between(pred_pos - PLATFORM_POS, gt_pos - PLATFORM_POS)

        if distance_to_leg_boundary(t) < TURN_MARGIN_S:
            turn_pos_err.append(pos_err)
            turn_ang_err.append(ang_err)
        else:
            steady_pos_err.append(pos_err)
            steady_ang_err.append(ang_err)

    def summarize(vals):
        if not vals:
            return {"mean": None, "p95": None, "n": 0}
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "n": len(arr)}

    return {
        "history_length": history_length,
        "velocity_smoothing_alpha": alpha,
        "steady_state": {
            "position_error_m": summarize(steady_pos_err),
            "angle_error_deg": summarize(steady_ang_err),
        },
        "near_turn": {
            "position_error_m": summarize(turn_pos_err),
            "angle_error_deg": summarize(turn_ang_err),
        },
    }


def main():
    cfg = load_config(os.path.join(project_root, "config", "aiming_engine.yaml"))
    muzzle_velocity = cfg.projectile.muzzle_velocity

    print("=" * 90)
    print(" SMASH 로드맵 2단계 리드각 정밀도 검증 (등속 직선 비행, 오라클 방법론)")
    print(f" 드론 속력: {DRONE_SPEED_MPS} m/s, 왕복 반경: {DRONE_PATH_HALF_LENGTH} m, 사거리: 35.0 m 고정")
    print(f" 탄속(muzzle_velocity, config/aiming_engine.yaml): {muzzle_velocity} m/s")
    print(f" 가정 위치노이즈(실측 아님): 횡방향 sigma={LATERAL_NOISE_STD_M} m, 사거리방향 sigma={RANGE_NOISE_STD_M} m")
    print("=" * 90)

    results = []
    rng_seed = 42
    for hl in HISTORY_LENGTHS:
        for alpha in SMOOTHING_ALPHAS:
            rng = np.random.default_rng(rng_seed)  # 설정 간 공정 비교를 위해 매번 같은 노이즈 시퀀스 사용
            r = run_one_config(hl, alpha, muzzle_velocity, rng)
            results.append(r)
            s = r["steady_state"]
            print(
                f" history_length={hl:>3} alpha={alpha:>4} | "
                f"정상비행 각도오차 평균 {s['angle_error_deg']['mean']:.4f} deg "
                f"(p95 {s['angle_error_deg']['p95']:.4f}) | "
                f"위치오차 평균 {s['position_error_m']['mean']:.4f} m "
                f"(p95 {s['position_error_m']['p95']:.4f})"
            )

    out_dir = os.path.join(project_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stage2_lead_accuracy_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "methodology": "TargetState 단독 오라클 검증, drone_motion_models.compute_drone_linear_position을 정답으로 사용",
            "fps": FPS,
            "duration_s": DURATION_S,
            "lateral_noise_std_m": LATERAL_NOISE_STD_M,
            "range_noise_std_m": RANGE_NOISE_STD_M,
            "note": "노이즈는 가정치이며 실측 아님. Level 3(실 하드웨어) 검증 전까지는 상대 비교 용도로만 사용.",
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print(f" 결과 저장: {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
