"""benchmarks/sim_measurement 의 ROS 비의존 부분(정렬 계산, 지연 분석, 결과 요약) 시험."""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "sim_measurement"))

from alignment import axis_offset_from_probe  # noqa: E402
from analyze_latency import analyze_data  # noqa: E402
from summarize_runs import summarize_case  # noqa: E402


@pytest.mark.parametrize("a,b", [(12.0, 5.0), (-30.0, 0.0), (0.0, 20.0), (3.0, -40.0)])
def test_axis_offset_recovers_signed_error_from_magnitudes(a, b):
    k = 1500.0  # px/rad
    step = 20.0 / k
    e0 = math.hypot(a, b)
    ep = math.hypot(a + k * step, b)
    d = axis_offset_from_probe(e0, ep, step, k)
    assert a + k * d == pytest.approx(0.0, abs=1e-9)


def _clock(rtf=0.5, t0=1000.0, n=2000):
    wall = t0 + np.arange(n) * 0.005
    return [[float(w), float((w - t0) * rtf)] for w in wall]


def test_analyze_data_recovers_known_latencies():
    rtf = 0.5
    clk = {"clock": _clock(rtf), "ann": [], "fire": []}
    t0 = 1000.0
    # 모의 시각 1.0 s 에 찍힌 프레임(실제 시각 t0+2.0) 을 30 ms 뒤 받아 20 ms 처리
    frames, shots = [], []
    for i in range(20):
        st = 1.0 + i * 0.1
        cap_wall = t0 + st / rtf
        frames.append([cap_wall + 0.030, cap_wall + 0.050, st, True])
        shots.append([cap_wall + 0.060, st + 0.060 * rtf, st])
    out = analyze_data(clk, {"frames": frames, "shots": shots})
    assert out["rtf"] == pytest.approx(rtf, rel=1e-6)
    assert out["capture_to_fcs_start_wall_ms"]["med"] == pytest.approx(30.0, abs=0.1)
    assert out["fcs_processing_wall_ms"]["med"] == pytest.approx(20.0, abs=0.1)
    assert out["capture_to_aim_wall_ms"]["med"] == pytest.approx(50.0, abs=0.1)
    assert out["capture_to_aim_sim_ms"]["med"] == pytest.approx(25.0, abs=0.1)
    assert out["capture_to_fire_wall_ms"]["med"] == pytest.approx(60.0, abs=0.1)
    assert out["node_clock_lag_sim_ms"]["med"] == pytest.approx(0.0, abs=0.1)


def test_analyze_data_ignores_unsolved_frames():
    clk = {"clock": _clock(), "ann": [], "fire": []}
    out = analyze_data(clk, {"frames": [[1002.1, 1002.2, 1.0, False]], "shots": []})
    assert "fcs_processing_wall_ms" not in out


def test_summarize_case_reads_probe_json_and_label(tmp_path):
    probe = {
        "target": [33.0, 0.0, 12.5], "acquire_s": 3.2, "align_final_px": 1.0, "fx": 5333.0, "camera_px": 640,
        "state_frac": {"READY": 0.9, "SEARCH": 0.02}, "shots": 10,
        "verdicts": {"HIT": 9, "MISS": 1}, "miss_m": {"mean": 0.12, "max": 0.21},
    }
    path = tmp_path / "f0.12_35m_r2.json"
    path.write_text(json.dumps(probe), encoding="utf-8")
    (tmp_path / "f0.12_35m_r2.prof.log").write_text(
        "_on_image                n=   180 rate=   9.0/s mean=  20.00ms\n"
        "  engine n=10 bboxW med=74.0px\n", encoding="utf-8")
    row = summarize_case(str(path))
    assert row["fov_deg"] == pytest.approx(6.9, abs=0.05)
    assert row["rep"] == 2
    assert row["range_m"] == pytest.approx(math.dist((0, 0, 0.7), (33.0, 0.0, 12.5)), abs=0.1)
    assert (row["hits"], row["judged"]) == (9, 10)
    assert row["bbox_px"] == 74.0
    assert row["fcs_hz"] == 9.0
    assert row["rtf"] is None  # 지연 기록이 없으면 비워 둔다


def _simulate_tracking(omega, servo_lag=0.1, true_lag=None, frame_dt=0.08, age=0.04, ff_dt=1 / 30, T=10.0,
                       el_offset=0.02, move_at=2.0):
    """포탑 = 순수 지연(true_lag 뒤 명령값에 도달) + 앙각 상수 차이, 영상 = frame_dt 마다 age 만큼 묵어 도착.
    리드는 move_at 초까지 멈춰 있다가 등각속도 omega 로 움직인다. (정상상태 최대 오차, 전체 최대 초과) 반환."""
    from tracking import LeadTracker

    true_lag = servo_lag if true_lag is None else true_lag
    tr = LeadTracker(servo_lag_s=servo_lag, step_s=ff_dt)
    tr.set_command(0.0, 0.25, 0.1)
    lead = lambda t: (0.3 + omega * max(0.0, t - move_at), 0.1)
    to_dir = lambda az, el: (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    bore = lambda t: (tr.command_at(t - true_lag)[0], tr.command_at(t - true_lag)[1] + el_offset)
    steady, overall, t, next_frame, next_ff = [], [], 0.0, 0.0, ff_dt
    while t < T:
        t = round(t + 0.001, 6)
        if t >= next_frame + age:  # next_frame 에 찍힌 영상이 지금 도착
            st = next_frame
            tr.correction(t, st, st + age, to_dir(*lead(st + age)), to_dir(*bore(st)))
            next_frame += frame_dt
        if t >= next_ff:
            tr.feedforward(t)
            next_ff += ff_dt
        if t > 1.0:
            err = abs(bore(t)[0] - lead(t)[0])
            overall.append(err)
            if t > T / 2:
                steady.append(err)
    return max(steady), max(overall)


@pytest.mark.parametrize("omega", [0.14, -0.3, 0.5])
def test_lead_tracker_follows_moving_lead_with_turret_lag(omega):
    # 35 m 에서 5 m/s 로 가로지르는 표적 ~ 0.14 rad/s. 명령이 30 Hz 계단이므로 정상상태에서 남는 오차는
    # 계단 반 칸(omega * dt / 2) 수준이어야 한다. 지연(서보 0.1 s + 영상 0.04 s) 자체는 남지 않아야 한다.
    dt = 1 / 30
    steady, _ = _simulate_tracking(omega, ff_dt=dt)
    assert steady < abs(omega) * dt / 2 * 1.2 + 1e-4


@pytest.mark.parametrize("true_lag", [0.05, 0.3])
def test_lead_tracker_is_stable_when_turret_lag_is_misjudged(true_lag):
    # 실측(2026-09-21): 예전 스미스 예측기 방식은 실제 지연이 가정보다 길자 1.8배 과속해 표적을 놓쳤다.
    # 지금 방식은 지연을 잘못 알아도 발산하지 않고, 오프셋 학습이 정상상태 오차를 없애야 한다.
    omega = 0.14
    steady, overall = _simulate_tracking(omega, servo_lag=0.1, true_lag=true_lag, T=20.0)
    assert steady < omega / 30 / 2 * 1.2 + 5e-4
    # 출발 순간의 과도 오차는 "지연 차이 + 영상 주기" 동안 움직인 만큼을 크게 넘지 않는다.
    assert overall < omega * (abs(true_lag - 0.1) + 0.3) + 2e-3


def test_track_diag_flags_box_stuck_off_target():
    from track_diag import summarize

    recs = []
    for i in range(20):
        gt_x = 300.0 + 10.0 * i  # 표적이 화면에서 오른쪽으로 이동
        box_x = gt_x if i < 10 else 390.0  # 10번째 프레임부터 박스가 제자리에 멈춤(배경 고착)
        recs.append({"t": i * 0.1, "rel": "STABLE", "bbox": [box_x - 20, 300, box_x + 20, 340],
                     "gt_px": [gt_x, 320.0], "gt_y": 0.1 * i})
    s = summarize(recs)
    assert s["tracked_frames"] == 20
    # 박스 크기(40 px)를 넘게 벌어진 프레임만 이탈: i = 14..19 (거리 50..100 px)
    assert s["off_target_frac"] == pytest.approx(6 / 20)
    assert s["off_target_while_stable_frac"] == pytest.approx(6 / 20)
    assert len(s["episodes"]) == 1 and s["episodes"][0]["frames"] == 6


def test_stall_tracker_measures_longest_gap_and_counts_stalls():
    from simclock import StallTracker

    st = StallTracker(stall_threshold_s=1.0)
    wall = 0.0
    for i in range(10):  # 정상: 0.1 s 마다 갱신
        st.update(i * 0.05, wall)
        wall += 0.1
    st.update(0.45, wall + 3.0)  # 같은 값은 진행이 아니다
    assert st.check(wall + 3.0) == pytest.approx(3.1)
    st.update(0.5, wall + 3.0)  # 3.1 s 멈춤 뒤 재개
    assert st.max_gap_s == pytest.approx(3.1)
    assert st.stalls == 1
    st.reset_stats()
    st.update(0.55, wall + 3.1)
    assert st.max_gap_s == pytest.approx(0.1) and st.stalls == 0


def test_alignment_rejects_implausible_probe_and_clamps_correction():
    from alignment import clamp_correction, probe_is_plausible

    k = 3189.0
    assert probe_is_plausible(3.0, 9.0, step_px=8.0)
    assert not probe_is_plausible(3.0, 145.0, step_px=8.0)  # 2026-09-22 실측 사례
    # -410 mrad 같은 보정은 오차 3 px 에서 나올 수 없다: (3 + 10) / k 로 잘린다
    assert clamp_correction(-0.41, e0=3.0, k=k) == pytest.approx(-13.0 / k)
    assert clamp_correction(0.001, e0=3.0, k=k) == 0.001


def test_est_diag_recovers_known_lag_and_reversal_bins():
    from est_diag import analyze, summarize

    def pos(t):  # 시선(+x)에 수직으로 5 m/s 등속 왕복, 반폭 3 m (한 방향 1.2 s)
        period = 4 * 3.0 / 5.0
        ph = (t % period) / period
        y = 12 * ph if ph < 0.25 else (3 - 12 * (ph - 0.25) if ph < 0.75 else -3 + 12 * (ph - 0.75))
        return [33.0, y, 12.5]

    recs, tof = [], 0.04
    for i in range(600):  # 20 Hz, 30 s
        t = i * 0.05
        v = [0.0, 5.0 if (t % 2.4) < 0.6 or (t % 2.4) >= 1.8 else -5.0, 0.0]
        recs.append({"t": t, "gt_pos": pos(t), "cam_pos": [0.0, 0.0, 0.7], "meas_pos": pos(t),
                     "est_pos": pos(t - 0.05), "est_vel": v, "lead_pos": pos(t + tof), "tof": tof})
    s = summarize(analyze(recs))
    assert s["moving_lag_ms"]["meas"] == pytest.approx(0.0, abs=1.0)
    assert s["moving_lag_ms"]["est"] == pytest.approx(50.0, abs=2.0)
    assert s["moving_lag_ms"]["lead"] == pytest.approx(0.0, abs=1.0)
    assert s["vel_ratio"] == pytest.approx(1.0, abs=0.05)
    # 방향 전환 뒤 0.2 s 이후 구간은 모두 같은 50 ms 늦음
    late_bins = [b for b in s["bins"] if b["since_rev_s"][0] >= 0.2 and b["n"] > 0]
    assert late_bins and all(abs(b["est_lag_ms"] - 50.0) < 3.0 for b in late_bins)


def test_resetting_cv_recovers_faster_after_reversal():
    from aiming_engine.target_state import ConstantVelocityEstimator
    from aiming_engine.types import Vector3
    from estimator_replay import ResettingCV

    def y(t):  # 5 m/s 로 가다가 t=2 s 에 방향 전환
        return 5.0 * t if t < 2.0 else 10.0 - 5.0 * (t - 2.0)

    base, reset = ConstantVelocityEstimator(20, 1e-3), ResettingCV(20, 0.25, 2, 3)
    errs = {"base": [], "reset": []}
    for i in range(60):  # 20 Hz
        t = i * 0.05
        z = Vector3(33.0, y(t), 12.5)
        base.update(z, t)
        reset.update(z, t)
        if 2.2 <= t <= 2.6:  # 전환 뒤 0.2~0.6 s, 0.08 s 앞 예측
            errs["base"].append(abs(base.predict(t + 0.08).position.y - y(t + 0.08)))
            errs["reset"].append(abs(reset.predict(t + 0.08).position.y - y(t + 0.08)))
    assert reset.resets >= 1
    assert max(errs["reset"]) < 0.5 * max(errs["base"])
    # 전환 전 등속 구간에서는 리셋하지 않는다
    steady = ResettingCV(20, 0.25, 2, 3)
    for i in range(40):
        steady.update(Vector3(33.0, 5.0 * i * 0.05, 12.5), i * 0.05)
    assert steady.resets == 0
