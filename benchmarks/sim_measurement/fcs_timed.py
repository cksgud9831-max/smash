#!/usr/bin/env python3
"""SmashFcsNode 를 단계별 실행 시간 계측과 함께 띄운다 (측정 전용).

런치가 띄운 FCS 노드를 죽이고 같은 --ros-args 로 이 스크립트를 대신 실행한다(run_case.sh PROFILE=1).
노드 코드는 바꾸지 않고 메서드를 감싸기만 한다.

출력:
    - 표준출력: 20초마다 단계별 호출 수·평균·최대·점유율, 입력 크기별 YOLO 호출 통계,
      조준 엔진 상태 요약(추적 신뢰도, 조준 오차, 거리, 표적 bbox 폭)
    - 환경변수 LAT_OUT 이 있으면 5초마다 JSON 으로 프레임별 시각과 격발 시각을 저장한다
      (analyze_latency.py 입력).
    - 환경변수 TRACK_DIAG_OUT 이 있으면 프레임마다 추적기 박스와 실측 표적 위치(촬영 시각으로 보간해
      같은 영상의 카메라 자세로 투영한 화소)를 JSON Lines 로 남긴다(track_diag.py 입력).
      추적기가 표적을 놓치고 배경에 붙는지 확인하는 용도다.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time

import rclpy
from drone_sim import smash_fcs_node as mod

STATS: dict = {}
SAMPLES: list = []  # (state, aim_ready, hit_prob, track_conf, follower_state, aim_err_mrad, dist, bbox_w)
YSTAT: dict = {}
LOCK = threading.Lock()
# frames: (wall_start, wall_end, image_stamp, solved)   shots: (wall, node_now, frame_ts)
LAT: dict = {"frames": [], "shots": []}
LAT_OUT = os.environ.get("LAT_OUT")
TRACK_DIAG_OUT = os.environ.get("TRACK_DIAG_OUT")
_DIAG_FILE = open(TRACK_DIAG_OUT, "w", buffering=1) if TRACK_DIAG_OUT else None


def timed(name, fn):
    def w(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t
            with LOCK:
                s = STATS.setdefault(name, [0, 0.0, 0.0])
                s[0] += 1
                s[1] += dt
                s[2] = max(s[2], dt)
    return w


def _wrap_on_image(fn):
    def w(self, msg):
        ws = time.time()
        before = self._frames_solved
        try:
            return fn(self, msg)
        finally:
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with LOCK:
                LAT["frames"].append((ws, time.time(), st, self._frames_solved > before))
            if _DIAG_FILE is not None:
                try:
                    _write_track_diag(self, st)
                except Exception as e:  # 진단 기록 실패로 FCS 가 죽으면 안 된다
                    print(f"track_diag write failed: {e!r}", flush=True)
    return w


def _vec(v) -> list[float]:
    return [round(float(x), 4) for x in v]


def _estimate_snapshot(manager, frame, out, admitted: bool) -> dict:
    """조준 엔진이 이 영상으로 무엇을 추정했는지(측정 위치, 추정 위치·속도, 리드 점). 모두 base(=월드) 프레임."""
    snap = {"t_frame": float(frame.timestamp), "admitted": bool(admitted)}
    if admitted:
        snap["meas_pos"] = _vec(manager._coordinate_transform.image_to_world(frame).to_array())
    est = manager._target_state.predict(frame.timestamp)
    snap["est_pos"] = _vec(est.position.to_array())
    snap["est_vel"] = _vec(est.velocity.to_array())
    snap["range_m"] = round(float(frame.laser_range_m), 3)
    sol = out.aim_solution
    if sol is not None:
        snap["lead_pos"] = _vec(sol.lead_point_world.to_array())
        snap["tof"] = round(float(sol.time_of_flight), 5)
    return snap


def _write_track_diag(node, stamp: float) -> None:
    rec = {"t": stamp, "state": node._state_label, "rel": node._reliability_state, "bbox": None, "gt_px": None}
    if node._last_bbox is not None:
        rec["bbox"] = [round(float(v), 1) for v in node._last_bbox]
    fc = node._fire_control
    if fc is not None:
        gt, age = fc.ground_truth_at(stamp)
        if gt is not None:
            px = node._project_world_point_to_pixel(gt)
            rec["gt_px"] = None if px is None else [float(px[0]), float(px[1])]
            rec["gt_age"] = age
            rec["gt_y"] = round(float(gt[1]), 3)
    if fc is not None and gt is not None:
        rec["gt_pos"] = _vec(gt)
    if node._last_camera_pose is not None:
        rec["cam_pos"] = _vec(node._last_camera_pose[:3, 3])
    est = getattr(node, "_diag_est", None)
    if est is not None and est.get("t_frame") == stamp:  # 이 영상으로 만든 추정만
        rec.update(est)
    rel = getattr(node, "_diag_rel", None)
    if rel is not None:
        rec.update(rel)
    last = getattr(node._tracker, "_diag_last", None)
    rec["yolo"] = rec["recovery"] = rec["score"] = None
    if last is not None:
        rec["yolo"], rec["recovery"], rec["score"] = last
    _DIAG_FILE.write(json.dumps(rec) + chr(10))


mod.SmashFcsNode._on_image = _wrap_on_image(mod.SmashFcsNode._on_image)
for _m in ("_on_image", "_on_laser", "_on_camera_info", "_publish_command", "_fire_control_tick",
           "_render", "_lookup_camera_pose", "_on_ground_truth_pose"):
    # Simulator control was split into turret_command_node.py. Keep this
    # profiler compatible with both layouts: missing simulator-only callbacks
    # are not part of the FCS image-to-aim path and can simply be skipped.
    if hasattr(mod.SmashFcsNode, _m):
        setattr(mod.SmashFcsNode, _m, timed(_m, getattr(mod.SmashFcsNode, _m)))


def _instrument(node) -> None:
    tracker_update = timed("tracker.update", node._tracker.update)

    def update_and_note(*a, **k):
        r = tracker_update(*a, **k)
        # 표적이 없으면 None 을 돌려준다(측정 도구가 노드를 죽이면 안 된다).
        node._tracker._diag_last = None if r is None else (
            bool(getattr(r, "used_yolo", False)), bool(r.is_recovery_event), round(float(r.score), 3))
        return r
    node._tracker.update = update_and_note

    rel_update = timed("reliability.update", node._reliability_engine.update)

    def rel_update_and_note(*a, **k):
        # 신뢰도 상태가 왜 그 값인지(품질 점수, 관문 실패 사유, STABLE 후보 연속 프레임 수).
        r = rel_update(*a, **k)
        te = r.temporal
        node._diag_rel = {
            "rel_q": None if te.current_quality_score is None else round(float(te.current_quality_score), 3),
            "rel_gate": list(te.current_gate_reasons),
            "rel_cand": int(te.stable_candidate_frames),
            "rel_why": r.transition.reason,
            "rel_stale": [k for k in ("suspect", "confirmed") if getattr(r.stale, k, False)],
            "rel_elig": bool(r.evidence.stable_eligible),
            "rel_ev": list(r.evidence.reasons),
        }
        return r
    node._reliability_engine.update = rel_update_and_note

    orig_predict = timed("yolo.predict", node._tracker._model.predict)

    def predict(*a, **k):
        r = orig_predict(*a, **k)
        key = (k.get("imgsz"), tuple(k["source"].shape[:2]) if "source" in k else None)
        b = r[0].boxes
        c = float(b.conf.max()) if b is not None and len(b) else 0.0
        with LOCK:
            s = YSTAT.setdefault(str(key), [0, 0, 0.0])
            s[0] += 1
            s[1] += int(c > 0)
            s[2] = max(s[2], c)
        return r
    node._tracker._model.predict = predict

    def wrap_mgr(m):
        f = timed("aiming.update", m.update)

        def upd(frame, *a, **k):
            out = f(frame, *a, **k)
            if _DIAG_FILE is not None:
                try:
                    node._diag_est = _estimate_snapshot(m, frame, out, k.get("measurement_admitted", True))
                except Exception as e:  # 진단 실패로 FCS 가 죽으면 안 된다
                    node._diag_est = {"est_error": repr(e)}
            with LOCK:
                SAMPLES.append((out.state.value, bool(out.aim_ready), float(out.hit_probability),
                                float(frame.tracking_confidence), str(frame.follower_state),
                                float(out.debug.get("aim_error_mrad", -1)), float(out.debug.get("distance_m", -1)),
                                float(frame.bbox[2] - frame.bbox[0])))
            return out
        m.update = upd
        return m

    # FCS 는 SEARCH 재진입 때 AimingManager 를 새로 만든다. 새 인스턴스도 감싸도록 팩토리를 바꾼다.
    orig_cls = node._AimingManager
    node._AimingManager = lambda *a, **k: wrap_mgr(orig_cls(*a, **k))
    wrap_mgr(node._aiming_manager)

    node._bridge_cv.imgmsg_to_cv2 = timed("imgmsg_to_cv2", node._bridge_cv.imgmsg_to_cv2)
    node._range_estimator.estimate = timed("range.estimate", node._range_estimator.estimate)
    if node._fire_control is not None:
        mf = timed("fc.maybe_fire", node._fire_control.maybe_fire)

        def maybe_fire(**kw):
            shot = mf(**kw)
            if shot is not None:
                with LOCK:
                    # 세 번째 값은 격발에 쓴 마지막 영상의 촬영 시각이다(kw["now"] 는 이제 격발 시각).
                    LAT["shots"].append((time.time(), node.get_clock().now().nanoseconds * 1e-9,
                                         node._last_frame_timestamp))
            return shot
        node._fire_control.maybe_fire = maybe_fire
        node._fire_control.update = timed("fc.update", node._fire_control.update)


def _report_loop() -> None:
    t0 = time.perf_counter()
    while True:
        time.sleep(20)
        el = time.perf_counter() - t0
        with LOCK:
            if YSTAT:
                print("  yolo " + "  ".join(f"{k}: n={v[0]} det={v[1]} maxc={v[2]:.2f}" for k, v in YSTAT.items()),
                      flush=True)
                YSTAT.clear()
            print(f"--- {el:.1f}s wall ---", flush=True)
            for k, (n, tot, mx) in sorted(STATS.items(), key=lambda kv: -kv[1][1]):
                print(f"{k:24s} n={n:6d} rate={n / el:6.1f}/s mean={1000 * tot / max(n, 1):7.2f}ms "
                      f"max={1000 * mx:7.1f}ms busy={100 * tot / el:5.1f}%", flush=True)
            STATS.clear()
            if SAMPLES:
                states: dict = {}
                fol: dict = {}
                for x in SAMPLES:
                    states[x[0]] = states.get(x[0], 0) + 1
                    fol[x[4]] = fol.get(x[4], 0) + 1

                def med(i):
                    return statistics.median(x[i] for x in SAMPLES)
                print(f"  engine n={len(SAMPLES)} states={states} ready={sum(x[1] for x in SAMPLES)} "
                      f"hitP med={med(2):.3f} min={min(x[2] for x in SAMPLES):.3f} "
                      f"trackConf med={med(3):.3f} min={min(x[3] for x in SAMPLES):.3f} follower={fol} "
                      f"aimErr med={med(5):.2f}mrad dist med={med(6):.1f}m bboxW med={med(7):.1f}px", flush=True)
                SAMPLES.clear()
        t0 = time.perf_counter()


def _dump_loop() -> None:
    while True:
        time.sleep(5)
        with LOCK:
            data = json.dumps(LAT)
        with open(LAT_OUT + ".tmp", "w") as f:
            f.write(data)
        os.replace(LAT_OUT + ".tmp", LAT_OUT)


def main() -> None:
    rclpy.init(args=sys.argv)
    node = mod.SmashFcsNode()
    _instrument(node)
    threading.Thread(target=_report_loop, daemon=True).start()
    if LAT_OUT:
        threading.Thread(target=_dump_loop, daemon=True).start()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node._tracker.close()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
