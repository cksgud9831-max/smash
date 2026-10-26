"""Operator-emulation pointing math (ROS-independent)."""

import math
import os
import sys

import numpy as np

_SIM_PKG = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "simulation",
    "ciws_turret_aerial_object_detection-main", "drone_sim",
))
sys.path.insert(0, _SIM_PKG)

from drone_sim.smash_fcs.operator_aim import LeadPredictor, world_to_base  # noqa: E402


def test_world_to_base_inverts_spawn_pose():
    yaw, tr = 0.7, np.array([1.0, -2.0, 0.05])
    base = np.array([30.0, 4.0, 8.0])
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.array([c * base[0] - s * base[1], s * base[0] + c * base[1], base[2]]) + tr
    assert np.allclose(world_to_base(world, tr, yaw), base)


def test_lead_predictor_estimates_velocity_and_leads():
    p = LeadPredictor(smoothing=0.5)
    for i in range(40):  # 20 Hz, 2.78 m/s in +y
        p.update([33.0, i * 0.05 * 2.78, 12.5], i * 0.05)
    assert abs(p.vel[1] - 2.78) < 0.05
    ahead = p.aim_point(0.1)
    assert abs(ahead[1] - (p.pos[1] + 0.278)) < 0.01


def test_teleport_does_not_become_a_huge_velocity():
    p = LeadPredictor(smoothing=0.5)
    for i in range(10):
        p.update([33.0, i * 0.1, 12.5], i * 0.05)
    p.update([33.0, -15.0, 12.5], 0.5)  # 끝점에서 반대쪽으로 순간 이동
    assert np.linalg.norm(p.vel) == 0.0


def test_no_aim_point_before_first_pose_and_after_reset():
    p = LeadPredictor()
    assert p.aim_point(0.1) is None
    p.update([1, 2, 3], 0.0)
    p.reset()
    assert p.aim_point(0.1) is None


def test_shift_left_moves_perpendicular_to_line_of_sight():
    from drone_sim.smash_fcs.operator_aim import shift_left

    out = shift_left([30.0, 0.0, 5.0], 0.5)
    assert np.allclose(out, [30.0, 0.5, 5.0])
    out = shift_left([0.0, 30.0, 5.0], 0.5)  # 시선이 +y 면 왼쪽은 -x
    assert np.allclose(out, [-0.5, 30.0, 5.0])
    assert np.allclose(shift_left([0.0, 0.0, 5.0], 0.5), [0.0, 0.0, 5.0])


def test_lead_offset_is_angle_between_lead_and_bbox_directions():
    from drone_sim.smash_fcs.operator_aim import lead_offset

    d_pan, d_tilt = lead_offset([math.cos(0.1), math.sin(0.1), 0.0], [1.0, 0.0, 0.0])
    assert abs(d_pan - 0.1) < 1e-9 and abs(d_tilt) < 1e-9
    d_pan, d_tilt = lead_offset([1.0, 0.0, math.tan(0.02)], [1.0, 0.0, 0.0])
    assert abs(d_pan) < 1e-9 and abs(d_tilt - 0.02) < 1e-9
    assert lead_offset(None, [1, 0, 0]) is None
    assert lead_offset([0, 0, 1], [1, 0, 0]) is None  # 수직 방향은 방위각이 정해지지 않는다


def test_rotate_aim_keeps_range_and_adds_angles():
    from drone_sim.smash_fcs.operator_aim import rotate_aim

    p = np.array([30.0, 5.0, 8.0])
    q = rotate_aim(p, 0.05, -0.02)
    assert abs(np.linalg.norm(q) - np.linalg.norm(p)) < 1e-9
    assert abs(math.atan2(q[1], q[0]) - (math.atan2(p[1], p[0]) + 0.05)) < 1e-9
    assert abs(math.atan2(q[2], math.hypot(q[0], q[1])) - (math.atan2(p[2], math.hypot(p[0], p[1])) - 0.02)) < 1e-9
    assert np.allclose(rotate_aim(p, 0.0, 0.0), p)
