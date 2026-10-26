"""Windows 런처 .bat 파일 형식 시험.

UTF-8(chcp 65001)로 실행되는 배치 파일에서 cmd.exe 는
  - LF 만 있는 줄이 섞이면 줄 위치를 잘못 계산하고,
  - 한글 등 비ASCII 문자로 끝나는 줄을 읽은 뒤 다음 줄 중간부터 실행한다
    ('�' is not recognized ... 오류, 입력 검사 줄이 깨져 무한 재입력).
두 조건을 파일 단위로 막는다.
"""

import os

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
BAT_FILES = ["simulation/run_smash_fcs.bat"]


def _read(rel):
    with open(os.path.join(ROOT, rel), "rb") as f:
        return f.read()


@pytest.mark.parametrize("rel", BAT_FILES)
def test_bat_uses_crlf_only(rel):
    data = _read(rel)
    lf_only = [i for i, line in enumerate(data.split(b"\n")[:-1], 1) if not line.endswith(b"\r")]
    assert not lf_only, f"{rel}: LF-only line endings at lines {lf_only}"


@pytest.mark.parametrize("rel", BAT_FILES)
def test_bat_lines_end_with_ascii(rel):
    lines = _read(rel).split(b"\r\n")
    bad = [i for i, line in enumerate(lines, 1) if line and line[-1] > 0x7F]
    assert not bad, f"{rel}: lines ending with a non-ASCII character: {bad}"


def _block(label):
    text = _read("simulation/run_smash_fcs.bat").decode("utf-8")
    start = text.index(f":{label}")
    end = text.index("\r\ngoto after", start)
    return text[start:end]


def test_launcher_has_no_manual_fire_mode():
    text = _read("simulation/run_smash_fcs.bat").decode("utf-8")
    assert "do_run_fire_manual" not in text
    for name in ("SMASH_AIM_ASSIST", "SMASH_FIRE_MODE"):
        assert name not in text


def test_auto_fire_launcher_enables_fire_control_and_plain_disables_it():
    assert "SMASH_ENABLE_FIRE_CONTROL=true" in _block("do_run_fire_auto")
    assert "SMASH_ENABLE_FIRE_CONTROL=false" in _block("do_run_plain")


def test_operator_emulation_uses_short_fcs_lag_but_can_be_overridden():
    text = _read("simulation/run_smash_fcs.bat").decode("utf-8")
    # 두 ENVPFX(메뉴 진입, 6번 선택 시) 모두 값을 WSL 로 넘긴다
    assert text.count("SMASH_MOTION_LAG_S=%MOTION_LAG_S%") == 2
    # 사용자가 미리 set MOTION_LAG_S=0.17 로 정하면 덮어쓰지 않는다
    assert 'if not defined MOTION_LAG_S if /i "%OPERATOR_FOLLOW%"=="true" set "MOTION_LAG_S=0.02"' in text
    assert 'if not defined MOTION_LAG_S set "MOTION_LAG_S=0.17"' in text
    # 코드·yaml 기본값은 그대로
    yaml = open(os.path.join(ROOT, "simulation", "ciws_turret_aerial_object_detection-main", "drone_sim",
                             "config", "smash_fcs.yaml"), encoding="utf-8").read()
    assert "motion_measurement_lag_s: 0.17" in yaml


def test_launcher_defaults_to_manual_single_shot_trigger_with_override():
    text = _read("simulation/run_smash_fcs.bat").decode("utf-8")
    assert text.count("SMASH_FIRE_TRIGGER=%FIRE_TRIGGER%") == 2
    assert 'if not defined FIRE_TRIGGER set "FIRE_TRIGGER=manual"' in text
    assert "do_run_fire_manual" not in text  # 예전 수동 격발 메뉴 항목은 되살리지 않는다


def test_manual_trigger_turns_the_kill_effect_on_by_default_and_it_can_be_overridden():
    text = _read("simulation/run_smash_fcs.bat").decode("utf-8")
    assert text.count("SMASH_DEMO_KILL=%DEMO_KILL%") == 2
    assert 'if not defined DEMO_KILL if /i "%FIRE_TRIGGER%"=="manual" set "DEMO_KILL=true"' in text
    assert 'if not defined DEMO_KILL set "DEMO_KILL=false"' in text


def _vel_gate_by_motion(tmp_path, user_override=None):
    """런처의 :set_vel_gate 를 cmd 로 직접 돌려 방식별 값을 얻는다."""

    import subprocess
    import sys

    text = _read("simulation/run_smash_fcs.bat").decode("utf-8").replace("\r\n", "\n")
    sub = text[text.index("\n:set_vel_gate\n") + 1:text.index("\n:select_camera_fov\n")]
    harness = (
        "@echo off\nsetlocal enabledelayedexpansion\n"
        'set "VEL_DISAGREE_USER=%VEL_DISAGREE_MPS%"\n'
        "for %%M in (none straight cross maneuver approach) do (\n"
        '  set "TARGET_MOTION=%%M"\n  call :set_vel_gate\n  echo %%M !VEL_GATE!\n)\n'
        "exit /b 0\n\n" + sub
    )
    path = tmp_path / "vel_harness.bat"
    path.write_bytes(harness.replace("\n", "\r\n").encode("ascii"))
    env = {k: v for k, v in os.environ.items() if k != "VEL_DISAGREE_MPS"}
    if user_override is not None:
        env["VEL_DISAGREE_MPS"] = user_override
    out = subprocess.run(["cmd", "/c", str(path)], capture_output=True, text=True, env=env).stdout
    return dict(line.split() for line in out.replace("\r", "").split("\n") if line.strip())


def test_velocity_gate_is_relaxed_only_for_the_figure_8_and_passed_to_wsl():
    text = _read("simulation/run_smash_fcs.bat").decode("utf-8")
    # 두 ENVPFX(메뉴 진입, 6번 선택 시) 모두 값을 WSL 로 넘기고, 넘기기 직전에 방식에 맞춰 계산한다
    assert text.count("SMASH_MOTION_VEL_DISAGREE_MPS=%VEL_GATE%") == 2
    assert text.count("call :set_vel_gate") == 2
    # 코드·yaml 기본값은 그대로
    yaml = open(os.path.join(ROOT, "simulation", "ciws_turret_aerial_object_detection-main", "drone_sim",
                             "config", "smash_fcs.yaml"), encoding="utf-8").read()
    assert "motion_ready_max_velocity_disagreement_mps: 0.75" in yaml


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe 가 필요하다")
def test_velocity_gate_values_by_motion_and_user_override(tmp_path):
    assert _vel_gate_by_motion(tmp_path) == {
        "none": "0.75", "straight": "0.75", "cross": "0.75", "maneuver": "1.5", "approach": "0.75"}
    # 사용자가 미리 set VEL_DISAGREE_MPS=... 로 정하면 모든 방식에 그 값을 쓴다
    assert set(_vel_gate_by_motion(tmp_path, "0.9").values()) == {"0.9"}
