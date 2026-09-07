"""SMASH 대드론 지능형 사격 통제 시스템 통합 시뮬레이션 마스터 파이프라인 런처

기능:
콘솔 인코딩 깨짐 없이 학습부터 조준/격추 검증까지 한글 메뉴로 원클릭 실행 제공

경로 규칙:
- 모든 경로는 이 파일의 위치(PROJECT_ROOT)를 기준으로 잡는다. 저장소를 어느 폴더에
  clone 하든 그대로 동작한다.
- WSL 쪽 경로는 wslpath 로 자동 변환한다.
"""

import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WSL_DISTRO = "Ubuntu-24.04"

# 학습에 사용한 GA 튜닝 산출물. 시뮬레이터/브릿지가 공통으로 쓰는 정본 가중치.
DEFAULT_WEIGHTS = os.path.join(
    PROJECT_ROOT, "detector+tracker", "ga_results",
    "yolo11s_ga_final-3", "weights", "best.pt",
)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pause():
    input("\n계속하려면 엔터 키를 누르세요...")


def to_wsl_path(win_path):
    """윈도우 경로를 WSL 경로로 변환한다. 실패하면 None 을 돌려준다."""
    try:
        result = subprocess.run(
            ["wsl.exe", "-d", WSL_DISTRO, "wslpath", "-a", win_path],
            capture_output=True, text=True, timeout=20,
        )
        path = result.stdout.strip()
        return path or None
    except Exception:
        return None


def run_wsl(command, new_window=False, title="SMASH"):
    """WSL 안에서 명령을 실행한다. new_window 면 별도 콘솔 창으로 띄운다."""
    wsl_root = to_wsl_path(PROJECT_ROOT)
    if not wsl_root:
        print("  [문제] WSL 경로 변환에 실패했습니다.")
        print(f"         WSL2 배포판 '{WSL_DISTRO}' 이 설치·실행 중인지 확인하십시오.")
        return False
    full = f'cd "{wsl_root}" && {command}'
    if new_window:
        subprocess.Popen(
            f'start "{title}" wsl.exe -d {WSL_DISTRO} -- bash -c \'{full}\'',
            shell=True,
        )
    else:
        subprocess.run(
            ["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-c", full],
        )
    return True


def run_python(rel_path, *args):
    """저장소 안의 파이썬 스크립트를 실행한다. 없으면 이유를 설명하고 넘어간다."""
    script = os.path.join(PROJECT_ROOT, *rel_path)
    if not os.path.isfile(script):
        print(f"  [문제] 스크립트를 찾지 못했습니다: {os.path.join(*rel_path)}")
        return False
    subprocess.run([sys.executable, script, *args])
    return True


def header(title):
    clear_screen()
    print("=======================================================================")
    print(f" {title}")
    print("=======================================================================\n")


def print_banner():
    print("=======================================================================")
    print("   SMASH 대드론 지능형 사격 통제 시스템 통합 시뮬레이션 파이프라인")
    print("=======================================================================")
    print(f" 프로젝트 경로: {PROJECT_ROOT}\n")
    print("  [1] 학습:   YOLO11s-GA 탐지 모델 재학습 (Anti-UAV 데이터셋 필요)")
    print("  [2] 조준:   사수 1인칭 인터랙티브 조준 사격 시뮬레이터 (마우스/스페이스바)")
    print("  [3] 3D:     Gazebo 물리 시뮬레이션 + 실시간 교전 드라이버 + 스코프 HUD")
    print("  [4] 벤치:   E2E 100회 자동 사격 벤치마크 및 결과 영상 렌더링")
    print("  [5] 검증:   오프라인 오라클 검증 (ROS 2 / Gazebo 불필요)")
    print("  [6] 전체:   2 → 3 → 4 순차 자동 검증")
    print("  [0] 종료")
    print("=======================================================================")
    print(" ※ ROS 2 Jazzy + Gazebo Harmonic 기반 CIWS 포탑 시뮬레이터는")
    print("    run_smash_fcs.bat 을 사용하십시오.")
    print("=======================================================================")


def step_train():
    header("[학습] YOLO11s-GA 탐지 모델 재학습")
    dataset_yaml = os.path.join(
        PROJECT_ROOT, "datasets", "AntiUAV300_YOLO_VISIBLE", "anti_uav.yaml"
    )
    print("GA 탐색으로 찾은 하이퍼파라미터로 YOLO11s 를 재학습합니다.")
    print(f"학습 스크립트: detector+tracker/ga_code/train_yolo11s_ga_final.py")
    print(f"데이터셋 설정: {dataset_yaml}\n")

    if not os.path.isfile(dataset_yaml):
        print("  [주의] Anti-UAV 데이터셋이 없어 학습을 시작할 수 없습니다.")
        print("         라이선스상 데이터셋은 저장소에 포함하지 않습니다.")
        print("         배포처에서 내려받아 위 경로에 배치한 뒤 다시 실행하십시오.")
        print("         상세: detector+tracker/final/docs/DATASET.md\n")
        print("  이미 학습된 가중치는 저장소에 포함돼 있으며, 2~6번은 바로 실행됩니다:")
        print(f"    {os.path.relpath(DEFAULT_WEIGHTS, PROJECT_ROOT)}")
        pause()
        return

    print("  (GPU 학습이므로 시간이 오래 걸립니다.)\n")
    run_python(("detector+tracker", "ga_code", "train_yolo11s_ga_final.py"))
    pause()


def step_interactive():
    header("[조준] 사수 1인칭 인터랙티브 조준 사격 시뮬레이터")
    print("조작법: 마우스로 조준, READY 신호 시 스페이스바로 사격, Q 또는 ESC 로 종료\n")
    run_python(("scripts", "04_interactive_aiming_shooter_sim.py"))
    pause()


def step_gazebo():
    header("[3D] Gazebo 물리 시뮬레이션 및 실시간 조준/격추")
    print("1. Gazebo GUI 3D 월드 실행 중...")
    if not run_wsl("gz sim -r gazebo/worlds/aiming_test.sdf",
                   new_window=True, title="Gazebo GUI"):
        pause()
        return
    time.sleep(3)

    print("2. 3D 타겟 드론 실시간 회피 기동 드라이버 실행 중...")
    run_wsl("cd gazebo/scripts && python3 00_live_demo_drive.py",
            new_window=True, title="Gazebo Target Driver")
    time.sleep(1)

    print("3. SMASH 스마트 스코프 실시간 조준 및 격추 HUD 창 실행 중...")
    run_python(("scripts", "06_gazebo_interactive_shooter.py"))
    pause()


def step_benchmark():
    header("[벤치] E2E 100회 자동 사격 벤치마크 및 결과 영상 렌더링")
    print("1. 100회 연속 사격 데이터 수집 중...")
    if not run_wsl("cd gazebo/scripts && python3 02_e2e_simulation_runner.py"):
        pause()
        return
    print("\n2. 사격 분석 비디오 렌더링 중...")
    run_wsl("cd gazebo/scripts && python3 03_render_result_video.py")
    print(f"\n결과 비디오: {os.path.join(PROJECT_ROOT, 'gazebo', 'level2_result.mp4')}")
    pause()


def step_offline_validation():
    header("[검증] 오프라인 오라클 검증 (ROS 2 / Gazebo 불필요)")
    print("[1/2] 1단계 FCS 오프라인 검증 (좌표변환 · 거리추정 · 종단조준)...\n")
    run_python(("scripts", "08_stage1_fcs_offline_validation.py"))
    print("\n[2/2] 2단계 리드각 정확도 검증...\n")
    run_python(("scripts", "07_stage2_lead_accuracy_validation.py"))
    print(f"\n산출물: {os.path.join(PROJECT_ROOT, 'results')}")
    pause()


def step_all():
    header("[전체] 조준 → 3D 교전 → 벤치마크 순차 자동 검증")
    print("[1/3] 1인칭 조준 사격 시뮬레이터...\n")
    run_python(("scripts", "04_interactive_aiming_shooter_sim.py"))
    print("\n[2/3] Gazebo 스마트 스코프 조준/격추...\n")
    run_python(("scripts", "06_gazebo_interactive_shooter.py"))
    print("\n[3/3] Gazebo E2E 벤치마크 및 영상 렌더링...\n")
    run_wsl("cd gazebo/scripts && python3 02_e2e_simulation_runner.py "
            "&& python3 03_render_result_video.py")
    print("\n모든 파이프라인 검증이 완료되었습니다.")
    pause()


MENU = {
    "1": step_train,
    "2": step_interactive,
    "3": step_gazebo,
    "4": step_benchmark,
    "5": step_offline_validation,
    "6": step_all,
}


def main():
    while True:
        clear_screen()
        print_banner()
        choice = input("실행할 파이프라인 번호를 입력하세요 (0~6): ").strip()

        if choice == "0":
            print("\n프로그램을 종료합니다.")
            break
        action = MENU.get(choice)
        if action is None:
            print("\n  잘못된 입력입니다. 0~6 중에서 선택하십시오.")
            time.sleep(1.5)
            continue
        action()


if __name__ == "__main__":
    main()
