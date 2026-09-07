"""SMASH 대드론 지능형 사격 통제 시스템 통합 시뮬레이션 마스터 파이프라인 런처

기능:
콘솔 인코딩 깨짐 없이 1단계부터 5단계까지 완벽한 한글 메뉴와 원클릭 실행 제공
"""

import os
import sys
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print("=======================================================================")
    print("   SMASH 대드론 지능형 사격 통제 시스템 통합 시뮬레이션 파이프라인")
    print("=======================================================================")
    print(f" 프로젝트 경로: {PROJECT_ROOT}\n")
    print("  [1] 1단계: 가상 드론 데이터셋 자동 수집 및 YOLO11s 파인튜닝")
    print("  [2] 2단계: Webots 3D 월드 및 실시간 스마트 스코프 HUD 시뮬레이터")
    print("  [3] 3단계: 사수 1인칭 인터랙티브 조준 사격 시뮬레이터 (마우스/스페이스바)")
    print("  [4] 4단계: Gazebo 3D 물리 시뮬레이션 및 실시간 교전 드라이버")
    print("  [5] 5단계: E2E 100회 자동 사격 벤치마크 및 종합 결과 영상 렌더링")
    print("  [6] 전체 4대 파이프라인 순차 자동 검증")
    print("  [0] 종료")
    print("=======================================================================")


def step1_finetune():
    clear_screen()
    print("=======================================================================")
    print(" [1단계] 가상 3D 드론 데이터셋 자동 수집 및 YOLO11s 파인튜닝 시작")
    print("=======================================================================\n")
    script_path = os.path.join(PROJECT_ROOT, "scripts", "05_auto_collect_and_finetune.py")
    subprocess.run([sys.executable, script_path])
    input("\n계속하려면 엔터 키를 누르세요...")


def step2_webots():
    clear_screen()
    print("=======================================================================")
    print(" [2단계] Webots 3D 교전 월드 및 실시간 스마트 스코프 HUD 실행")
    print("=======================================================================\n")
    wbt_path = os.path.join(PROJECT_ROOT, "webots", "worlds", "aiming_world.wbt")
    print(f"월드 파일: {wbt_path}")

    # Webots 실행 경로 탐색
    webots_paths = [
        "webots",
        r"C:\Program Files\Webots\msys64\mingw64\bin\webotsw.exe",
        rf"C:\Users\{os.environ.get('USERNAME', '')}\AppData\Local\Programs\Webots\msys64\mingw64\bin\webotsw.exe",
    ]
    
    launched = False
    for p in webots_paths:
        try:
            subprocess.Popen([p, wbt_path], shell=True)
            print("Webots 월드가 정상적으로 실행되었습니다.")
            launched = True
            break
        except Exception:
            continue

    if not launched:
        print("\nWebots 실행 프로그램을 자동으로 찾지 못했습니다.")
        print(f"Webots 프로그램에서 [File] -> [Open World]에서 다음 파일을 직접 열어주세요:\n{wbt_path}")

    input("\n계속하려면 엔터 키를 누르세요...")


def step3_interactive():
    clear_screen()
    print("=======================================================================")
    print(" [3단계] 사수 1인칭 인터랙티브 조준 사격 시뮬레이터 실행")
    print("=======================================================================\n")
    print("조작법: 마우스로 조준, READY 신호 시 스페이스바로 사격, Q 또는 ESC로 종료\n")
    script_path = os.path.join(PROJECT_ROOT, "scripts", "04_interactive_aiming_shooter_sim.py")
    subprocess.run([sys.executable, script_path])
    input("\n계속하려면 엔터 키를 누르세요...")


def step4_gazebo():
    clear_screen()
    print("=======================================================================")
    print(" [4단계] Gazebo 3D 물리 시뮬레이션 및 실시간 조준/격추 시뮬레이터 실행")
    print("=======================================================================\n")
    print("1. Gazebo GUI 3D 월드 실행 중...")
    subprocess.Popen('start "Gazebo GUI" wsl.exe -d Ubuntu-24.04 -- bash -c "cd /mnt/d/Aiming/gazebo && gz sim -r worlds/aiming_test.sdf"', shell=True)
    time.sleep(3)
    print("2. 3D 타겟 드론 실시간 회피 기동 드라이버 실행 중...")
    subprocess.Popen('start "Gazebo Target Driver" wsl.exe -d Ubuntu-24.04 -- bash -c "cd /mnt/d/Aiming/gazebo/scripts && python3 00_live_demo_drive.py"', shell=True)
    time.sleep(1)
    print("3. SMASH 스마트 스코프 실시간 조준 및 격추 HUD 창 실행 중...")
    script_path = os.path.join(PROJECT_ROOT, "scripts", "06_gazebo_interactive_shooter.py")
    subprocess.run([sys.executable, script_path])
    input("\n계속하려면 엔터 키를 누르세요...")


def step5_benchmark():
    clear_screen()
    print("=======================================================================")
    print(" [5단계] E2E 100회 자동 사격 벤치마크 및 결과 영상 렌더링")
    print("=======================================================================\n")
    print("1. 100회 연속 사격 데이터 수집 중...")
    subprocess.run('wsl.exe -d Ubuntu-24.04 -- bash -c "cd /mnt/d/Aiming/gazebo/scripts && python3 02_e2e_simulation_runner.py"', shell=True)
    print("\n2. 사격 분석 비디오 렌더링 중...")
    subprocess.run('wsl.exe -d Ubuntu-24.04 -- bash -c "cd /mnt/d/Aiming/gazebo/scripts && python3 03_render_result_video.py"', shell=True)
    print(f"\n결과 비디오 파일 생성 완료: {os.path.join(PROJECT_ROOT, 'gazebo', 'level2_result.mp4')}")
    input("\n계속하려면 엔터 키를 누르세요...")


def step6_all():
    clear_screen()
    print("=======================================================================")
    print(" [전체 파이프라인 순차 자동 검증]")
    print("=======================================================================\n")
    print("[1/3] 파인튜닝 검증...")
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "scripts", "05_auto_collect_and_finetune.py")])
    print("\n[2/3] 1인칭 사격 시뮬레이터 실행...")
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "scripts", "04_interactive_aiming_shooter_sim.py")])
    print("\n[3/3] Gazebo E2E 벤치마크 실행...")
    subprocess.run('wsl.exe -d Ubuntu-24.04 -- bash -c "cd /mnt/d/Aiming/gazebo/scripts && python3 02_e2e_simulation_runner.py && python3 03_render_result_video.py"', shell=True)
    print("\n모든 파이프라인 검증이 완료되었습니다!")
    input("\n계속하려면 엔터 키를 누르세요...")


def main():
    while True:
        clear_screen()
        print_banner()
        choice = input("실행할 파이프라인 번호를 입력하세요 (0~6): ").strip()

        if choice == "1":
            step1_finetune()
        elif choice == "2":
            step2_webots()
        elif choice == "3":
            step3_interactive()
        elif choice == "4":
            step4_gazebo()
        elif choice == "5":
            step5_benchmark()
        elif choice == "6":
            step6_all()
        elif choice == "0":
            print("\n프로그램을 종료합니다.")
            break


if __name__ == "__main__":
    main()
