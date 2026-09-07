# `test/` — 외부 데이터셋 배치 위치

이 폴더는 **Anti-UAV300 데이터셋의 검증 시퀀스**를 두는 자리다.
데이터셋 본체는 재배포 권한 문제로 저장소에 포함하지 않는다
(`detector+tracker/final/docs/DATASET.md`, `detector+tracker/final/LICENSE_NOTICE.md` 참고).

`tests/`(pytest 단위 테스트)와는 다른 폴더다. 혼동하지 말 것.

---

## 필요한 파일

| 파일 | 내용 | 저장소 포함 |
|---|---|---|
| `test/visible.mp4` | 검증 시퀀스 영상 (약 11.7MB, 1000 프레임) | **없음 — 직접 배치** |
| `test/visible.json` | 프레임별 정답 bbox 라벨 (약 44KB) | 있음 (재배포 권한 검토 대상) |

## 출처 및 배치 방법

Anti-UAV300 데이터셋을 배포처에서 내려받은 뒤, 아래 시퀀스의 파일을 이 폴더에 그대로 복사한다.

```
Anti-UAV300/test/20190925_111757_1_1/visible.mp4   →  test/visible.mp4
Anti-UAV300/test/20190925_111757_1_1/visible.json  →  test/visible.json
```

시퀀스 제원 (`DATASET.md` 기준)

- **시퀀스 1** — ID `20190925_111757_1_1`, 총 1000 프레임, UAV 존재 1000 / 부재 0
- (참고) 시퀀스 2 — ID `20190926_134054_1_1`, 총 1000 프레임, UAV 존재 871 / 부재 129.
  정답 상태 전이: frame 0 PRESENT → frame 1 ABSENT → frame 130 PRESENT

## 이 파일들을 쓰는 곳

| 위치 | 용도 |
|---|---|
| `gazebo/scripts/02_e2e_simulation_runner.py:53` | `PROJECT_ROOT / "test" / "visible.json"` 을 정답으로 사용 |
| `gazebo/scripts/03_render_result_video.py` | 결과 영상에 정답 bbox(노란 박스)를 겹쳐 렌더링 |
| `examples/render_aim_overlay.py` | `--gt test/visible.json --output test/visible_aim_overlay.mp4` |
| `config/bridge.test_video.yaml` | 이 영상 해상도(1920×1080)에 맞춘 데모용 내부 파라미터 |

**`visible.mp4` 가 없어도 나머지는 전부 동작한다.** 단위 테스트 114개, 오프라인 오라클 검증
(`scripts/07`·`08`), Gazebo/ROS 2 시뮬레이터는 이 파일에 의존하지 않는다.
영향을 받는 것은 위 표의 Level-2 통합 테스트와 영상 데모 경로뿐이다.

## 재배포에 관한 주의

`LICENSE_NOTICE.md` 는 공개 전환 전에 다음을 확인하라고 명시한다.

- Anti-UAV 데이터셋 재배포 조건
- 데이터셋에서 파생된 데모 영상의 재배포 권리

따라서 `visible.mp4` 는 물론, 이미 커밋돼 있는 `visible.json` 도 저장소를 공개로 전환하기 전에
재배포 가능 여부를 확인해야 한다.
