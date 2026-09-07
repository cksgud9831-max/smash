# 프로젝트 폴더 전수 조사 및 불필요 파일 정돈 보고서

## 1. 분석 개요

본 보고서는 스마트 조준경 프로젝트 워크스페이스 내 모든 폴더와 파일을 전수 조사하여, 현재 시스템 동작에 불필요하거나 정리 필요한 파일 목록 및 삭제/재배치 방안을 정리한 문서입니다.

## 2. 불필요 및 정리 대상 파일 분류 명단

전수 조사 결과 총 5가지 유형의 정리 대상 파일이 도출되었습니다.

### 1유형: Dead Code 및 레거시 미사용 백엔드 파일 (bridge 계층)
* bridge/detector.py: OpticalFlowTracker 도입으로 인해 현재 메인 파이프라인에서 전혀 호출되지 않는 백엔드 파일.
* bridge/trt_infer.py: 이전 백엔드용 TensorRT 추론 모듈로 현재 파이프라인에서 쓰이지 않음.
* 이유 및 조치: 실장비 배포 시 ultralytics 직접 추론 방식을 채택함에 따라 완전 삭제하거나 legacy 폴더로 이동 조치 권장.

### 2유형: 딥러닝 학습 과정의 중복 가중치 및 임시 시각화 파일 (detector+tracker)
* detector+tracker/ga_results/yolo11s_ga_final_3/weights/last.pt (19.1MB): 학습 중 중간 저장 가중치로, 최종 최적 가중치인 best.pt 가 존재하므로 불필요.
* detector+tracker/ga_results/yolo11s_ga_final_3/train_batch*.jpg 및 val_batch*.jpg (약 3MB): 학습 진단용 임시 배치 이미지 파일들.
* 이유 및 조치: 용량을 차지하는 중복 가중치 last.pt 및 학습 임시 이미지를 삭제하여 워크스페이스 용량 확보.

### 3유형: 파이프라인 구동 결과 생성 렌더링 영상 및 임시 테스트 파일
* test/visible_aim_overlay.mp4 (15.2MB): 1.5절 조준점 흔들림 검증 시 생성한 오버레이 결과 렌더링 영상.
* 이유 및 조치: 이미 검증이 완료되었으므로 visualizations 폴더로 이동하거나 용량 확보를 위해 삭제.

### 4유형: 파이썬 및 테스트 자동 생성 캐시 폴더
* .pytest_cache/ : pytest 실행 시 생성되는 임시 캐시 디렉터리.
* __pycache__/ (aiming_engine, bridge, examples, tests 내부) : 파이썬 바이트코드 바이트 캐시 파일들.
* .claude/scheduled_tasks.lock : CLI 스케줄 작업용 임시 락 파일.
* 이유 및 조치: 필요 시 자동 재생성되는 캐시 파일들로 .gitignore 등록 및 빌드 정리 시 클리어 대상.

### 5유형: 프로젝트 규칙 미준수 폴더 외 존재 문서 파일들
* paper_text.txt : 참고 논문 OCR 추출 임시 텍스트 파일. => pdfs 폴더로 이동 권장.
* aiming_engine_report.html : 초기 조준 엔진 보고서 HTML 파일. => interim_reports 또는 visualizations 폴더로 이동 권장.

## 3. 정돈 및 삭제 실행 계획 권가

1. **상위 용량 확보를 위한 1차 삭제 대상**:
   * detector+tracker/ga_results/yolo11s_ga_final_3/weights/last.pt (19.1MB)
   * test/visible_aim_overlay.mp4 (15.2MB)

2. **소프트웨어 구조 정돈을 위한 2차 삭제/이동 대상**:
   * dead code: bridge/detector.py, bridge/trt_infer.py
   * 레퍼런스 파일 이동: paper_text.txt => pdfs 폴더, aiming_engine_report.html => interim_reports 폴더

3. **캐시 클리어 스크립트 작성 (scripts/00_clean_cache.py)**:
   * __pycache__ 및 .pytest_cache 자동 삭제 스크립트를 준비하여 개발 환경 정돈 유지.

## 4. 정돈 및 삭제 수행 결과

1. **파일 삭제 조치 완료**:
   * detector+tracker/ga_results/yolo11s_ga_final_3/weights/last.pt (19.1MB) 및 학습 임시 이미지 삭제 완료
   * test/visible_aim_overlay.mp4 (15.2MB) 삭제 완료
   * bridge/detector.py 및 bridge/trt_infer.py 데드 코드 삭제 완료
   * 약 35MB 상당의 워크스페이스 공간 확보 완료

2. **문서 및 데이터 파일 규정 폴더 재배치 완료**:
   * paper_text.txt => pdfs/paper_text.txt 이동 완료
   * aiming_engine_report.html => interim_reports/aiming_engine_report.html 이동 완료

3. **캐시 청소 및 최종 단위 테스트 검증**:
   * scripts/00_clean_cache.py 구동하여 캐시 폴더 클리어 완료
   * 삭제 후 전체 109개 pytest 단위 테스트 수행 결과, 109개 모두 오류 없이 100% 정상 작동 검증 (109 passed in 1.81s)
