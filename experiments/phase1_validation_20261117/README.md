# Phase 1 검증 모델 3개

2026-11-17 검증한 seed 0·1·2의 최종 모델과 설정·평가 결과를 보존한다.

| seed | 원래 run | development / holdout 완주 |
|---|---|---|
| 0 | `parallel_1_20261116_225606_528246` | 1,150/1,150 · 1,150/1,150 |
| 1 | `parallel_1_20261117_015702_574729` | 1,079/1,150 · 1,079/1,150 |
| 2 | `parallel_1_20261117_045910_064394` | 1,150/1,150 · 1,150/1,150 |

**전체 판정은 성능 미달이다.** seed 1의 신호 위반·시간 초과 때문에 세 seed 모두 통과한다는 목표를 달성하지 못했다. seed 0·1의 재출발 집계에는 측정 오류가 있고, seed 2에도 정지선에 지나치게 가까이 정차하는 극단 사례가 있다.

자세한 근거: [최종 검증 보고서](../../docs/phase1_retraining_validation_report.md), [검증 가이드](../../docs/phase1_validation_guide.md).

## 보존 파일

각 `seed0/`, `seed1/`, `seed2/` 폴더에는 다음이 있다.

- `model.pt`: iteration 1920의 최종 정책 가중치. optimizer와 난수 상태를 포함한 학습 재개 체크포인트는 아니다.
- `parallel_config.json`, 설정 Python 파일: 실제 학습에 사용한 옵션·관측·보상 설정.
- `source/`: 학습 시 저장한 소스 snapshot. 재현 근거이며, 평가 스크립트가 이 폴더의 구현을 자동 실행하지는 않는다.
- `analysis/quick/`, `analysis/quick_holdout/`: episode CSV, 요약 JSON, 평가 그림.
- `analysis/validation_traces/`: 사례별 관측·행동·보상·신호 기록과 사례 목록. 일부 trace 필드는 행동 전후 시각이 다르므로 보고서의 해석 주의사항을 확인한다.

학습 콘솔 로그, `training_log.csv`, TensorBoard 이벤트, trace 실행 stdout 로그, 중간 체크포인트, 생성된 SUMO 도로 파일은 포함하지 않았다. 그래프에 이미 그려진 학습 곡선은 평가 그림의 일부다. 사례 trace CSV와 평가 CSV는 검증 근거로 포함했다.

`manifest.json`은 보존 payload 파일 179개의 크기와 SHA256을 기록한다. README·manifest·`.gitattributes` 자체는 manifest의 대상에 포함하지 않는다. `.gitattributes`는 원본 파일 바이트와 해시를 유지하기 위해 줄바꿈 변환을 끈다.

## 평가 다시 실행하기

프로젝트 루트에서 SUMO 및 Python 의존성이 설치된 환경으로 실행한다. 다른 seed는 경로의 `seed0`를 변경한다.

```powershell
.venv/Scripts/python.exe scripts/evaluate_quick.py experiments/phase1_validation_20261117/seed0 --num-workers 4 --offset-set development
.venv/Scripts/python.exe scripts/evaluate_quick.py experiments/phase1_validation_20261117/seed0 --num-workers 4 --offset-set holdout
```

평가는 실행한 checkout의 환경·평가 코드를 사용한다. 기존 평가 코드 커밋은 `2303bcd418ae950cb822d4705b5df78d242f1a7f`다. 이후 평가기를 고친 결과와 비교할 때는 코드 버전을 함께 기록한다. 학습 로그를 제외했으므로 재평가 시 학습 곡선 부분을 다시 그리지 못할 수 있다.

위 명령은 보존된 `analysis/` 결과를 덮어쓴다. 기존 증거를 유지하려면 **seed 폴더를 별도 작업 경로로 복사한 뒤 그 복사본에서 평가**한다. 원본 run은 로컬 `results/`에 그대로 남아 있다.
