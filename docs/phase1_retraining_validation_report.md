# Phase 1 재학습 검증 결과

검증일: 2026-11-17. 기준: [검증 가이드](phase1_validation_guide.md).

## 판정

**세 seed의 학습과 총 6,900회 평가를 완료했다. 현재 학습 설정의 최종 판정은 성능 미달이다.** seed 0·2는 각각 2,300회 모두 완주했지만 seed 1은 적신호 위반 42건·시간 초과 100건이 발생했다. 목표인 세 seed 모두의 기준 충족을 달성하지 못했다.

seed 0은 재출발 지표 측정 오류로 종합 판정 보류이며, seed 2는 정의된 수치 기준을 통과했다. 다만 seed 2에서도 극단적으로 가까운 정차와 조기 정차 후 재접근 사례를 확인했다. 최종 비교와 seed 2의 근거는 문서 마지막 절에 있다. 아래는 seed 0·1을 검증한 순서대로 보존한 상세 기록이다.

## 학습 완료 및 평가 조건

| 항목 | 확인 결과 |
|---|---|
| 새 run | `results/parallel_1_20261116_225606_528246` |
| 학습 seed / 초기화 | 0 / random, 기존 모델 가중치를 불러오지 않음 |
| 학습 완료 | 2026-11-17 01:34:17 KST 모델 저장 |
| 실제 iteration / 경험 수 | 1,920 / 62,914,560 |
| 로그 검증 | `training_log.csv` 1,920행, 마지막 iteration·steps가 목표와 일치 |
| 모델 SHA256 | `111f2585dc980ccbabf78d066f68045d76b73dd2b271ebfb7aa05f4f78e4b190` |
| 체크포인트 검증 | 최종 모델과 `checkpoints/iteration_1920/model.pt`의 SHA256 일치 |
| 환경 / 워커 / rollout | 512 / 4 / 환경별 64스텝 |
| epoch / minibatch | 10 / 1,024 |
| 조건 | 정상 신호, 배경 차량 없음, 랜덤 출발 차선 |
| 평가 코드 커밋 | `2303bcd418ae950cb822d4705b5df78d242f1a7f` |
| 평가 정책 | CPU에서 deterministic |

학습 당시 저장된 `source/`와 평가 시 checkout의 `env/sumo_env.py`, `env/parallel_sumo.py`, `utils/networks.py`, `algorithms/ppo.py`, `algorithms/parallel_ppo.py`, `utils/evaluator.py` 파일 내용이 일치함을 확인했다. 관측·보상 및 학습 옵션은 run의 설정 snapshot과 `parallel_config.json`을 사용했다. 학습이나 모델·평가 소스를 수정하지 않고 평가했다.

실행 명령:

```powershell
.venv/Scripts/python.exe scripts/evaluate_quick.py results/parallel_1_20261116_225606_528246 --num-workers 4 --offset-set development
.venv/Scripts/python.exe scripts/evaluate_quick.py results/parallel_1_20261116_225606_528246 --num-workers 4 --offset-set holdout
```

세트당 고유 경로 23개 × 신호 offset 50개 = 1,150행이며 중복·누락이 없다. development와 holdout의 같은 경로·offset 순번에서 실제 출발 차선 불일치가 0/1,150건이다. 평가 결과의 모델 해시도 최종 모델과 일치한다. 평가 전에 모델을 동결했고 두 세트 결과로 모델이나 보상을 변경하지 않았다.

## 수치 결과

| 항목 | development | holdout | 기준 대비 |
|---|---:|---:|---|
| 완주 | 1,150/1,150 | 1,150/1,150 | 통과 |
| 경로별 완주 | 모든 경로 50/50 | 모든 경로 50/50 | 통과 |
| 충돌·적신호 위반·wrong lane·timeout·unknown | 각각 0 | 각각 0 | 통과 |
| 마지막 적신호 정차 n | 702 | 700 | 통과 |
| 마지막 적신호 정차 중앙값 / P90 | 2.310 / 2.635m | 2.309 / 2.558m | 통과 |
| 최초 황·적신호 정차 n | 702 | 700 | 통과 |
| 최초 황·적신호 정차 중앙값 / P90 | 2.381 / 2.635m | 2.380 / 2.635m | 통과 |
| 경로별 정차 기준 위반（n ≥ 10） | 0경로 | 0경로 | 통과 |
| 재출발 observed | 702 | 700 | 표본 수 충족 |
| 재출발 no_restart 원시 집계 | **39** | **39** | 0회 기준 미충족, 측정 오류 확인 |
| 관측 성공한 재출발 지연 중앙값 / P90 | 0.5 / 0.5초 | 0.5 / 0.5초 | 지연 기준 충족, 미관측 제외 값 |
| 최초 정차 최대 거리 | **34.817m** | **18.415m** | 수치 통과와 별도로 사례 검토 필요 |
| 최초 정차 거리 > 5m | 10건 | 13건 | 잔여 행동 문제 |
| 마지막 적신호 정차 거리 > 5m | 2건 | 3건 | 잔여 행동 문제 |

평가 출력:

- [development 요약](../experiments/phase1_validation_20261117/seed0/analysis/quick/evaluation_summary.json), [episode CSV](../experiments/phase1_validation_20261117/seed0/analysis/quick/route_evaluation.csv), [평가 그림](../experiments/phase1_validation_20261117/seed0/analysis/quick/learning_curve.png)
- [holdout 요약](../experiments/phase1_validation_20261117/seed0/analysis/quick_holdout/evaluation_summary.json), [episode CSV](../experiments/phase1_validation_20261117/seed0/analysis/quick_holdout/route_evaluation.csv), [평가 그림](../experiments/phase1_validation_20261117/seed0/analysis/quick_holdout/learning_curve.png)

## 재출발 지표 오류

각 세트의 no_restart 39건은 모두 `ego_d_A4_A5`에서 발생했다. 출발 차선별로 0번 14건, 1번 14건, 2번 11건이다. 원시 결과는 수정하지 않고 보존했다.

대표 사례: A4→A5, 출발 차선 2, offset 1.0초.

| 행동 후 시각 | 속도 | 차선·신호 관측 |
|---|---:|---|
| 74.5초 | 0m/s | 차선 1, 신호 `nC` link 25가 초록으로 전환 |
| 75.0초 | **2.458m/s** | 차선 2로 변경, 같은 신호의 link **26**이 초록 |
| 75.5초 | 4.934m/s | 신호 통과, green_pass_count 1 |

실제로는 초록 전환 후 0.5초에 재출발했다. 하지만 `utils/evaluator.py`의 `_record_green_restart`는 `(tls_id, link_index)`가 같아야 출발로 인정한다. 차선 변경으로 link 25→26이 되자 추적을 끝내고 no_restart로 집계했다. 다음 스텝의 정상 통과도 집계를 복구하지 않는다.

이는 대표 사례로 확인한 **측정 오류**다. 78건 전체가 동일한 원인이라고 확정하거나 no_restart를 임의로 0으로 정정하지 않는다. 수정 시에도 다른 신호나 적색 연결로 변경한 것을 재출발로 잘못 인정하지 않도록, 같은 정지선·주행 연결과 차선 변경을 함께 판별해야 한다.

근거: [상세 신호·차선 기록](../experiments/phase1_validation_20261117/seed0/analysis/validation_traces/restart_link_diagnostic.json), [대표 trace](../experiments/phase1_validation_20261117/seed0/analysis/validation_traces/restart_debug.csv).

## 조기 정차 후 재접근

development에서 최초 정차 거리가 가장 큰 사례는 A3→A5, 출발 차선 2, offset 53.5초다.

- 황색에서 감속한 뒤 약 34.8m를 남기고 속도 0.5m/s 이하가 됐다.
- 잠시 저속·정차한 뒤 적색 상태에서 다시 접근했다.
- 최종적으로 약 2.45m 앞에 정차했고 초록에서 출발해 완주했다.

따라서 34.8m 앞에서 신호가 바뀔 때까지 계속 정차한 사례는 아니다. **감속 도중 지나치게 일찍 저속·정차한 뒤 다시 전진하는 행동**이다. 최종 정차 거리만 보면 이 과정이 드러나지 않는다. 완주 성능은 좋지만 정차 접근 품질까지 완성됐다고 판정하기에는 남은 문제다.

근거: [최초 정차 최악 사례 trace](../experiments/phase1_validation_20261117/seed0/analysis/validation_traces/case_00.csv).

trace 도구의 위치·속도·관측은 행동 전, 신호·보상과 `time_s`는 행동 후 값이다. 단계 간 시각 차이 0.5초를 감안해 해석했다. 재출발 원인은 별도의 행동 후 상태 기록으로 확인했다.

## 사례 재생 범위와 남은 작업

세트별 최초 정차 거리 상위 3개, 마지막 적신호 정차 거리 상위 3개, 관측된 재출발 지연 상위 3개, Ring 5개 경로 각 1개, no_restart 대표 사례를 합쳐 **28개 고유 사례**를 재생했다. 전부 완주했고 원 평가와 episode 길이가 일치했다. 누적 보상 차이의 최댓값은 약 0.000035였다. 이는 표본 사례의 재현 확인이며 전체 2,300회의 전 과정을 trace로 검토한 것은 아니다.

사례 목록과 근거는 [manifest](../experiments/phase1_validation_20261117/seed0/analysis/validation_traces/manifest.json), [원 평가와 비교](../experiments/phase1_validation_20261117/seed0/analysis/validation_traces/replay_comparison.json)에 있다. trace 실행 stdout 기록은 게시 대상에서 제외했다.

다음 순서로 진행한다.

1. 모델과 학습 조건을 유지한 채 재출발 측정기를 수정하고, 동일 link 출발·초록 차선 변경·적색 차선 변경·진짜 미출발 사례의 회귀 검증을 추가한다.
2. 현재 원시 결과를 보존하고 수정한 평가기의 결과를 새 경로에 저장한다. 두 결과의 코드 버전과 지표 차이를 명시한다.
3. seed 1·2를 동일 조건으로 새로 학습·평가한다. seed 0 한 개의 성공만으로 재현성을 선언하지 않는다.
4. 조기 정차 문제는 별도 개선 대상으로 남긴다. 이번 재현성 검증 도중 보상·관측을 바꾸지 않는다.

이번 작업에서는 seed 1·2 학습을 시작하지 않았고 학습·평가 소스도 변경하지 않았다.

## 추가 검증: seed 1

2026-11-17 사용자가 완료한 seed 1 모델을 같은 평가기로 검증했다. seed 2 학습은 시작하지 않았다.

| 항목 | 확인 결과 |
|---|---|
| run | `results/parallel_1_20261117_015702_574729` |
| 완료 시각 | 2026-11-17 03:59:54 KST |
| 실제 학습량 | 1,920 iteration, 62,914,560스텝, CSV 1,920행 |
| seed 0 대비 설정 차이 | `parallel_config.json`에서 seed 0→1만 다름 |
| 소스 비교 | 위에서 비교한 학습·환경·네트워크·평가 파일 6개 모두 seed 0 snapshot 및 현재 코드와 일치 |
| SHA256 | `6ab446183292db8049accab8851667bcec8db7ed0afda290dde996eacaec546d` |
| 모델 검증 | 최종 모델과 iteration 1920 체크포인트의 해시 일치 |
| 두 평가 세트의 출발 차선 | 1,150쌍 모두 일치 |

| 항목 | development | holdout |
|---|---:|---:|
| 완주 | **1,079/1,150 (93.83%)** | **1,079/1,150 (93.83%)** |
| 적신호 위반 | **21** | **21** |
| 시간 초과 | **50** | **50** |
| 충돌·wrong lane·unknown | 각각 0 | 각각 0 |
| A3→A5 | 50/50 시간 초과 | 50/50 시간 초과 |
| A3→A4 | 20건 신호 위반 | 19건 신호 위반 |
| A3→A2 | 1건 신호 위반 | 2건 신호 위반 |
| 마지막 적신호 정차 중앙값 / P90 | 1.829 / 2.782m | 1.829 / 2.782m |
| 최초 황·적신호 정차 중앙값 / P90 | 2.370 / 3.207m | 2.370 / 3.172m |
| 재출발 no_restart 원시 값 | 168 | 168 |

전체 정차 통계는 기준에 들어가지만, A1→A2·A1→A5·A4→A5 등의 경로에서 지나치게 가까운 정차가 나타난다. 예를 들어 development의 마지막 적신호 정차 중앙값은 A1→A2 약 0.645m, A4→A5 약 0.543m이며 경로별 기준을 충족하지 못한다. 전체 평균으로 경로별 실패를 가리지 않는다.

대표 사례 재생:

- **A3→A5, 출발 차선 0, offset 1초:** 1,000스텝·500초 시간 초과를 재현했다. 주행 후 A3_in에서 정지했고, 174.5·299.5·424.5초의 청신호 구간에도 속도 0을 유지했다. 후반 100스텝도 전부 정지 상태였다. 마지막 스텝에는 속도 추종 보상이 약 +0.0967로 지급됐고 후반 100스텝의 보상 합은 약 +9.67이었다. **실제 청신호 미출발과 정지 상태 보상 지급**을 확인한 사례다. 이것만으로 학습 실패의 단일 원인을 확정하지는 않는다.
- **A3→A4, 출발 차선 1, offset 1초:** 원 평가에 포함된 사례에서 57스텝·28.5초에 `red_light_violation` 종료를 재현했다.
- 재출발 집계에는 앞서 확인한 측정 오류가 여전히 있어 168건 전부를 실제 미출발로 단정할 수 없다. 하지만 위 시간 초과·신호 위반은 그 지표와 독립적으로 확인한 실제 실패다.

근거: [development 요약](../experiments/phase1_validation_20261117/seed1/analysis/quick/evaluation_summary.json), [holdout 요약](../experiments/phase1_validation_20261117/seed1/analysis/quick_holdout/evaluation_summary.json), [시간 초과 trace](../experiments/phase1_validation_20261117/seed1/analysis/validation_traces/timeout.csv), [신호 위반 trace](../experiments/phase1_validation_20261117/seed1/analysis/validation_traces/redlight_evaluation_case.csv).

**seed 1 판정: 성능 미달.** 학습 완료·동일 조건은 확인됐으나 Phase 1 성능은 통과하지 못했다.

seed 1 검증 당시 seed 2도 동일 설정으로 진행하도록 권고했다. 아래 최종 검증에서는 사용자가 완료한 seed 2를 평가했다. seed 1 실패를 제외하고 전체 재현성 통과로 선언하지 않는다.

## 최종 검증: seed 2 및 세 seed 비교

### 완료 및 무결성

| 항목 | 확인 결과 |
|---|---|
| seed 2 run | `results/parallel_1_20261117_045910_064394` |
| 완료 시각 | 2026-11-17 06:48:29 KST |
| 실제 학습량 | 1,920 iteration, 62,914,560스텝, CSV 1,920행 |
| seed 0 대비 설정 차이 | `parallel_config.json`에서 seed 0→2만 다름 |
| 소스 비교 | 위의 학습·환경·네트워크·평가 파일 6개가 seed 0 snapshot 및 평가 코드와 일치 |
| 모델 SHA256 | `fca78831435602d90799cef0c10159b375553d64c2f582393899540c085b4ab7` |
| 체크포인트 검증 | 최종 모델과 iteration 1920 체크포인트 내용 일치 |
| 평가 코드 커밋 | `2303bcd418ae950cb822d4705b5df78d242f1a7f` |
| 평가 입력 | 두 세트 각각 1,150개 고유 조합, 출발 차선 1,150쌍 일치 |

학습 모델과 환경·보상·평가 소스를 바꾸지 않고 기존 두 평가 명령의 run 경로만 seed 2로 바꾸어 실행했다. 개발·검증 두 세트는 신호 offset을 나눈 평가 세트이며, 학습 분포와 전혀 다른 환경에 대한 검증은 아니다.

### 최종 6개 결과

실패 열은 신호 위반 / 시간 초과다. 모든 행에서 충돌·wrong lane·unknown은 0이다. 정차 값의 단위는 m다.

| seed | 세트 | 완주 | 실패 | 마지막 정차 n / 중앙값 / P90 | 최초 정차 n / 중앙값 / P90 | 재출발 observed / no_restart / P90(초) |
|---|---|---:|---|---|---|---|
| 0 | development | 1,150/1,150 | 0 / 0 | 702 / 2.310 / 2.635 | 702 / 2.381 / 2.635 | 702 / 39 / 0.5 |
| 0 | holdout | 1,150/1,150 | 0 / 0 | 700 / 2.309 / 2.558 | 700 / 2.380 / 2.635 | 700 / 39 / 0.5 |
| 1 | development | **1,079/1,150** | **21 / 50** | 714 / 1.829 / 2.782 | 714 / 2.370 / 3.207 | 693 / 168 / 0.5 |
| 1 | holdout | **1,079/1,150** | **21 / 50** | 712 / 1.829 / 2.782 | 712 / 2.370 / 3.172 | 693 / 168 / 0.5 |
| 2 | development | **1,150/1,150** | 0 / 0 | 705 / 2.441 / 2.667 | 705 / 2.482 / 2.832 | **705 / 0 / 0.5** |
| 2 | holdout | **1,150/1,150** | 0 / 0 | 701 / 2.450 / 2.667 | 701 / 2.482 / 2.832 | **701 / 0 / 0.5** |

seed 2는 모든 경로에서 50/50 완주했으며 n ≥ 10인 경로별 정차 거리 기준의 위반도 없었다.

seed 0·1의 no_restart는 알려진 측정 오류가 섞인 원시 값이다. seed 1의 실제 위반·시간 초과를 이 오류 탓으로 제외하지 않는다. seed 2가 실패하지 않았다고 해서 seed 1의 실패 원인이 단순히 운이 나빴다고 결론 내릴 수도 없다. 동일 설정의 학습 결과가 seed에 민감하다는 근거다.

seed 2 평가 출력: [development 요약](../experiments/phase1_validation_20261117/seed2/analysis/quick/evaluation_summary.json), [development CSV](../experiments/phase1_validation_20261117/seed2/analysis/quick/route_evaluation.csv), [holdout 요약](../experiments/phase1_validation_20261117/seed2/analysis/quick_holdout/evaluation_summary.json), [holdout CSV](../experiments/phase1_validation_20261117/seed2/analysis/quick_holdout/route_evaluation.csv).

### seed 2 잔여 행동 문제

두 세트의 최초·최종 정차 거리 상위 사례, 재출발 지연 상위 사례, 최소 정차 거리 사례, Ring 5개 경로를 합쳐 **30개 고유 사례**를 재생했다. 모두 완주했고 원 평가와 episode 길이가 일치했으며 누적 보상 차이의 최댓값은 약 0.000015였다. [사례 manifest](../experiments/phase1_validation_20261117/seed2/analysis/validation_traces/manifest.json), [재생 비교](../experiments/phase1_validation_20261117/seed2/analysis/validation_traces/replay_comparison.json)에 근거를 보존했다. 시각은 기존 trace의 행동 전후 차이를 감안해 해석했다.

- development의 A5→A5, 출발 차선 1, offset 106초에서 적색 정차 거리가 **약 0.00947m(약 0.95cm)**였다. 황색 접근 후 정지선 바로 앞에 정차했으며 적색을 넘지는 않아 신호 위반으로 집계되지 않았다. 이는 목표 정차 여유 1.5~3m를 크게 벗어난 극단 사례다. 전체·경로별 중앙값/P90 기준이 이런 단일 사례를 놓칠 수 있다.
- 최초 황·적신호 정차 거리가 5m보다 큰 사례는 두 세트에 각각 1건이었다. development의 최댓값은 약 23.50m다. 최종 정차만으로 접근 품질을 판단하지 않는다.
- 따라서 seed 2의 판정은 **현재 수치 기준 통과, 행동 품질 개선 필요**다. 검증 후 최솟값 기준을 새로 추가해 기존 사전 기준의 통과 여부를 소급 변경하지는 않는다. 다음 실험에서는 하위 꼬리·최소 여유·제동 과정에 대한 기준을 사전에 보강해야 한다.

### 전체 결론과 다음 작업

정식 학습은 seed당 62,914,560개, 총 **188,743,680개 경험**이며, 평가 본편은 총 **6,900 episode**다. 추가 trace 재생은 이 평가 수에 포함하지 않는다.

여섯 결과 CSV의 23개 경로·경로당 50행·중복/누락 없음, 각 JSON의 모델 해시 일치, 세 seed·두 세트 전체의 같은 경로·offset 순번별 실제 출발 차선 일치를 교차 검증했다. [세 seed 통합 원시 요약](../experiments/phase1_validation_20261117/seed2/analysis/three_seed_summary.json)에 여섯 결과를 함께 보존했다.

**세 seed 재현성 검증은 완료했으며, 현재 설정은 성능 미달이다.** Phase 2 진입보다 다음 작업을 우선한다.

1. 재출발 측정기의 차선 변경 추적 오류를 고치고 기존 모델 3개를 동일한 수정 평가기로 비교한다. 원시 결과는 보존한다.
2. seed 1의 A3→A5 청신호 미출발과 A3→A4/A3→A2 신호 위반을 고정 회귀 사례로 삼아 관측·정책·보상 성분을 비교한다.
3. 멈춰 있는데도 적신호 속도 추종 가점이 누적되는 현상과, 지나치게 가까운 정차가 보상으로 충분히 구분되는지 조사한다. 단일 원인으로 확정하기 전 seed 0·2의 같은 상황과 비교한다.
4. 수정은 별도 실험으로 진행한다. 추가 seed를 계속 돌려 성공한 모델만 고르거나 기존 실패를 제외하지 않는다.

원본 학습·평가 소스와 모델은 이 검증에서 변경하지 않았다.
