# Phase 1 검증 마무리: 후속 에이전트 실행 지침

작성일: 2026-11-13. 검토 기준: `parallel_train`, `ebafef3`와 현재 미커밋 변경 및 로컬 결과.

## 1. 목적과 우선순위

**현재 최종 보상 후보를 고정하고, 동일 학습 설정 seed 0/1/2의 개발·별도 offset 평가를 완성하여 합격 또는 미달을 증거로 판정한다. 동시에 trace 기록 오류와 결과 문서의 오류를 바로잡는다.**

이 문서는 `phase1_agent_handoff.md`의 후속 지침이다. 충돌하는 부분은 이번 문서를 우선한다. 특히 평가 경로는 현재 코드의 **23개 고유 경로**로 확정하고, 추가 보상 탐색은 수행하지 않는다. 기존 지침의 27개 ID 기준에서 바뀐 이유는 아래에 명시한다.

문서만 고치거나 학습 한 번 실행하고 끝내지 않는다. 새 seed 2 모델의 평가와 세 모델의 최종 판정표까지 완성한다. 목표 미달이면 기준을 낮추거나 더 유리한 seed로 바꾸지 말고 미달 보고로 종료한다.

## 2. 검토에서 확인된 사실

| 역할 | run | 확인된 상태 |
|---|---|---|
| 최종 seed 0 | `parallel_1_20261113_142521_414407` | 개발 평가 1,150/1,150, 별도 offset 평가 없음 |
| 최종 seed 1 | `parallel_1_20261113_144440_871432` | 개발·별도 평가 각각 1,150/1,150 |
| 기존 seed 2, 참고용 | `parallel_1_20261113_150435_751583` | 개발·별도 평가 각각 1,150/1,150이나 학습 설정이 달라 동일 조건 재현성에서 제외 |

실제 파일 SHA256:

- seed 0: `df8bb61c748bf30f4037d3f711a1f7ff0d2369dc45ac370acc7c925af600c5a0`
- seed 1: `ad20cadc69195c91646ea09193432c48c8649de5bd850142ce3e433e5cb6e20a`
- 기존 seed 2: `7bab0b0e94ffcf04b72ad83c767f535bf4f84fc9f8568c8e4cec9fd5f73b61e3`

기존 seed 2는 전체 환경 **4,096**, 워커 4, 워커당 환경 1,024, 환경당 rollout 24, iteration 333, epoch 5, minibatch 2,048, 실제 32,735,232 transition이다. 문서의 `1,024 × 24 × 333 ≈ 32.7M`은 잘못된 식이다. 기존 모델과 결과는 참고용으로 보존한다.

개발 평가를 원시 CSV에서 재집계한 최종 후보 정차 중앙값/p90은 seed 0: 4.20/4.65m, seed 1: 4.07/4.47m, 기존 seed 2: 4.63/5.18m다. 수치와 모델 해시는 확인됐지만 이것만으로 동일 조건 seed 3개 검증이 완료되지는 않았다.

## 3. 작업 범위와 실행 한도

- 현재까지 smoke를 제외한 장기 학습은 10회가 확인됐다. 시작할 때 추가 실행이 있었는지 다시 확인한다.
- 이번 작업의 정상적인 추가 학습은 **seed 2 한 번**, 8,192,000 transition이다.
- 자원 오류 등으로 완주하지 못한 경우에만 원인 수정 후 한 번 재시도할 수 있다. 이전 차수를 포함한 장기 학습 12회 한도를 지킨다.
- 완주했지만 성능이 낮다는 이유로 동일 seed를 재실행하거나 보상을 바꾸지 않는다. 이 경우 미달로 판정한다.
- 기존 seed 0/1을 다시 학습하지 않는다. 배경차 학습, 랜덤 차선 성능 튜닝, 도로 수정도 하지 않는다.
- 한 번에 장기 학습 하나만 실행한다. 이미 실행 중이면 중복 실행하지 않는다.
- 사용자 미커밋 변경과 staged 변경을 보존한다. `git reset`, `git clean`, 결과 폴더 삭제로 정리하지 않는다.

## 4. 시작 전 확인 및 기록

1. AGENTS.md, 현재 branch/commit, `git status`, `docs/phase1_progress.md`를 읽는다.
2. Python 실행 경로, Python/PyTorch/SUMO/libsumo 버전과 장치를 확인한다. 이전 기록은 Python 3.11.9, PyTorch 2.11.0+cu128, CUDA, libsumo 1.28.0이다. 환경이 다르면 차이를 먼저 기록한다.
3. 위 세 모델의 해시와 저장 설정을 확인한다. 별도 offset 결과가 새로 생겼으면 중복 작업 대신 유효성을 검사한다.
4. seed 0/1 run의 저장 코드와 현재 코드에서 관측, 보상, 경로 샘플링, 종료 판정, PPO 설정을 비교한다. 필요한 로깅 수정 외에 동작이 달라졌으면 그 상태로 동일 설정 재현성이라고 주장하지 않는다. 기존 run은 수정하지 않고 동일 동작의 실행 소스를 별도로 확보한다.
5. 진행 문서의 상태를 `검증 보완 중 — 완료 판정 보류`로 바꾸고 수행할 명령과 학습 횟수를 적는다.

## 5. Trace 오류 수정 및 검증

### 5.1 총보상 덮어쓰기 오류

`scripts/trace_evaluation.py`에서 `"reward": reward` 뒤의 dictionary comprehension이 `reward` 키까지 포함한다. `info.get("reward")`가 None이어서 총보상을 덮어쓴다. 실제 저장 CSV의 reward 열도 비어 있다.

보상 항 목록에서 총보상 키를 제외하거나 총보상을 마지막에 대입하여 수정한다. 가짜 숫자나 보상 항 일부의 합으로 총보상을 대신하지 말고 **env.step이 반환한 실제 reward**를 저장한다.

검증 조건:

- 생성한 trace의 모든 step에서 reward가 유한한 수치다.
- trace reward 합과 같은 사례의 평가 `ep_return`이 부동소수점 오차 범위에서 일치한다. 권장 허용 오차: 절대 1e-4 또는 상대 1e-5.
- 동일 경로·offset·출발 난수에서 종료 원인이 평가 CSV와 같다.
- 실제 dict 조립에서 총보상이 None으로 덮이지 않는 작은 회귀 테스트를 추가하거나 동등한 명시적 검증을 남긴다.

### 5.2 대표 trace 재생성

기존 trace는 덮어쓰지 말고 `analysis/traces_verified/`에 수정 후 결과를 저장한다.

seed 0의 세 사례:

```text
python scripts/trace_evaluation.py results/parallel_1_20261113_142521_414407 ego_d_A5_A1 103.5 --output results/parallel_1_20261113_142521_414407/analysis/traces_verified/case_1.csv
python scripts/trace_evaluation.py results/parallel_1_20261113_142521_414407 ego_d_A4_A3 81.0 --output results/parallel_1_20261113_142521_414407/analysis/traces_verified/case_2.csv
python scripts/trace_evaluation.py results/parallel_1_20261113_142521_414407 ego_d_A1_A1 1.0 --output results/parallel_1_20261113_142521_414407/analysis/traces_verified/case_3.csv
```

랜덤 차선 실패 사례:

```text
python scripts/trace_evaluation.py results/parallel_1_20261113_144440_871432 ego_d_A1_A1 11.0 --randomize-departures --output results/parallel_1_20261113_144440_871432/analysis/traces_verified/wrong_lane_case.csv
```

기존 trace의 `lane_change_rejected`는 거절 여부일 뿐 이유가 아니다. 진단상 필요하면 내부 차선·없는 목표 차선·안전 간격 등 거절 이유를 행동 로직 변경 없이 기록한다. 이유를 구현하지 않았다면 보고서에 미수집이라고 명시한다. lane index 변화만으로 edge가 바뀌는 지점까지 실제 차선 변경으로 세지 않는다.

## 6. 누락된 seed 0 평가

```text
python scripts/evaluate_quick.py results/parallel_1_20261113_142521_414407 --offset-set separate
```

기존 seed 0 개발 평가와 seed 1의 두 평가는 출력 파일 존재 여부만 확인하지 말고 아래 공통 검사를 수행한다. 같은 모델·같은 개발 평가의 반복 종료 결과 일치 증거가 없으면 seed 0 개발 평가를 한 번 재실행하여 비교한다. 기존 원시 CSV와 JSON은 실행 전 별도 보존한다.

## 7. 동일 설정 seed 2 재학습

다음 세 보상뿐 아니라 적용되는 **전체 보상·관측·ACTION·HPARAMS**를 기존 seed 0/1과 대조해 고정한다.

```text
red_stop_clearance_reward = 0.02
red_stop_clearance_reward_scale = 2.0
red_stop_penalty_coef = 0.02
```

프로젝트 루트에서 한 줄 명령으로 실행한다. Python 실행 환경은 앞에서 확인한 환경을 사용한다.

```text
python train_parallel.py --phase 1 --num-envs 256 --num-workers 4 --steps-per-env 64 --iterations 500 --learning-epochs 10 --minibatch-size 512 --randomize-departures False --eval-interval 0 --seed 2
```

`--init-model`은 사용하지 않는다. 새 run 경로를 즉시 진행 문서에 기록한다. 기본값만으로 실행하지 않는다.

학습 완료 후 `parallel_config.json`과 `training_log.csv`에서 직접 확인한다:

| 키 | 필수 값 |
|---|---:|
| phase / seed | 1 / 2 |
| num_envs / num_workers / envs_per_worker | 256 / 4 / 64 |
| steps_per_env / rollout_steps | 64 / 16,384 |
| iterations / epochs / minibatch_size | 500 / 10 / 512 |
| total_timesteps 및 로그 마지막 steps | 8,192,000 |
| randomize_departures / traffic_curriculum | false / null |
| initialization | random |

새 모델의 해시와 코드 commit·미커밋 diff·정확한 명령을 저장한다. seed가 JSON뿐 아니라 정책과 환경에 실제로 전달되는지도 코드로 확인한다.

```text
python scripts/evaluate_quick.py results/<새_seed2_run>
python scripts/evaluate_quick.py results/<새_seed2_run> --offset-set separate
```

## 8. 평가 집합과 완료 기준

### 8.1 평가 집합 확정

`ebafef3`의 `env/ego_routes.py`는 동일 edge sequence의 중복 route ID를 제거한다. 현재 기준은 Ring 미경유 18개 + Ring 경유 5개, 총 23개 고유 경로다. 삭제한 네 ID와 보존한 동일 edge 경로의 대응표를 XML에서 생성해 보고서에 넣는다. 손실된 고유 경로가 없는지 확인한다.

개발: 1.0, 3.5, …, 123.5초. 별도: 2.0, 4.5, …, 124.5초. 각각 50개 offset, 총 1,150회다. 최종 평가는 중앙 차선 출발과 배경차 0으로 고정한다.

원래 인계 문서의 1,350회 기준을 이번 후속 작업에서 1,150회로 명시적으로 갱신한다. 이는 중복 경로 제거에 따른 분모 변경이며, 실패 사례 제거가 아니다. 신호 위반·충돌·정차 품질과 경로별 기준은 그대로 유지한다.

### 8.2 각 seed × 각 평가 집합의 합격 조건

| 지표 | 합격 조건 |
|---|---|
| 전체 도착 | 1,150회 중 1,139회 이상, 99% 이상 |
| 경로별 도착 | 23개 경로 각각 50회 중 48회 이상 |
| 적신호 위반 / 충돌 / unknown | 각각 0회 |
| 적신호 정차 표본 | 100개 이상 |
| 정차 중앙값 | 3~8m |
| 정차 p90 | 15m 이하 |

동일 설정의 seed 0/1/새 seed 2 × 개발/별도 평가, 총 **6행 모두** 충족해야 합격이다. 기존 대규모 seed 2를 대신 넣거나 seed 평균으로 판정하지 않는다. 결과를 보고 보상이나 평가 집합을 수정하지 않는다.

### 8.3 원시 자료 검증

- 모델 실제 SHA256과 요약 JSON의 해시가 같다.
- CSV가 정확히 1,150행이고 `(route, signal_offset_s)`가 모두 고유하다.
- 23개 경로 각각 offset 50개, offset 집합은 지정값과 정확히 같다.
- 모든 행이 한 종료 원인으로 분류되고 합계가 전체 행 수와 같다.
- CSV에서 도착 수·위반 수·경로별 도착·정차 표본·중앙값·p90을 다시 계산해 JSON과 대조한다.
- 결측 정차 표본은 0m로 대체하지 않는다. 표본 수와 coverage를 기록한다.
- 정차 지표는 에피소드의 **마지막 측정 가능한 적신호 정차**를 나타낸다. 접근 과정 전체의 쾌적성이나 모든 정차의 품질이 검증됐다고 주장하지 않는다.

## 9. 문서 수정과 누락 실험 보충

`docs/phase1_progress.md`, `docs/phase1_experiment_report.md`, `docs/phase1_results.md`를 일관되게 갱신한다. 기존 최종 후보의 참고 성과와 이번 동일 설정 검증 결과를 구분한다.

반드시 수정할 사항:

1. seed 2 전체 환경 수 4,096과 워커당 1,024를 구분하고, epoch·minibatch 차이까지 기록한다.
2. 검증 전에는 “기준 통과” 대신 “후보 확보, 최종 검증 보완 중”으로 표시한다. 검증 후 6행 판정표로 결론을 갱신한다.
3. 적신호 거리 penalty는 멀수록 감점이 커져 접근을 유도한다. “적신호 진행을 억제한다”는 설명을 수정한다. late-brake penalty와 혼동하지 않는다.
4. Phase 2는 커리큘럼 구현이 존재한다. “미구현”을 구현 상태와 성능 검증 상태로 나누어 수정한다. 이번에 Phase 2를 검증했다고 쓰지 않는다.
5. 아래 누락 후보를 저장 설정과 CSV로 확인해 전체 실험 표에 추가한다.

| run | 변경 | 검토 시 정차 중앙값 / p90 |
|---|---|---|
| `parallel_1_20261113_134618_825826` | clearance reward 0.05 | 22.03 / 30.49m |
| `parallel_1_20261113_140556_054790` | reward 0.02, scale 2.0, penalty 0 | 3.96 / 25.29m |

최초 기준 모델 `parallel_1_20261113_120614_157702`은 평가 1,350회이며 후속 모델은 1,150회다. `ebafef3`는 평가뿐 아니라 학습 경로 샘플링도 바꿨다. 저장 코드 비교로 적용 버전을 확인하고, 최초 기준→최종 후보의 개선을 보상만의 인과 효과로 단정하지 않는다. 이 차수에서 인과 비교를 위해 기준 모델을 추가로 장기 학습하지 않는다.

`docs/train.md`와 `docs/parallel_train.md`에는 새 단계 정의와 최신 보고서 링크를 반영한다. 기존 `train.md`의 과거 성과는 과거 조건으로 보존한다. 이전 문서의 “train.md 갱신 차단” 주장은 이번 실행에서 실제 차단 사유가 확인된 경우에만 반복한다. 수정 불가 시 정확한 사유와 대신 작성한 문서를 보고하고, 나머지 검증은 계속한다.

## 10. 최종 보고와 다음 에이전트 인계

진행 문서에는 실행 중인 명령·세션 또는 PID·로그·실제 run·현재 장기 학습 횟수·다음 명령을 계속 갱신한다. 종료 시 다음을 남긴다.

- 동일 설정 최종 모델 3개와 SHA256, 실제 학습량, 적용 설정.
- 6행 평가표: seed, run, 개발/별도, 도착/전체, 최악 경로, 위반, 충돌, unknown, 정차 n/coverage/median/p90, 합격 여부.
- 수정 후 trace 4개와 총보상 합 검증 결과.
- 23개 고유 경로와 중복 ID 대응표.
- 장기 학습 전체 실행 목록 및 횟수.
- 결론: **달성**, **성능 미달**, **환경 문제로 미완료** 중 하나. 미달이면 실패 경로·offset과 다음 가설만 제안하고 이번 차수에서 추가 튜닝하지 않는다.

성능 목표가 달성돼도 랜덤 차선 35건 실패는 별도 일반화 한계로 유지한다. 중앙 차선의 성공을 랜덤 차선·배경차·실도로 안전성으로 확대해 표현하지 않는다.

## 11. 다음 에이전트에게 보낼 요청

> `docs/phase1_verification_followup.md`를 우선 지침으로 읽고 검증 마무리를 수행해줘. 보상과 모델 구조를 더 튜닝하지 말고, trace 총보상 기록 오류 수정 → seed 0 누락 평가 → 동일 설정 seed 2 재학습 및 두 평가 → 원시 CSV 검증 → 문서 갱신까지 진행해줘. 이미 완료된 작업은 진행 기록과 실제 산출물을 검증한 뒤 재사용해줘. seed 2는 문서의 전체 옵션을 빠짐없이 사용하고, 기존 대규모 seed 2를 최종 재현성 표에 넣지 마. 6행 판정표가 모두 충족될 때만 달성으로 표시하고, 실패하면 기준을 바꾸지 말고 미달 원인과 결과를 남겨줘.
