# Phase 1 최종 결과 정리

> 상태: **미달 (성능 미달).** 동일 학습 설정 seed 0/1/2 × 개발/별도 offset
> 6행 판정에서 seed 2가 기준을 충족하지 못했다.

Phase 1 신호 대응 후보의 검증 결과를 정리한다. 최신 개요는
[train.md](train.md)에 연결돼 있다. 진행 상세는 [phase1_progress.md](phase1_progress.md),
실험 의사결정과 결론은 [phase1_experiment_report.md](phase1_experiment_report.md),
작업 지침은 [phase1_agent_handoff.md](phase1_agent_handoff.md)를 참고한다.

## 1. 현재 Phase 정의

| 항목 | 현재 의미 |
|---|---|
| `train_parallel.py --phase 1` | 정상 신호 + 배경 차량 없음 + 중앙 차선 출발 |
| `train_parallel.py --phase 2` | 정상 신호 + 배경 차량 교통량 커리큘럼 (구현됨, 성능 미검증) |
| 과거 `docs/train.md`의 Phase 1 | 항상 초록불 + 배경 차량 없음. 현재 Phase 1과 다름 |

## 2. 최종 보상 후보

| 파라미터 | 값 |
|---|---|
| `red_stop_clearance_reward` | 0.02 |
| `red_stop_clearance_reward_scale` | 2.0 |
| `red_stop_penalty_coef` | 0.02 |

## 3. 최종 후보 평가 결과

평가 분모는 최신 평가기의 고유 경로 23개 × offset 50개 = 1,150회다.
development와 separate offset 집합을 각각 평가했다.

| 실행 | 도착 | 충돌 | 잘못된 차선 | 적신호 위반 | 정차 표본 | median | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 0 | 1,150/1,150 | 0 | 0 | 0 | 713 | 4.20m | 4.65m |
| seed 1 | 1,150/1,150 | 0 | 0 | 0 | 712 | 4.07m | 4.47m |
| (참고) 기존 seed 2 | 1,150/1,150 | 0 | 0 | 0 | 715 | 4.63m | 5.18m |
| **동일 설정 seed 2** | 1,050/1,150 | 0 | 100 | 0 | 753 | 25.21m | 41.80m |

동일 설정 seed 2는 개발·별도 평가 모두 도착 1,050/1,150(91.3%)으로 전체
도착 기준(99% 이상)을 밑돌았고, wrong-lane 100건이 Ring 경로 두 곳에
집중됐으며 정차 median 25.21m / p90 41.80m로 정차 기준도 크게 벗어났다.
따라서 최종 후보는 3개 seed에서 재현되지 않아 **미달**이다. 기준을 낮추거나
더 유리한 seed로 바꾸지 않았다.

### 3.1. 모델 SHA256

| seed | SHA256 |
|---|---|
| 0 | `df8bb61c748bf30f4037d3f711a1f7ff0d2369dc45ac370acc7c925af600c5a0` |
| 1 | `ad20cadc69195c91646ea09193432c48c8649de5bd850142ce3e433e5cb6e20a` |
| (참고) 기존 2 | `7bab0b0e94ffcf04b72ad83c767f535bf4f84fc9f8568c8e4cec9fd5f73b61e3` |
| 동일 설정 2 | `f23a95c3207060aa637adeb3cd78dd2c4f9a67fd144e98fc29446e1c63b8d3ae` |

## 4. seed 2 학습 예산 차이 (재현성 제한)

| seed | 환경 × rollout × iteration | epoch | minibatch | 대략적 transition |
|---|---|---|---:|---:|---:|
| 0 / 1 | 256 × 64 × 500 | 10 | 512 | 8,192,000 |
| (참고) 기존 2 | 4,096 × 24 × 333 | 5 | 2,048 | 32,735,232 |

기존 seed 2는 전체 환경 **4,096**(워커 4 × 워커당 1,024), 환경당 rollout
24, iteration 333로 저장되어 seed 0/1의 256 × 64 × 500 설정과 다르다.
`1,024 × 24 × 333 ≈ 32.7M`이라는 기존 식은 잘못됐다(워커당 환경 수를
환경당 rollout과 곱했다). 결과는 후보의 독립 검증으로만 참고하고 동일
학습 예산의 엄밀한 seed 재현성 증거로 해석하지 않는다. 동일 설정 seed 2를
256 × 64 × 500(8,192,000 transition)으로 재학습한 결과는 도착
1,050/1,150, wrong-lane 100건으로 실패해 미달로 판정했다.

## 5. 보상 후보 비교 (버린 후보 포함)

| 후보 | 결과 |
|---|---|
| clearance reward 0.20 | 정차는 개선됐으나 wrong-lane 82건, 적신호 위반 1건 → 폐기 |
| clearance reward 0.05 (`parallel_1_20261113_134618_825826`) | 정차 median 22.03m / p90 30.49m → 폐기 |
| clearance reward 0.02 + scale 2.0 + penalty 0 (`parallel_1_20261113_140556_054790`) | 정차 median 3.96m / p90 25.29m → 폐기 |
| **scale 2.0 + penalty 0.02 (최종 후보)** | 동일 설정 seed 0/1 통과, seed 2 실패 → 재현성 미달 |

적신호 거리 penalty(`red_stop_penalty_coef`)는 멀수록 감점이 커져 정지선에
접근하도록 유도하는 항이며, "적신호 진행을 억제한다"는 설명은 틀렸다.
이는 정차 후 재제동 감점(`red_stopped_brake_coef`)과 다르다.

### 5.1 최초 기준과의 비교 한계

최초 기준 모델 `parallel_1_20261113_120614_157702`은 27개 경로 × 50 offset
= 1,350회 평가였고, 후속 모델은 23개 고유 경로 × 50 = 1,150회다.
`ebafef3`는 평가뿐 아니라 학습 경로 샘플링도 변경했다. 따라서 최초 기준→
최종 후보의 개선을 보상만의 인과 효과로 단정하지 않는다.

## 6. 평가 경로 (27개 ID → 23개 고유 경로)

`ebafef3`의 `env/ego_routes.py`는 동일 edge sequence의 중복 route ID를
제거해 Ring 미경유 18개 + Ring 경유 5개, 총 23개 고유 경로로 평가한다.
중복 ID 4쌍(Ring ID와 동일 edge의 direct ID)은 아래와 같다. Ring ID가
보존되고 동일 edge의 direct ID가 제거된다.

| 제거된 direct ID | 보존된 Ring ID | 공통 edge sequence |
|---|---|---|
| `ego_d_A1_A3` | `ego_r_RING_A1_A3_A1_A3` | A1_in2 → RING_A1_A3 → A3_out1 |
| `ego_d_A2_A4` | `ego_r_RING_A2_A4_A2_A4` | A2_in1 → A2_in → RING_A2_A4 → A4_out1 → A4_out2 |
| `ego_d_A4_A1` | `ego_r_RING_A4_A1_A4_A1` | A4_in2 → A4_in1 → RING_A4_A1 → A1_out1 → A1_out2 |
| `ego_d_A5_A2` | `ego_r_RING_A5_A2_A5_A2` | A5_in1 → RING_A5_A2 → A2_out1 |

중복 제거로 손실된 고유 경로는 없다. 이에 따라 분모가 27 → 23으로
바뀌었고, 27 × 50 = 1,350회에서 23 × 50 = 1,150회가 됐다. 이는 중복 경로
제거에 따른 분모 변경이며 실패 사례 제거가 아니다.

## 7. 랜덤 출발 차선 진단 (일반화 진단, 합격 기준 아님)

seed 1 모델에 `--randomize-departures`를 적용해 1,150회 실행했다.

| 지표 | 값 |
|---|---:|
| 도착 | 1,115/1,150 (96.96%) |
| 잘못된 차선 | 35 (3.04%) |
| 충돌 | 0 |
| 적신호 위반 | 0 |

실패는 A1 출발 경로에 집중됐다. 중앙 차선 고정 학습 분포와 랜덤 출발 차선
분포 사이의 일반화 한계를 보여주며, Phase 1 합격 기준을 완화하는 근거로
사용하지 않는다. 이는 다음 작업의 기준 결과로 남긴다.

## 7.1 다음에 검증할 가설

1. Ring A2→A4의 실패는 진입 전 차선 이동 지연이다. 성공 seed 0과 실패 seed 2를
   같은 조건에서 비교해 연결성 관측, 차선 행동 확률, 최초 변경 시점과 속도를 확인한다.
   저장 trace에서는 마지막 변경 명령이 수락됐으므로 안전 게이트 완화를 먼저 하지 않는다.
2. 먼 정차는 Ring과 별개다. 직접 경로 18개 모두에서 나타나며 Ring의 정차 표본은 0개다.
   A4→A2, A5→A3, A1→A2의 최초 제동·정차와 내부 차선 진입 시점을 비교한다.
   내부 차선에서 연결성 관측을 1로 처리하는 구현이 원인인지는 아직 가설이다.
3. 대규모 seed 2의 성공은 학습량뿐 아니라 환경 수·rollout·epoch·minibatch가 달라
   학습량만의 효과로 볼 수 없다. 우선 위 두 진단으로 관측 부족과 행동 선택 문제를 구분한다.

## 8. 대표 trace

- seed 0 정차 사례(수정 후 `traces_verified/`):
  - `case_1.csv`: `ego_d_A5_A1`, offset 103.5s (reward 합 33.0604 = ep_return)
  - `case_2.csv`: `ego_d_A4_A3`, offset 81.0s (32.1382)
  - `case_3.csv`: `ego_d_A1_A1`, offset 1.0s (32.1800)
- 랜덤 출발 차선 실패: seed 1 `analysis/traces_verified/wrong_lane_case.csv`
  (`ego_d_A1_A1`, offset 11.0s, reward 합 −0.9838 = ep_return, outcome wrong_lane)
- 동일 설정 seed 2 Ring 실패: `analysis/traces_verified/ring_fail_case.csv`
  (`ego_r_RING_A2_A4_A2_A4`, offset 1.0s, outcome wrong_lane)

## 9. 실행 명령

```bash
# 기준 학습 (Phase 1, seed 지정 가능)
python train_parallel.py --phase 1 --num-envs 256 --num-workers 4 \
  --steps-per-env 64 --iterations 500 --learning-epochs 10 --minibatch-size 512 \
  --randomize-departures False --eval-interval 0 --seed 0

# 개발 평가
python scripts/evaluate_quick.py results/<실제_run_폴더>

# 별도 offset 평가
python scripts/evaluate_quick.py results/<실제_run_폴더> --offset-set separate

# 랜덤 출발 차선 진단
python scripts/evaluate_quick.py results/<실제_run_폴더> --randomize-departures

# 단일 에피소드 trace
python scripts/trace_evaluation.py results/<실제_run_폴더> <route> <offset> \
  --output results/<실제_run_폴더>/analysis/traces/example.csv
```

## 10. 검증 환경

- Python 3.11.9, PyTorch 2.11.0+cu128, CUDA, libsumo 1.28.0
- 기준 브랜치: `parallel_train`, 기준 commit `1b8fb31` (작업 후 `ebafef3`로 갱신)
- 게시 준비 검증: trace, parallel PPO, PPO time limits, stop clearance, parallel SUMO 테스트 총 22 passed 및 6 subtests passed.
- 게시된 자료 manifest 전체 SHA256과 최종 6개 평가 CSV의 행 수·고유 조건·모델 해시를 재검증했다. 장기 학습은 재실행하지 않았다.

## 게시된 검증 자료

실제 모델·저장 설정·학습 로그·평가 CSV/JSON·수정 후 trace는
[실험 자료](../experiments/phase1_20261113/README.md)에 포함했다.
본문의 `results/` 경로는 원래 로컬 실행 위치이며, 게시본은 같은 run 이름으로 보관했다.
