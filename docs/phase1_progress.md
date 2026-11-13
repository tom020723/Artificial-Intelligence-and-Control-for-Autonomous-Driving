# Phase 1 진행 기록

## 상태

**미달 (성능 미달).** 동일 학습 설정(256 × 64 × 500, 8,192,000 transition)의
seed 0/1/2 × 개발/별도 offset 6행 판정에서 seed 2가 기준을 충족하지 못했다.
후보 보상은 `red_stop_clearance_reward=0.02`,
`red_stop_clearance_reward_scale=2.0`, `red_stop_penalty_coef=0.02`이다.
seed 2는 동일 설정 재학습 결과 도착 1,050/1,150(91.3%), wrong-lane 100건으로
실패했다. 기준을 낮추거나 더 유리한 seed로 바꾸지 않고 미달로 판정한다.

## 검증 환경

- Python 3.11.9, PyTorch 2.11.0+cu128, CUDA, libsumo 1.28.0
- 최신 평가기: 고유 경로 23개 × offset 50개 = 1,150회
- 기준 안전 조건: 도착, 충돌 0, 잘못된 차선 0, 적신호 위반 0
- 정차 품질: 유효 표본 100개 이상, median 3~8m, p90 15m 이하

## 최종 후보 결과

기존 대규모 seed 2는 아래 표의 참고 수치로만 남기고, 동일 설정
재현성 표에는 넣지 않는다.

| 실행 | 도착 | 충돌 | 잘못된 차선 | 적신호 위반 | 정차 표본 | median | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 0 | 1,150/1,150 | 0 | 0 | 0 | 713 | 4.20m | 4.65m |
| seed 1 | 1,150/1,150 | 0 | 0 | 0 | 712 | 4.07m | 4.47m |
| (참고) 기존 seed 2 | 1,150/1,150 | 0 | 0 | 0 | 715 | 4.63m | 5.18m |
| **동일 설정 seed 2** | 1,050/1,150 | 0 | 100 | 0 | 753 | 25.21m | 41.80m |

## 6행 판정표 (동일 설정 seed 0/1/2 × 개발/별도)

| seed | 평가 | 도착/전체 | 경로 최소 | 위반 | 충돌 | unknown | 정차 n | median | p90 | 합격 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 개발 | 1,150/1,150 | 50 | 0 | 0 | 0 | 713 | 4.20m | 4.65m | ✅ |
| 0 | 별도 | 1,150/1,150 | 50 | 0 | 0 | 0 | 709 | 4.20m | 4.65m | ✅ |
| 1 | 개발 | 1,150/1,150 | 50 | 0 | 0 | 0 | 712 | 4.07m | 4.47m | ✅ |
| 1 | 별도 | 1,150/1,150 | 50 | 0 | 0 | 0 | 705 | 4.07m | 4.47m | ✅ |
| 2 | 개발 | 1,050/1,150 | 0 | 0 | 0 | 0 | 753 | 25.21m | 41.80m | ❌ |
| 2 | 별도 | 1,050/1,150 | 0 | 0 | 0 | 0 | 740 | 25.21m | 41.80m | ❌ |

동일 설정 seed 2는 개발·별도 모두 도착 1,050/1,150(91.3%)으로 전체 도착
기준(1,139회 이상)을 밑돌았고, wrong-lane 100건이 모두 Ring 경로 두 곳
(`ego_r_RING_A2_A4_A2_A4`, `ego_r_RING_A3_A5_A3_A5`, 각 50/50)에 집중됐다.
정차 median 25.21m·p90 41.80m도 기준(3~8m / ≤15m)을 크게 벗어났다.

seed 1은 development와 separate offset 모두 기준을 통과했다. seed 0의
separate offset 평가도 1,150/1,150 도착(정차 표본 709, median 4.20m,
p90 4.65m)으로 통과했다.

기존 seed 2는 저장된 실행 설정이 seed 0/1과 다르다. 전체 환경
**4,096**(워커 4 × 워커당 환경 1,024), 환경당 rollout **24**, iteration
**333**, epoch **5**, minibatch **2,048**로, 실제 **32,735,232**
transition이다. `1,024 × 24 × 333 ≈ 32.7M`이라는 기존 식은 잘못됐다.
따라서 기존 seed 2는 후보의 독립 검증으로만 참고하고, 동일 학습 예산의
엄밀한 seed 재현성 증거로 해석하지 않는다. 동일 설정 seed 2는
256 × 64 × 500(8,192,000 transition)으로 재학습했고 그 결과가 위 판정표의
seed 2 행이다.

모델 SHA256:

- seed 0: `df8bb61c748bf30f4037d3f711a1f7ff0d2369dc45ac370acc7c925af600c5a0`
- seed 1: `ad20cadc69195c91646ea09193432c48c8649de5bd850142ce3e433e5cb6e20a`
- (참고) 기존 seed 2: `7bab0b0e94ffcf04b72ad83c767f535bf4f84fc9f8568c8e4cec9fd5f73b61e3`
- **동일 설정 seed 2**: `f23a95c3207060aa637adeb3cd78dd2c4f9a67fd144e98fc29446e1c63b8d3ae`

## 보상 후보 비교

- clearance reward 0.20: 정차는 개선됐지만 wrong-lane 82건과 적신호 위반
  1건으로 폐기했다.
- clearance reward 0.05 (`parallel_1_20261113_134618_825826`): 정차
  median 22.03m / p90 30.49m로 폐기했다.
- clearance reward 0.02 + scale 2.0 + penalty 0
  (`parallel_1_20261113_140556_054790`): 정차 median 3.96m / p90 25.29m로,
  p90이 기준(≤15m)을 넘어 폐기했다.
- scale 2.0 및 penalty 0.02를 추가한 최종 후보는 seed 0/1(및 참고용 대규모
  seed 2)에서는 안전·도착·정차 기준을 만족했지만, 동일 설정 seed 2에서는
  실패해 최종적으로 재현되지 않았다(미달).

## 랜덤 출발 차선 진단

seed 1 모델에 `--randomize-departures`를 적용한 별도 진단은
1,115/1,150 도착(96.96%), wrong-lane 35건(3.04%), 충돌 0,
적신호 위반 0이었다. 실패는 A1 출발 경로에 집중됐다. 이는 중앙 차선
고정 학습 분포와 랜덤 출발 차선 분포 사이의 일반화 문제로 해석하며,
Phase 1 합격 기준을 완화하는 근거로 사용하지 않는다.

## 대표 trace

기존 trace는 총보상(`reward`) 열이 `info.get("reward")`(None)로 덮여 비어
있었다. `scripts/trace_evaluation.py`를 수정해 총보상을 `env.step` 반환값으로
기록하도록 고치고, 수정 후 결과를 `analysis/traces_verified/`에 다시
저장했다. 총보상 합과 평가 `ep_return`의 일치(오차 < 1e-4)를 검증했다.

- seed 0 정차 사례:
  - `case_1.csv`: `ego_d_A5_A1`, offset 103.5s (reward 합 33.0604 = ep_return)
  - `case_2.csv`: `ego_d_A4_A3`, offset 81.0s (32.1382)
  - `case_3.csv`: `ego_d_A1_A1`, offset 1.0s (32.1800)
- 랜덤 출발 차선 실패: seed 1 `analysis/traces_verified/wrong_lane_case.csv`
  (`ego_d_A1_A1`, offset 11.0s, reward 합 −0.9838 = ep_return, outcome wrong_lane)
- 동일 설정 seed 2 Ring 실패: `analysis/traces_verified/ring_fail_case.csv`
  (`ego_r_RING_A2_A4_A2_A4`, offset 1.0s, outcome wrong_lane)

각 파일은 edge/lane, 신호 상태와 거리, route 연결성, raw/applied lane
command, 거절 여부, 정차·적신호 보상 항을 step별로 기록한다.
`lane_change_rejected`는 거절 여부(불리언)일 뿐 거절 사유가 아니며,
내부 차선·없는 목표 차선·안전 간격 등 거절 사유는 이번에 수집하지 않았다.

## 게시된 검증 자료

실제 모델·저장 설정·학습 로그·평가 CSV/JSON·수정 후 trace는
[실험 자료](../experiments/phase1_20261113/README.md)에 포함했다.
본문의 `results/` 경로는 원래 로컬 실행 위치이며, 게시본은 같은 run 이름으로 보관했다.
