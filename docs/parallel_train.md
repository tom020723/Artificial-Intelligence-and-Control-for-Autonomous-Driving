# Phase 1 병렬 학습

프로젝트 루트에서 SUMO와 `libsumo`를 사용할 수 있는 Python 환경으로 실행한다. 현재 병렬 환경은 모든 신호가 청신호이고 배경 차량이 없는 Phase 1 설정을 사용한다.

```bash
python train_parallel.py --phase 1
```

`--phase`의 기본값은 `1`이며, 현재 지원하는 단계는 Phase 1이다. [도로 설정](../env/road_config.py)의 `ROUTE_ONLY_EXPERIMENT`를 `True`로 둔다.

기본값은 도로 **1,024개**(`--num-envs`), 차량당 **32 step**(`--steps-per-env`), PPO **100 iterations**(`--iterations`), 미니배치 **1,024**(`--minibatch-size`)다. Iteration당 32,768 step, 전체 3,276,800 step을 수집하며 무작위 가중치에서 시작한다. Iteration당 수집량은 미니배치 크기의 배수여야 한다.

## 구현 방식

- [도로 설정](../env/road_config.py)으로 [원본 도로](../env/road_builder.py)를 만들고, [병렬 환경](../env/parallel_sumo.py)이 서로 연결되지 않은 도로 복제본을 **SUMO 인스턴스 하나**에 넣는다. 도로당 ego 한 대를 두어 차량 간 간섭을 없앤다.
- 모든 ego에 같은 PPO 정책으로 계산한 행동을 적용한 뒤 SUMO를 한 step 진행한다. 관측·보상·종료 판정은 [기존 환경](../env/sumo_env.py)을 재사용하고, 에피소드가 끝난 차량만 재투입한다.
- 한 iteration은 rollout 수집과 PPO 학습으로 구성된다. [병렬 PPO](../algorithms/parallel_ppo.py)는 차량마다 GAE를 계산한 뒤 경험을 합쳐 학습한다.
- 정기 평가는 기본 **10 iterations마다** 실행한다. 별도 SUMO 프로세스에서 각 복제 도로를 결정적 정책으로 한 번씩 주행해 기본 **1,024개 에피소드**를 집계한다. 매번 같은 평가 seed를 사용한다. 간격은 `--eval-interval`로 바꿀 수 있고, `0`이면 정기 평가를 끈다.

기본 장치 설정에서는 CUDA를 사용할 수 있으면 정책 추론과 PPO 학습을 GPU에서 실행한다. SUMO 시뮬레이션과 관측·보상 계산은 CPU에서 수행한다.

결과는 `results/parallel_<phase>_<시간>/`에 저장된다. `parallel_config.json`에서 환경 수 등 실행 설정을, `training_log.csv`와 `eval_log.csv`에서 학습·평가 기록을 확인할 수 있다. 최종 가중치는 `model.pt`이며, 생성한 복제 도로 파일은 `parallel/`에 들어간다.

# Phase 2 병렬 학습 (신호등 작동, 배경 차량 없음)

```bash
python train_parallel.py --phase 2 --num-envs 256 --steps-per-env 64 --iterations 244 --minibatch-size 512 --eval-interval 0     --set reward.off_route_lane_penalty=0.05 --set reward.red_stop_penalty_coef=0.1 --set reward.red_stop_distance_scale=50
python scripts/evaluate_parallel.py results/parallel_2_<시간>
```

위 설정은 아래 "정지 위치 실험"에서 고른 값이다. 244 iterations = 약 400만 step. seed는 `--set hparams.seed=값`으로 바꾼다.

- `--phase 2`는 `road_config.py`의 `signal_plan` 그대로 신호를 켠다. `ROUTE_ONLY_EXPERIMENT` 값은 바꿀 필요 없다.
- 복제 도로마다 신호 주기의 시작 시점을 다르게 준다. 한 번의 rollout 안에 초록·노랑·빨강이 모두 섞여 들어온다.
- **Phase 1 모델을 불러오지 말고 무작위 가중치에서 시작한다.** Phase 1 모델은 가속 출력이 최대값에 붙어 있어 이어 학습해도 정지를 배우지 못했다 (아래 표).
- `--set reward.키=값`, `--set hparams.키=값`, `--set obs.키=값`으로 파일 수정 없이 설정을 바꿔 실험할 수 있다. 바꾼 값은 `parallel_config.json`에 남는다.
- `--signal-green MIN MAX`: 도로마다 초록불 길이를 무작위로 정한다. `scripts/evaluate_parallel.py`에도 같은 옵션이 있어 학습 때 보지 못한 신호 길이로 평가할 수 있다.
- 관측은 `docs/train.md`의 49차원(목적지 one-hot 제외, 신호 1개) 기준이다. `env/mdp_config.py` 기본값을 여기에 맞췄다.

## 2026-11-12 실험 결과

모두 seed 0 한 번, 256 에피소드 결정적 평가, 고정 신호(초록 20초, 주기 125초). 보상은 `train.py`의 현재 값.

| 시작 가중치 | 학습량 | 도착 | 신호 위반 | 잘못된 차선 |
|---|---:|---:|---:|---:|
| Phase 1 모델 그대로 (추가 학습 없음) | 0 | 55.1% | 38.7% | 6.3% |
| Phase 1 모델에서 이어 학습 | 98만 step | 54.7% | 5.1% | 40.2% |
| **무작위 가중치에서 시작** | 200만 step | **99.6%** | 0.4% | 0% |

- 무작위 시작 모델은 학습 때 보지 못한 신호 길이(초록 5~40초 무작위)에서도 256/256 도착했다.
- 남은 문제: 빨간불에서 정지선보다 13~45m 앞(도로별로 일정)에서 멈춘다. → 아래 "정지 위치 실험"에서 원인을 찾아 해결.
- 효과가 없었던 변경(각 1회 실험): 초기 탐험 폭 축소(`init_log_std=-1.0`), `gamma=0.995`, `wrong_lane_penalty=5`, `red_stop_distance_scale=20`, 적신호 거리 감점 삭제.

## 2026-11-12 정지 위치 실험 (빨간불에서 너무 일찍 멈추는 문제)

모두 256 에피소드 결정적 평가, 고정 신호(초록 20초, 주기 125초). 정지 거리는 빨간불에 멈춰 있는 동안(0.5 m/s 미만) 정지선까지의 거리.

### 원인

1. **교차로 내부 차선에서 관측이 튀었다.** 내부 차선(`:nM_A1_...` 등)에 들어가면 차선 연결성 9개가 전부 1, 밀도 5개가 전부 0으로 고정됐다. 정책은 정지선 거리 대신 이 특이한 입력을 "멈출 곳" 신호로 썼고, 그래서 접근 도로마다 일정한 지점(정지선 직전 edge가 시작되는 노드)에서 멈췄다. SUMO의 자동 제동은 꺼져 있어(`speedMode 0`) 정책 자신의 행동이다.
2. **잘못된 차선에 감점 없는 대기 자리가 있었다.** 경로로 이어지지 않는 차선에서는 차선 끝 5m 이내에 서야만 종료·감점된다. 그보다 뒤에 서 있으면 감점이 없어서, 정책이 "서서 우연한 차선변경(스텝당 1~3% 표본)을 기다리기"를 배웠다. 학습 중에는 빠져나오지만 결정적 평가(argmax)에서는 계속 서 있는다. 같은 모델을 차선 변경만 표본 추출해 평가하면 도착 89.5% → 98.4%.
3. **정지 위치를 정하는 보상이 약했다.** 적신호 거리 감점이 30m 앞에 서도 스텝당 약 0.013이라, 정지 위치가 학습마다 12~33m로 달라졌다. 이전에 시도한 `red_stop_distance_scale=20`은 23m보다 멀면 감점이 상한에 붙어 기울기가 없다.
4. **측정 오류.** 병렬 환경에서 300초 넘게 멈춘 차를 SUMO가 순간이동시켰고, 이때 늘어난 이동 거리가 신호 위반으로 집계됐다.

### 변경

- `env/sumo_env.py`: `_reference_lane` 추가. 내부 차선에서는 그 차선이 이어지는 다음 일반 차선을 기준으로 차선 연결성·밀도·lookahead 연결성을 계산한다. 관측 차원은 그대로지만 값의 의미가 바뀌어 **이전 모델과 호환되지 않는다.**
- `env/sumo_env.py`, `env/mdp_config.py`: `off_route_lane_penalty` 추가. 경로로 이어지지 않는 차선에 있고 그 끝이 시야 안인 동안(현재 차선 연결성 관측 < 1) 매 스텝 감점. 기본값 0(비활성).
- `env/parallel_sumo.py`: `--time-to-teleport -1`. 갇힌 차는 시간 초과로 끝난다. (단일 환경 `sumo_env.py`의 SUMO 옵션은 바꾸지 않았다.)

### 결과

단계별 (seed 0, 200만 step):

| 설정 | 도착 | 신호 위반 | 잘못된 차선 | 정지 거리 중간값 / 상위 10% |
|---|---:|---:|---:|---:|
| 변경 전 | 99.6% | 0.4% | 0% | 약 24m / 39m |
| 관측 수정 | 89.5% | 10.5% (순간이동 오집계) | 0% | 12.1m / 22m |
| + `off_route_lane_penalty=0.05` | 100% | 0% | 0% | 33.0m / 48m |
| + `red_stop_penalty_coef=0.1`, `red_stop_distance_scale=50` | 99.6% | 0.4% | 0% | **5.8m / 8.1m** |
| + `red_stop_penalty_coef=0.2`, `red_stop_distance_scale=50` | 54.7% | 5.1% | 40.2% | 학습 실패 |

최종 설정의 seed별 재현성:

| seed | 학습량 | 도착 | 신호 위반 | 잘못된 차선 | 정지 거리 중간값 / 상위 10% |
|---|---:|---:|---:|---:|---:|
| 0 | 200만 | 99.6% | 0.4% | 0% | 5.8m / 8.1m |
| 1 | 200만 | 61.3% | 9.8% | 28.9% | 27.9m / 64m |
| 1 | 200만 + 이어서 200만 | 100% | 0% | 0% | 10.4m / 20.4m |
| 2 | 200만 | 98.8% | 0.8% | 0.4% | 23.7m / 52m |
| 2 | 200만 + 이어서 200만 | 100% | 0% | 0% | 5.0m / 8.4m |

- 200만 step은 seed에 따라 부족하다 (seed 1은 훈련 보상이 5~6에서 오르지 않은 채 끝남). 400만 step에서는 세 seed 모두 도착 99.6% 이상.
- seed 1·2의 400만 step은 `--init-model`로 이어 학습한 것이다(옵티마이저 초기화). 위 권장 명령(244 iterations 한 번에)을 그대로 실행해 본 적은 없다.
- seed 1은 400만 step에서도 정지 거리가 10m(상위 10%는 20m)로 남는다.
- `red_stop_penalty_coef=0.2`는 너무 세서 학습이 무너졌다. 0.1 근처가 상한.
- `tests/test_lookahead_connectivity.py`의 6개 테스트는 관측 59차원을 기대해, 49차원 기본값에서는 이번 변경과 무관하게 실패한다.
