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
