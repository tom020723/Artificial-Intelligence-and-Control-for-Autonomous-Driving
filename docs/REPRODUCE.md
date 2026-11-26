# 전체 과정 재현

평가 세트 생성 → IDM 전문가 수집 → BC 학습 → BC 평가 → PPO 파인튜닝 → PPO 평가를 한 번에 실행한다.

```bash
PY=~/anaconda3/envs/isaaclab2/bin/python PPO_PY=~/.holosoma_deps/miniconda3/envs/sumo/bin/python \
  bash scripts/reproduce_pipeline.sh               # 전체 (약 4시간)
SMOKE=1 bash scripts/reproduce_pipeline.sh         # 모든 단계를 최소 규모로 (약 3분)
```

| 환경 변수 | 기본값 | 내용 |
|---|---|---|
| `OUT` | `results/reproduce_<시각>` | 결과 폴더. 같은 `OUT`으로 다시 실행하면 끝난 단계(`OUT/.done_<단계>`)는 건너뛴다 |
| `PY` | `python` | 0–3단계 Python |
| `PPO_PY` | `$PY` | 4–5단계(PPO 학습과 평가) Python |
| `SUMO_HOME` | `/usr/share/sumo` | `tools/`를 `PYTHONPATH`에 넣는다. SUMO 1.28.0 |
| `WORKERS` | `12` | 전문가 수집 프로세스 수 |

최종 모델은 단계별로 다른 Python 환경에서 만들었다. 같은 결과를 내려면 같은 환경을 써야 한다.

| 단계 | 환경 | torch / CUDA | numpy |
|---|---|---|---|
| 0–3 (세트, 수집, BC 학습, BC 평가) | `~/anaconda3/envs/isaaclab2` | 2.7.0 / 12.8 | 1.26.0 |
| 4–5 (PPO 학습, PPO 평가) | `~/.holosoma_deps/miniconda3/envs/sumo` (셸의 `python`) | 2.14.1 / 13.0 | 2.2.6 |

torch/CUDA 버전이 다르면 GPU 연산 결과가 마지막 자릿수에서 달라진다. 그 차이가 시뮬레이션을 거치며 커져서 PPO 학습이 다른 모델로 끝난다. 같은 환경 안에서는 PPO 학습도 매번 같은 결과를 낸다.

중간에 실패한 단계는 그 단계의 출력 폴더를 지운 뒤 다시 실행한다. 수집과 PPO 학습은 이미 있는 출력 폴더에 쓰지 않는다.

전체 실행에는 여유 공간이 약 20GB 필요하다. 수집 원본과 선별 데이터셋이 합쳐 약 8GB이고, 수집 스크립트는 여유 공간이 8GB 아래로 내려가면 멈춘다.

## 단계

시간은 최종 모델을 만들 때 걸린 시간이다.

| 단계 | 스크립트 | 출력 (`OUT` 아래) | 시간 |
|---|---|---|---|
| 0. 고정 경로 평가 세트 | `scripts/evaluate_fixed_routes.py prepare` | `suite_fixed_routes/` | 수 초 |
| 1–2. 전문가 수집, BC 학습 | `scripts/measure_calibrated_idm_coverage.py` | `BC/` (`BC/dataset/` 약 4.8GB, `BC/training/model.pt`) | 수집 1.6시간, 학습 1.2시간 |
| 3. 무작위 출발 차선 세트, BC 평가 | `scripts/evaluate_diverse_start_lanes.py` | `suite_diverse_test/`, `suite_diverse_validation/`, `BC/evaluation_diverse_start_lanes/` | 수 분 |
| 4. PPO 파인튜닝 | `python -m scripts.finetune_bc_ppo` | `PPO/` | 24분 (GPU) |
| 5. 체크포인트 선택, PPO 평가 | `scripts/ppo_checkpoints.py`, `scripts/evaluate_diverse_start_lanes.py` | `PPO/selection.json`, `PPO/eval_runs/iterNNN/` | 평가당 1–2분 |

### 0. 고정 경로 평가 세트

```bash
python scripts/evaluate_fixed_routes.py prepare OUT/suite_fixed_routes \
  --traffic-vph 4740 --idm-only-speed-gain
```

- 도로(`env/road_config.py`), 배경 교통 4,740대/시간, 800초 예열 뒤 802–950초의 교통 상태 75개를 만든다. 이후 모든 단계가 이 세트를 입력으로 쓴다.
- 세트의 설정(`suite_fixed_routes/reference/`)은 저장소 코드(`env/`, `train.py`)와 `benchmark/`의 파일 세 개로만 만든다(`benchmark/README.md`). 이전 실험 결과는 읽지 않는다.
- 배경 차량은 Krauss 35% / IDM 30% / EIDM 20% / ACC 15%이고, `--idm-only-speed-gain`은 IDM만 속도를 위해 추월하게 한다(`lcSpeedGain` IDM 1.0, 나머지 0.0).
- 최종 모델을 만들 때는 같은 설정을 이전 실험의 사본(`results/eval_fixed_routes_20261119_214714/reference`)에서 복사했다. `benchmark/`는 그 사본에서 결과에 쓰이는 값만 옮긴 것이고, 두 방식의 축소판 재현이 모든 단계에서 같은 결과를 낸다(아래 "같게 나오는 것과 아닌 것").
- 원래 세트 `results/eval_fixed_routes_4740_idm_only_speedgain_20261121_013051`은 이 옵션이 생기기 전에 같은 처리를 하는 임시 스크립트로 만들었다. 지금 명령으로 다시 만들면 사례 목록, 교통 상태, SUMO 입력 파일이 원래 세트와 같다. 다른 곳은 도로망 파일의 생성 시각 주석과 `suite.json`의 기록용 값 `spawn_protect_m`(30 → 0, 평가에서 읽지 않음)뿐이다.

### 1–2. 전문가 수집, BC 학습

```bash
python -u scripts/measure_calibrated_idm_coverage.py OUT/suite_fixed_routes OUT/BC \
  --workers 12 --train-quota 19000 --validation-quota 1000 --epochs 100 --lane-weight-root 4 \
  --connected-lane-bumper-gaps --signed-adjacent-gaps --idm-spawn-protect-m 20
```

- 알고리즘은 `reproducible-pipeline` 브랜치의 `docs/final_BC_trianing.md`. 그 문서의 처음 표는 기본 설정이고, 최종 BC는 개선 사항 1, 3, 4, 5를 더한 것이다.

  | 옵션 | 개선 사항 |
  |---|---|
  | `--lane-weight-root 4` | 1. 차선 변경 가중치를 네제곱근으로 완화 |
  | `--connected-lane-bumper-gaps` | 3. 앞차가 다른 도로에 있을 때의 차간거리 |
  | `--signed-adjacent-gaps` | 4. 좌우 차선 차간거리의 음수 허용 |
  | `--idm-spawn-protect-m 20` | 5. IDM 전문가의 출발 20m 차선 변경 금지와 출발 위치 관측(52차원 관측) |

- 수집이 끝나면 `scripts/train_bc_shards.py`로 학습한다(seed 0, 100 epoch, batch 8192, LR 0.00042, 256×256 ReLU). 검증 손실이 가장 낮은 epoch가 `BC/training/model.pt`다. 최종 BC는 88 epoch였다.
- 학습 뒤 스크립트가 고정 경로 평가(`BC/evaluation/`)를 함께 돌린다. 지금 코드는 ego 규칙 장치를 끈 상태가 기본이라, 장치가 켜져 있던 당시 결과와 수치가 다르다.

### 3. 무작위 출발 차선 세트, BC 평가

```bash
python scripts/evaluate_diverse_start_lanes.py prepare OUT/suite_fixed_routes OUT/suite_diverse_test        # 시드 52101–52125, 7,050회
python scripts/evaluate_diverse_start_lanes.py prepare OUT/suite_fixed_routes OUT/suite_diverse_validation \
  --seeds 62101 62102 62103 62104 62105 62106 62107 62108 62109 62110                                    # 2,820회
python scripts/evaluate_diverse_start_lanes.py evaluate OUT/suite_diverse_test \
  --model OUT/BC/training/model.pt --output OUT/BC/evaluation_diverse_start_lanes
```

- 테스트 세트는 최종 판단에만 쓰고, 검증 세트는 PPO 체크포인트를 고르는 데 쓴다.
- 평가는 사례마다 SUMO를 새로 시작한다(`reproducible-pipeline` 브랜치 `docs/ppo_finetuning.md`의 "평가 재현성").

### 4. PPO 파인튜닝

```bash
PPO_PY -m scripts.finetune_bc_ppo OUT/BC/training/model.pt OUT/suite_fixed_routes \
  --iterations 100 --init-std 0.05 --lane-sharpness 2 --lr 5e-5 --anchor-coef 0.03 --overspeed-penalty 0 --seed 0 \
  --workers 16 --output OUT/PPO
```

- 환경 512개를 SUMO 프로세스 16개(프로세스당 32개)에 나눈다. 반복당 약 5.5초로 4개(13.5초)보다 2.5배 빠르다(2026-11-26 측정, GPU 사용률 1–3%). 작업자 수를 바꾸면 SUMO 난수 흐름이 달라져 결과가 seed를 바꾼 것처럼 달라진다. 이전 최종 모델은 작업자 4개로 학습했다.

방법과 보상은 `reproducible-pipeline` 브랜치의 `docs/ppo_finetuning.md`. 10회 반복마다 `model_iterNNN.pt`를 저장한다.

### 5. 체크포인트 선택, PPO 평가

```bash
python scripts/ppo_checkpoints.py export OUT/PPO --iterations 50 60 70 80 90 100
for i in 050 060 070 080 090 100; do
  PPO_PY scripts/evaluate_diverse_start_lanes.py evaluate OUT/suite_diverse_validation \
    --model OUT/PPO/eval_runs/iter$i/model.pt --output OUT/PPO/eval_runs/iter$i/evaluation_validation
done
python scripts/ppo_checkpoints.py select OUT/PPO --evaluation evaluation_validation
PPO_PY scripts/evaluate_diverse_start_lanes.py evaluate OUT/suite_diverse_test \
  --model OUT/PPO/eval_runs/iterNNN/model.pt --output OUT/PPO/eval_runs/iterNNN/evaluation_diverse_start_lanes
```

- `export`는 평가 스크립트가 읽는 BC 폴더 형식(BC의 `config.json`, `mdp_config.py`, `normalization.npz`, `road_config.py`, `train.py`)에 PPO 체크포인트를 `model.pt`로 넣는다. PPO는 BC와 신경망 구조와 관측이 같다.
- `select`는 검증 성공률이 가장 높은 체크포인트를 `PPO/selection.json`에 쓴다. 같으면 뒤의 체크포인트를 고른다.
- 최종 PPO는 반복 100이다. 당시 검증은 SUMO를 공유하던 이전 방식이었고 반복 100이 가장 높았다(96.38%). 지금 방식으로 검증하면 반복 70(96.42%)과 반복 100(96.38%)이 2,820회 중 1회 차이라 스크립트는 반복 70을 고른다. 이 차이는 평가 잡음 안이므로 최종 모델은 바꾸지 않았다. 테스트 결과는 반복 70이 96.60%, 반복 100이 97.06%다.

## 같게 나오는 것과 아닌 것

2026-11-25에 전체를 다시 돌려(`results/reproduce_20261125`, 약 4시간) 원래 결과와 비교했다. 위 표의 환경을 썼다. 이 재현은 세트 설정을 이전 실험의 사본에서 복사하던 방식이었다.

이어서 세트 설정을 `benchmark/`에서 만드는 지금 방식으로 바꾸고 세 가지를 확인했다.

- 지금 방식으로 만든 고정 경로 세트가 원래 최종 세트와 같다. 사례 목록, 교통 상태, SUMO 입력, 도로 설정이 모두 같다.
- 두 방식으로 축소판(`SMOKE=1`)을 한 번씩 돌린 결과가 모든 단계에서 같다. 수집 데이터셋, BC `model.pt`, PPO 체크포인트는 바이트 단위로 같다. 나머지 파일은 기록된 출력 경로, 생성 시각, 실행 시간, 저장 순서만 다르다.
- 지금 방식으로 전체를 다시 돌린 결과(`results/reproduce_20261125_benchmark`, 3시간 34분)가 첫 재현과 같다. 선별 데이터셋, BC 학습 로그와 `model.pt`, PPO 체크포인트가 바이트 단위로 같고, BC 평가와 PPO 검증·테스트 평가의 사례별 결과도 모두 같다.

저장소를 `results/`와 `data/` 없이 복사한 곳에서도 축소판이 끝까지 돈다.

| 단계 | 원래와 비교 |
|---|---|
| 0, 3. 평가 세트 | 사례, 교통 상태, SUMO 입력이 같다 |
| 1. 전문가 수집 | 1,824회, 5억 5,474만 쌍. 선별 데이터셋 1,827개 파일이 바이트 단위로 같다 |
| 2. BC 학습 | 100 epoch 손실 곡선, 선택 epoch(88), `model.pt`가 바이트 단위로 같다 |
| BC 평가 | 고정 경로 1,800건의 결과가 모두 같다(94.56%). 무작위 출발 차선 테스트는 90.04%(원래 90.03%는 SUMO 공유 방식) |
| 4. PPO 학습 | 100회 반복의 학습 로그, 모든 체크포인트, critic이 바이트 단위로 같다 |
| 5. PPO 평가 | 반복 100의 테스트 7,050건이 모두 같다(97.06%). 체크포인트 선택은 위 5단계 설명대로 반복 70이 된다 |

- 평가는 같은 모델과 같은 세트면 결과가 같다(사례별 SUMO seed).
- PPO만 0–3단계 환경(torch 2.7.0)으로 돌리면 반복 1에서 정책 출력이 약 10⁻⁹ 차이 나고, 반복 2부터 주행이 갈라져 다른 모델이 된다. 그 모델은 테스트 95.99%였다(`results/reproduce_20261125/PPO_isaaclab2`). 이 차이는 학습 시드를 바꾼 것과 같은 효과로, 이 절차의 PPO는 테스트 96–97% 정도이고 최종 PPO는 그중 좋은 편이다(학습 시드 1, 2의 검증 성공률은 94.5%, 94.9%). 어느 PPO든 BC보다 6%p 이상 높다.

## 최종 결과물

120초 묶음 신호, 4,700대/시간 기준(2026-11-26). BC와 세트는 `OUT=results/final3`로 이 절차를 돌린 출력이고, PPO는 같은 BC에서 lane 변경 벌점 0.01로 학습한 것이다(지금 재현 스크립트의 4단계 설정과 같다). git에는 세트, 모델, 설정, 평가 요약과 사례별 결과만 넣었다.

평가는 교차로 안 충돌까지 세는 것이 기본이다(SUMO `--collision.check-junctions`, 평가 스크립트의 기본값; 학습과 수집은 SUMO 기본값인 꺼짐). 검사를 끈 결과는 `*_no_junction_check` 폴더에 함께 둔다.

| 항목 | 경로 | 테스트 7,050건 (검사 켬) | 검사 끔 |
|---|---|---|---|
| 고정 경로 평가 세트 | `results/final/suite_fixed_routes` | | |
| 무작위 출발 차선 테스트·검증 세트 | `results/final/suite_diverse_test`, `results/final/suite_diverse_validation` | | |
| 최종 BC | `results/final/BC/training/model.pt` (epoch 100) | 90.79% (고정 경로 92.94%) | 91.33% (고정 경로 94.61%) |
| 최종 PPO | `results/final/PPO/eval_runs/iter080/model.pt` (검증 96.49%로 선택) | **96.84%** | 97.52% |

모델 폴더에는 수집 설정의 사본(`collection_config.json`)이 있어 데이터셋 없이도 평가할 수 있다.

### 학습 데이터셋

BC를 학습한 데이터셋(shard 2,269개와 `manifest.json`, `config.json`, 약 4.4GB)은 git에 넣지 않고 이 컴퓨터의 `results/final3/BC/dataset`에 둔다. 파일별 SHA-256은 `results/final/BC/dataset/SHA256SUMS`에 있다. 그 폴더에서 `sha256sum -c` 로 확인한다. 수집 원본(약 30GB)은 지웠고, 필요하면 재현 스크립트 1단계로 같은 데이터를 다시 만든다.

## 정리된 파일

이 브랜치(`final2`)에는 최종 과정이 쓰는 파일과 그 테스트만 남겼다. 이전 실험 스크립트, 문서, 결과는 `reproducible-pipeline` 브랜치와 git 기록에 있다. 이전 스크립트에서 가져다 쓰던 함수는 아래로 옮겼다.

| 함수 | 지금 위치 |
|---|---|
| 평가 컨텍스트 생성 `evaluation_context`, 액터 불러오기 `load_actor` | `utils/bc_policy.py` |
| 결과 집계 `summarize`, 상태 라벨 `OUTCOMES` | `utils/evaluation_summary.py` |
| BC 학습 곡선 `plot_bc_training`, 세트 도로 준비 `prepare_road` | `scripts/evaluate_fixed_routes.py` |
| DAgger 전문가 작업자 `native_expert_worker` | `scripts/measure_calibrated_idm_coverage.py` |
| 저장된 설정 복원 `load_saved_configs` | `utils/saved_configs.py` |
| BC 학습 함수 `epoch`, `lane_class`, `export_policy`, `initialize_from_raw_policy` | `scripts/train_bc_shards.py` |

`SMOKE=1`로 돌렸을 때 저장소에서 쓰이는 파일은 아래 44개다(Python 파일이 연 파일을 모두 기록해 확인). `results/`, `data/`, `archive/`, `docs/`, `tests/`는 쓰지 않는다.

| 폴더 | 파일 |
|---|---|
| `scripts/` | `reproduce_pipeline.sh`, `evaluate_fixed_routes.py`, `measure_calibrated_idm_coverage.py`, `train_bc_shards.py`, `evaluate_diverse_start_lanes.py`, `finetune_bc_ppo.py`, `ppo_checkpoints.py` |
| 최상위 | `train.py`(보상·하이퍼파라미터), `collect_background_data.py`(IDM 관측 수집기) |
| `benchmark/` | `road_context.py`, `mdp_context.py`, `traffic_curriculum.json` |
| `env/` | `__init__`, `road_config`, `mdp_config`, `sumo_env`, `road_builder`, `route_guidance`, `ring_observation`, `connected_lane_gaps`, `collision_detection`, `control_timing`, `ego_routes`, `bc_ppo_reward`, `parallel_sumo`, `parallel_traffic`, `traffic_snapshots`, `background_demand`, `lane_inflow`, `adaptive_traffic_schedule`, `uniform_idm` |
| `algorithms/` | `__init__`, `ppo`, `parallel_ppo`, `anchored_ppo` |
| `utils/` | `__init__`, `bc_policy`, `evaluation_summary`, `saved_configs`, `evaluator`, `networks`, `buffer`, `logger`, `bc_results` |
