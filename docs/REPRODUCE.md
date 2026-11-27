# 전체 과정 재현

평가 세트 생성 → IDM 전문가 수집 → BC 학습 → BC 평가 → PPO 파인튜닝 → PPO 평가를 한 번에 실행한다. 환경은 120초 묶음 신호, 배경 교통 4,700대/시간이다.

```bash
PY=~/anaconda3/envs/isaaclab2/bin/python PPO_PY=~/.holosoma_deps/miniconda3/envs/sumo/bin/python \
  bash scripts/reproduce_pipeline.sh               # 전체 (약 3.5시간)
SMOKE=1 bash scripts/reproduce_pipeline.sh         # 모든 단계를 최소 규모로 (약 3분)
```

| 환경 변수 | 기본값 | 내용 |
|---|---|---|
| `OUT` | `results/reproduce_<시각>` | 결과 폴더. 같은 `OUT`으로 다시 실행하면 끝난 단계(`OUT/.done_<단계>`)는 건너뛴다. 새로 돌리려면 새 폴더를 쓴다 |
| `PY` | `python` | 0–3단계 Python |
| `PPO_PY` | `$PY` | 4–5단계(PPO 학습과 평가) Python |
| `SUMO_HOME` | `/usr/share/sumo` | `tools/`를 `PYTHONPATH`에 넣는다. SUMO 1.28.0 |
| `WORKERS` | `12` | 전문가 수집 프로세스 수 |
| `SUMO_CHECK_JUNCTIONS` | `0` | `1`이면 수집과 PPO 학습에서도 교차로 안 충돌을 검사한다. 평가는 이 값과 무관하게 항상 검사한다(아래 "평가") |

최종 모델은 단계별로 다른 Python 환경에서 만들었다. 같은 결과를 내려면 같은 환경을 써야 한다.

| 단계 | 환경 | torch / CUDA | numpy |
|---|---|---|---|
| 0–3 (세트, 수집, BC 학습, BC 평가) | `~/anaconda3/envs/isaaclab2` | 2.7.0 / 12.8 | 1.26.0 |
| 4–5 (PPO 학습, PPO 평가) | `~/.holosoma_deps/miniconda3/envs/sumo` (셸의 `python`) | 2.14.1 / 13.0 | 2.2.6 |

torch/CUDA 버전이 다르면 GPU 연산 결과가 마지막 자릿수에서 달라진다. 그 차이가 시뮬레이션을 거치며 커져서 PPO 학습이 다른 모델로 끝난다. 같은 환경 안에서는 PPO 학습도 매번 같은 결과를 낸다.

중간에 실패한 단계는 그 단계의 출력 폴더를 지운 뒤 다시 실행한다. 수집과 PPO 학습은 이미 있는 출력 폴더에 쓰지 않는다.

전체 실행에는 여유 공간이 약 40GB 필요하다. 수집 원본이 약 30GB, 선별 데이터셋이 약 4.4GB이고, 수집 스크립트는 여유 공간이 8GB 아래로 내려가면 멈춘다. 학습이 끝나면 수집 원본은 지워도 된다(BC 학습에는 선별 데이터셋만 쓴다).

## 단계

시간은 `results/final3`를 만들 때(2026-11-26) 걸린 시간이다.

| 단계 | 스크립트 | 출력 (`OUT` 아래) | 시간 |
|---|---|---|---|
| 0. 고정 경로 평가 세트 | `scripts/evaluate_fixed_routes.py prepare` | `suite_fixed_routes/` | 수 초 |
| 1–2. 전문가 수집, BC 학습 | `scripts/measure_calibrated_idm_coverage.py` | `BC/` (`BC/dataset/` 약 4.4GB, `BC/training/model.pt`) | 수집 약 2시간, 학습 1.1시간 |
| 3. 무작위 출발 차선 세트, BC 평가 | `scripts/evaluate_diverse_start_lanes.py` | `suite_diverse_test/`, `suite_diverse_validation/`, `BC/evaluation_diverse_start_lanes/` | 2분 |
| 4. PPO 파인튜닝 | `python -m scripts.finetune_bc_ppo` | `PPO/` | 10분 (GPU) |
| 5. 체크포인트 선택, PPO 평가 | `scripts/ppo_checkpoints.py`, `scripts/evaluate_diverse_start_lanes.py` | `PPO/selection.json`, `PPO/eval_runs/iterNNN/` | 6분 |

### 0. 고정 경로 평가 세트

```bash
python scripts/evaluate_fixed_routes.py prepare OUT/suite_fixed_routes --idm-only-speed-gain
```

- 도로(`env/road_config.py`), 배경 교통(보정값 4,700대/시간, `env/background_demand.py`), 800초 예열 뒤 802–950초의 교통 상태 75개를 만든다. 이후 모든 단계가 이 세트를 입력으로 쓴다.
- 세트의 설정(`suite_fixed_routes/reference/`)은 저장소 코드(`env/`, `train.py`)와 `benchmark/`의 파일 세 개로만 만든다(`benchmark/README.md`). 이전 실험 결과는 읽지 않는다.
- 배경 차량은 Krauss 35% / IDM 30% / EIDM 20% / ACC 15%이고, `--idm-only-speed-gain`은 IDM만 속도를 위해 추월하게 한다(`lcSpeedGain` IDM 1.0, 나머지 0.0).

### 1–2. 전문가 수집, BC 학습

```bash
python -u scripts/measure_calibrated_idm_coverage.py OUT/suite_fixed_routes OUT/BC \
  --workers 12 --train-quota 19000 --validation-quota 1000 --epochs 100 --lane-weight-root 4 \
  --connected-lane-bumper-gaps --signed-adjacent-gaps --idm-spawn-protect-m 20
```

- 새 SUMO 시뮬레이션(3,600초)을 반복 실행하면서 모든 `car_idm` 배경 차량의 닫힌 궤적을 ego와 같은 관측으로 기록한다. 경로 24개마다 학습 19,000개, 검증 1,000개 궤적을 출발 시각 세 구간에 고르게 고른다. 출발 lane은 SUMO가 무작위로 정하므로 따로 맞추지 않는다. 최종 데이터셋은 2,268회 실행에서 고른 9,876만 쌍이다.
- 이어서 `scripts/train_bc_shards.py`로 학습한다(seed 0, 100 epoch, batch 8192, LR 0.00042, 256×256 ReLU). 검증 손실이 가장 낮은 epoch가 `BC/training/model.pt`다(최종 BC는 epoch 100).
- 학습 뒤 스크립트가 고정 경로 평가(`BC/evaluation/`)를 함께 돌린다.

| 옵션 | 내용 |
|---|---|
| `--lane-weight-root 4` | lane 변경 class 가중치를 빈도 역수의 네제곱근으로 완화 |
| `--connected-lane-bumper-gaps` | 앞차가 다른 도로에 있을 때도 lane 연결을 따라 bumper 간격을 잰다 |
| `--signed-adjacent-gaps` | 옆 lane 간격이 겹치면 음수를 허용 |
| `--idm-spawn-protect-m 20` | IDM 전문가는 출발 20m 안에서 lane을 바꾸지 않게 하고, 출발 위치 관측(52번째 값)을 넣는다 |

### 3. 무작위 출발 차선 세트, BC 평가

```bash
python scripts/evaluate_diverse_start_lanes.py prepare OUT/suite_fixed_routes OUT/suite_diverse_test        # seed 52101–52125, 7,050건
python scripts/evaluate_diverse_start_lanes.py prepare OUT/suite_fixed_routes OUT/suite_diverse_validation \
  --seeds 62101 62102 62103 62104 62105 62106 62107 62108 62109 62110                                    # 2,820건
python scripts/evaluate_diverse_start_lanes.py evaluate OUT/suite_diverse_test \
  --model OUT/BC/training/model.pt --output OUT/BC/evaluation_diverse_start_lanes
```

- 두 세트는 같은 94개 조합(경로 × 출발 lane)을, seed만 다른 교통 상태(seed당 3개)에서 평가한다. 두 세트의 seed는 서로 겹치지 않고, PPO 학습 교통과도 겹치지 않는다.
- 테스트 세트는 최종 판단에만 쓰고, 검증 세트는 PPO 체크포인트를 고르는 데 쓴다.

### 4. PPO 파인튜닝

```bash
PPO_PY -m scripts.finetune_bc_ppo OUT/BC/training/model.pt OUT/suite_fixed_routes \
  --iterations 100 --init-std 0.05 --lane-sharpness 2 --lr 5e-5 --anchor-coef 0.03 --overspeed-penalty 0 \
  --lane-change-penalty 0.01 --seed 0 --workers 16 --output OUT/PPO
```

- BC actor에서 시작하고 BC policy에 KL anchor를 건다. 보상은 사건 보상(도착 +1, wrong lane·시간 초과 −1, 충돌·빨강 신호 −2, step −10⁻⁴, 빨강 신호 앞 정지 보너스)과 실행된 lane 변경 1회당 −0.01이다. 속도나 진행 보상은 없다.
- 환경 512개를 SUMO 프로세스 16개(프로세스당 도로 32벌 복제)에 나눈다. 반복당 약 5.5초로 4개(13.5초)보다 2.5배 빠르다(GPU 사용률 1–3%). 작업자 수를 바꾸면 SUMO 난수 흐름이 달라져 결과가 seed를 바꾼 것처럼 달라진다.
- 10회 반복마다 `model_iterNNN.pt`를 저장한다. 알고리즘과 하이퍼파라미터는 `docs/report/report_ko.tex`와 `docs/algorithm_bc_ppo_ko.tex`에 있다.

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

- `export`는 평가 스크립트가 읽는 BC 폴더 형식(BC의 `config.json`, `collection_config.json`, `mdp_config.py`, `normalization.npz`, `road_config.py`, `train.py`)에 PPO 체크포인트를 `model.pt`로 넣는다. PPO는 BC와 신경망 구조와 관측이 같다.
- `select`는 검증 성공률이 가장 높은 체크포인트를 `PPO/selection.json`에 쓴다. 같으면 뒤의 체크포인트를 고른다. 마지막 체크포인트가 가장 좋다는 보장은 없다(최종 실행에서 반복 100은 테스트 93.38%, 검증으로 고른 반복 80은 96.84%).

## 평가

- **교차로 안 충돌:** 평가는 교차로 안 충돌까지 센다(SUMO `--collision.check-junctions true`). 두 평가 스크립트의 기본값이고, 결과 파일에 `"junction_collisions": true`로 남는다. 끄려면 `--no-junction-check`를 준다. 전문가 수집과 PPO 학습은 SUMO 기본값(검사하지 않음)을 쓴다. 학습에서 켜 보았을 때는 교차로 상충 차량이 관측에 없어 성능이 오히려 떨어졌다.
- **재현성:** 무작위 출발 차선 평가는 사례마다 SUMO를 새로 시작하고 사례별 seed를 쓴다. 같은 모델과 세트면 작업자 수나 사례 순서와 무관하게 결과가 같다.
- **결과 그림:** 평가가 끝나면 결과 폴더에 `summary.png`를 저장한다. 결과별 건수와 비율, 경로별 결과, 경로 × 출발 lane 성공률, 정지 거리, 학습 곡선(BC는 epoch별 손실과 lane 지표, PPO는 rollout 결과, 체크포인트별 검증 성공률, BC와의 KL, 탐색 폭)이 들어 있다. 평가를 다시 돌리지 않고 그림만 다시 그리려면 `python scripts/evaluate_fixed_routes.py replot <평가 폴더>`를 쓴다.

## 같게 나오는 것과 아닌 것

- 같은 Python 환경이면 전 과정이 바이트 단위로 같다. 이전 최종 모델(125초 신호)로 전체를 두 번 돌려(2026-11-25), 선별 데이터셋, BC 학습 로그와 `model.pt`, PPO 체크포인트, BC·PPO 평가의 사례별 결과가 모두 같음을 확인했다.
- 지금 브랜치에서는 `SMOKE=1`로 모든 단계가 끝까지 돌고 평가가 교차로 안 충돌을 세는 것을 확인했다(2026-11-26). 전체 테스트는 `python -m unittest discover -s tests`(155개)로 돌린다.
- `results/final3`는 lane 변경 벌점을 넣기 전의 스크립트로 돌린 출력이라, 그 안의 `PPO/`는 벌점 없는 실행이다. 최종 PPO는 같은 BC에서 지금 스크립트의 4단계와 같은 설정(벌점 0.01, 16 작업자, seed 0)으로 학습했다.
- PPO만 0–3단계 환경(torch 2.7.0)으로 돌리면 반복 1에서 정책 출력이 약 10⁻⁹ 차이 나고, 반복 2부터 주행이 갈라져 다른 모델이 된다. 이 차이는 학습 seed를 바꾼 것과 같은 효과다. 이 절차의 PPO는 검증으로 고르면 테스트 96.5–97.5%(교차로 안 충돌 제외 기준)에 모인다.

## 최종 결과물

`results/final`에 있다(git에 포함). BC와 세트는 `OUT=results/final3`로 이 절차를 돌린 출력이고, PPO는 같은 BC에서 lane 변경 벌점 0.01로 학습한 실행(`results/ablation_lane_penalty/bc_c0.01`)이다. git에는 세트, 모델, 설정, 평가 요약과 사례별 결과, 그림만 넣었다.

기본 평가 폴더(`evaluation`, `evaluation_diverse_start_lanes`, `evaluation_validation`)는 교차로 안 충돌을 센 결과이고, 세지 않은 결과는 `*_no_junction_check` 폴더에 함께 둔다.

| 항목 | 경로 | 테스트 7,050건 (교차로 검사 켬) | 검사 끔 |
|---|---|---|---|
| 고정 경로 평가 세트 | `results/final/suite_fixed_routes` | | |
| 무작위 출발 차선 테스트·검증 세트 | `results/final/suite_diverse_test`, `results/final/suite_diverse_validation` | | |
| 최종 BC | `results/final/BC/training/model.pt` (epoch 100) | 90.79% (고정 경로 92.94%) | 91.33% (고정 경로 94.61%) |
| 최종 PPO | `results/final/PPO/eval_runs/iter080/model.pt` (검증 96.49%로 선택) | **96.84%** | 97.52% |

모델 폴더에는 수집 설정의 사본(`collection_config.json`)이 있어 데이터셋 없이도 평가할 수 있다. 보고서는 `docs/report/report_ko.tex`이고, 그림은 `docs/report/figures/`(`scripts/plot_report_figures.py`로 생성)에 있다.

### 학습 데이터셋

BC를 학습한 데이터셋(shard 2,268개와 `manifest.json`, `config.json`, 약 4.4GB)은 git에 넣지 않고 이 컴퓨터의 `results/final3/BC/dataset`에 둔다. 파일별 SHA-256은 `results/final/BC/dataset/SHA256SUMS`에 있다. 그 폴더에서 `sha256sum -c`로 확인한다. 수집 원본(약 30GB)은 지웠고, 필요하면 재현 스크립트 1단계로 같은 데이터를 다시 만든다.

## 정리된 파일

이 브랜치에는 최종 과정이 쓰는 파일과 그 테스트, 과제 자료(`project_materials/`)만 남겼다. 이전 실험 스크립트, 문서, 결과는 `reproducible-pipeline` 브랜치와 git 기록에 있다. 이전 스크립트에서 가져다 쓰던 함수는 아래로 옮겼다.

| 함수 | 지금 위치 |
|---|---|
| 평가 컨텍스트 생성 `evaluation_context`, 액터 불러오기 `load_actor` | `utils/bc_policy.py` |
| 결과 집계 `summarize`, 상태 라벨 `OUTCOMES` | `utils/evaluation_summary.py` |
| BC 학습 곡선 `plot_bc_training`, 세트 도로 준비 `prepare_road` | `scripts/evaluate_fixed_routes.py` |
| 저장된 설정 복원 `load_saved_configs` | `utils/saved_configs.py` |
| BC 학습 함수 `epoch`, `lane_class`, `export_policy`, `initialize_from_raw_policy` | `scripts/train_bc_shards.py` |

재현 스크립트가 쓰는 파일은 `scripts/`의 `reproduce_pipeline.sh`, `evaluate_fixed_routes.py`, `measure_calibrated_idm_coverage.py`, `train_bc_shards.py`, `evaluate_diverse_start_lanes.py`, `finetune_bc_ppo.py`, `ppo_checkpoints.py`와 최상위 `train.py`(보상·하이퍼파라미터), `collect_background_data.py`(IDM 관측 수집기), 그리고 `benchmark/`, `env/`, `algorithms/`, `utils/`다. `results/`, `data/`, `docs/`, `tests/`는 쓰지 않는다.

## 제출용 주행 데이터 수집 (PR #7 통합)

`collect_background_data.py`는 BC 학습용 IDM 전문가 수집기이고,
`collect_pomdp_data.py`는 학습된 BC/PPO를 주행시켜 제출용 transition을 기록하는 별도 수집기다.

```bash
python collect_pomdp_data.py --model results/final/BC/training/model.pt --traffic-vph 4700 --episodes 10 --workers 2 --randomize-departures --name final_bc
python collect_pomdp_data.py --model results/final/PPO/eval_runs/iter080/model.pt --traffic-vph 4700 --episodes 10 --workers 2 --randomize-departures --name final_ppo
```

출력은 `data/<이름>.jsonl.gz`, `.npz`, `.metadata.json`, `.sumo/`다.
충돌 상대, 신호, 경로, 시드와 종료 원인을 함께 기록한다. 칼럼 정의는
`docs/data_columns_submission.md`에 있다. 최종 BC와 PPO를 자동 판별하며,
PPO 평가용 폴더에서는 상위 PPO 실행의 설정을 읽어 사건 보상과 차선 변경 벌점을 보존한다.
모델 수집은 ego 규칙 제한을 해제하고, 교차로 내부 충돌 검사도 기본으로 켠다
(`--no-junction-check`로 끌 수 있다). `--traffic-vph`를 주면 예열 기본값은 250초이며
`--warmup-seconds`로 바꿀 수 있다. 노형오거리의 배경차는 고정 평가와 같은
경로별 가중치로 배분하며, 입력한 총 유입량에 맞춰 비례 조정한다.
이 수집은 저장된 평가 스냅샷 대신 새 교통을 실행하므로
최종 평가 성공률을 재현하는 명령은 아니다. 구형 `BCPolicy` 체크포인트는 지원하지 않는다.
