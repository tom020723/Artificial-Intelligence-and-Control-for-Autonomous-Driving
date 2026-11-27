# 학습·평가·수집 실행

모델 구조, 52차원 관측, 보상과 실험 결과는 [보고서 PDF](report/report_ko.pdf)와
[LaTeX 원본](report/report_ko.tex)에 모아 둔다. 이 문서는 현재 스크립트의 실행 방법을 다룬다.

## 실행 환경

저장소 루트에서 명령을 실행한다. Python 의존성은 `requirements.txt`를 참고하고,
headless 병렬 실행과 데이터 수집에 필요한 `libsumo`도 설치한다.
SUMO의 `sumo`, `sumo-gui`, `netconvert` 실행 파일을 PATH에 등록한다.
현재 재현 스크립트는 Bash와 `seq`를 사용하므로 Windows에서는 Bash 실행 환경이 필요하다.
Python·PyTorch·CUDA·SUMO 버전과 작업자 수가 달라지면 학습 결과도 달라질 수 있다.

## 전체 파이프라인

```bash
bash scripts/reproduce_pipeline.sh
SMOKE=1 bash scripts/reproduce_pipeline.sh
```

`SMOKE=1`은 작은 데이터·환경·반복 수로 전체 실행 경로를 확인한다.

| 환경 변수 | 기본값 | 역할 |
|---|---|---|
| `OUT` | `results/reproduce_<시각>` | 결과 폴더. smoke 실행은 `results/reproduce_smoke_<시각>` |
| `PY` | `python` | 세트 생성, 전문가 수집, BC 학습·평가 |
| `PPO_PY` | `$PY` | PPO 학습·평가 |
| `WORKERS` | `12` | 전문가 수집 프로세스 수 |
| `SUMO_HOME` | `/usr/share/sumo` | SUMO Python tools 경로 |
| `SUMO_CHECK_JUNCTIONS` | `0` | 전문가 수집·PPO 학습의 교차로 내부 충돌 검사. 평가는 기본으로 검사 |

같은 `OUT`으로 다시 실행하면 `.done_<단계>`가 있는 단계는 건너뛴다.
설정이나 코드를 바꿨다면 새 `OUT`으로 실행한다. 실패한 단계의 부분 출력은 자동으로 복구하지 않는다.

| 순서 | 담당 스크립트 | `OUT` 아래 출력 |
|---|---|---|
| 0 | `scripts/evaluate_fixed_routes.py prepare` | `suite_fixed_routes/` |
| 1–2 | `scripts/measure_calibrated_idm_coverage.py` → `scripts/train_bc_shards.py` | `BC/dataset/`, `BC/training/`, `BC/evaluation/` |
| 3 | `scripts/evaluate_diverse_start_lanes.py` | 테스트·검증 세트, BC 테스트 평가 |
| 4 | `python -m scripts.finetune_bc_ppo` | `PPO/`, 10회 반복마다 체크포인트 |
| 5 | `scripts/ppo_checkpoints.py`, `scripts/evaluate_diverse_start_lanes.py` | 검증 평가, `PPO/selection.json`, 선택 모델 테스트 평가 |

도로와 평가 설정은 `env/`, `train.py`, `benchmark/`에서 만든다.
배경 교통량은 `env/background_demand.py`의 `CALIBRATED_VPH`(4,700대/시간),
경로별 비율은 같은 파일의 `FLOW_SHARES`를 사용한다.
신호 계획은 `env/road_config.py`의 120초 묶음 신호다.

## 저장된 최종 모델 평가

```bash
python scripts/evaluate_diverse_start_lanes.py evaluate results/final/suite_diverse_test \
  --model results/final/BC/training/model.pt --output results/check_final_bc
python scripts/evaluate_diverse_start_lanes.py evaluate results/final/suite_diverse_test \
  --model results/final/PPO/eval_runs/iter060/model.pt --output results/check_final_ppo
```

평가는 기본으로 교차로 내부 충돌을 검사한다. 비교 실행에서 끄려면 `--no-junction-check`를 준다.
무작위 출발 차선 평가는 사례별 SUMO를 새로 시작한다.
결과 폴더에는 `evaluation_summary.json`, `route_evaluation.csv`, `summary.png`가 생성된다.
저장된 결과의 그림만 갱신하려면 다음 명령을 쓴다.

```bash
python scripts/evaluate_fixed_routes.py replot results/check_final_ppo
```

`results/final/PPO/selection.json`이 최종 체크포인트 선택 결과다.
새 학습에서는 검증 평가 뒤 `scripts/ppo_checkpoints.py select`가 기록한 모델을 사용한다.
`iter060`이 모든 재학습에서 선택되는 것은 아니다.
학습 데이터 shard는 Git에 포함하지 않으며, 보관된 목록과 체크섬은 `results/final/BC/dataset/`에 있다.

## 제출용 주행 데이터 수집

`collect_background_data.py`는 BC 학습용 IDM 전문가 수집기다.
`collect_pomdp_data.py`는 학습된 모델을 주행시켜 제출용 transition을 기록한다.

```bash
python collect_pomdp_data.py --model results/final/BC/training/model.pt \
  --traffic-vph 4700 --episodes 10 --workers 2 --randomize-departures --name final_bc
python collect_pomdp_data.py --model results/final/PPO/eval_runs/iter060/model.pt \
  --traffic-vph 4700 --episodes 10 --workers 2 --randomize-departures --name final_ppo
```

출력 형식과 필드는 [제출 데이터 규격](data_columns_submission.md)을 참고한다.
최종 BC/PPO를 자동 판별하고, PPO 평가용 폴더에서는 상위 PPO 실행의 설정을 읽는다.
모델 수집은 ego 규칙 제한을 해제하고 교차로 내부 충돌 검사도 기본으로 켠다.
검사를 끄려면 `--no-junction-check`를 준다.

`--traffic-vph`를 주면 신호 무작위 시작을 기본으로 켜고, 예열 기본값은 250초다.
`--warmup-seconds`로 예열 시간을 바꿀 수 있다. 배경차는 경로별 보정 비율로 배분한다.
새 교통을 실행하므로 저장된 평가 스냅샷의 성공률을 재현하는 명령은 아니다.
같은 seed·옵션·결정적 모델의 수집은 워커 수와 무관하게 재현되지만,
`--stochastic`과 random 정책은 워커별 정책 난수도 사용한다.
구형 `BCPolicy` 체크포인트는 지원하지 않는다.

## 보고서와 그림 생성

```bash
python scripts/plot_report_figures.py --out docs/report/figures \
  --method bc=results/final/BC/evaluation_diverse_start_lanes \
  --method bc_ppo=results/final/PPO/eval_runs/iter060/evaluation_diverse_start_lanes \
  --training bc_ppo=results/final/PPO --validation bc_ppo=results/final/PPO
cd docs/report
latexmk -xelatex report_ko.tex
```

보고서 원본과 PDF, 보고서가 참조하는 그림 PDF를 Git에 보관한다.
LaTeX 보조 파일과 그림의 PNG·표 생성물은 `.gitignore`에서 제외한다.
