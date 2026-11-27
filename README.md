# Artificial Intelligence and Control for Autonomous Driving

> **자율주행 인공지능 및 제어 — Course Project Materials**

수업 프로젝트에 필요한 자료를 순서대로 정리합니다.

## 🚗 팀 최종 모델 (IDM 전문가 BC → Anchored PPO)

- 무작위 출발 차선 테스트 7,050건(교차로 안 충돌 포함): BC 90.79%, **BC → PPO 97.21%**
- 알고리즘 전체(관측, 행동, 데이터, 학습, 보상, 평가): [docs/ALGORITHM.md](./docs/ALGORITHM.md)
- 단계별 재현 설명과 최종 결과물 위치: [docs/REPRODUCE.md](./docs/REPRODUCE.md)


### 저장소

```bash
git clone https://github.com/tom020723/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git
cd Artificial-Intelligence-and-Control-for-Autonomous-Driving      # 기본 브랜치 main
```

GitHub: <https://github.com/tom020723/Artificial-Intelligence-and-Control-for-Autonomous-Driving>

### 실행 환경

최종 모델을 만든 환경이다. 가중치로 평가·주행하거나 처음부터 학습하는 데는 Python 환경 하나면 된다. 두 환경(A, B)은 `results/final`과 바이트 단위로 같은 모델을 다시 만들 때만 필요하다(아래 "처음부터 학습 재현하기").

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 22.04 |
| SUMO | 1.28.0 (`/usr/share/sumo`, `traci`·`sumolib`·`libsumo`는 SUMO의 `tools/`에서 사용) |
| Python 환경 A (수집, BC 학습, BC 평가) | Python 3.11, torch 2.7.0+cu128, numpy 1.26.0, gymnasium 1.2.0, matplotlib 3.10 |
| Python 환경 B (PPO 학습, PPO 평가) | Python 3.10, torch 2.14.1+cu130, numpy 2.2.6, gymnasium 1.4.0, matplotlib 3.10 |
| 하드웨어 | AMD Ryzen 9 7900X (12코어), NVIDIA RTX 5070 |

나머지 패키지는 [requirements.txt](./requirements.txt)에 있다. 모든 명령은 저장소 최상위 폴더에서 실행하고, 먼저 경로를 잡는다.

```bash
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH=$PWD:$SUMO_HOME/tools
```

### 올려 둔 가중치로 실행하기

가중치는 저장소에 들어 있다(따로 받을 것 없음).

| 모델 | 경로 |
|---|---|
| 최종 BC → PPO | `results/final/PPO/eval_runs/iter060/model.pt` |
| BC | `results/final/BC/training/model.pt` |

**테스트 평가** (무작위 출발 차선 7,050건, 교차로 안 충돌 포함. 24코어에서 약 1.5분)

```bash
python scripts/evaluate_diverse_start_lanes.py evaluate results/final/suite_diverse_test \
  --model results/final/PPO/eval_runs/iter060/model.pt --output results/my_eval_ppo
# --smoke 1 을 붙이면 4건만 빠르게 돌려 동작만 확인한다
```

결과는 `results/my_eval_ppo/`에 `evaluation_summary.json`(성공률과 실패 종류), `route_evaluation.csv`(사례별 결과), `summary.png`(결과 그림)로 저장된다. 같은 모델이면 사례별 결과가 `results/final/PPO/eval_runs/iter060/evaluation_diverse_start_lanes/`와 똑같이 나온다. BC를 평가하려면 `--model`만 BC 경로로 바꾼다.

**SUMO 화면에서 주행 보기**

```bash
python collect_pomdp_data.py --model results/final/PPO/eval_runs/iter060/model.pt \
  --traffic-vph 4700 --episodes 1 --randomize-departures --gui
```

`--gui`를 빼면 화면 없이 돌고, 주행 기록이 `data/`에 저장된다([docs/data_columns_submission.md](./docs/data_columns_submission.md)).

### 처음부터 학습 재현하기

평가 세트 생성 → IDM 전문가 데이터 수집 → BC 학습 → BC 평가 → PPO 학습 → 검증으로 checkpoint 선택 → 테스트까지 한 번에 실행한다.

**Python 환경 하나로 학습하기** (가장 간단함)

```bash
PY=<python> bash scripts/reproduce_pipeline.sh            # 전체 (약 3.5시간)
SMOKE=1 PY=<python> bash scripts/reproduce_pipeline.sh    # 모든 단계를 최소 규모로 (약 3분)
```

`PPO_PY`를 주지 않으면 모든 단계가 `PY` 하나로 돈다. 전 과정이 끝까지 돌고 비슷한 성능의 모델이 나오지만, `results/final`과 똑같은 모델은 아니다. torch/CUDA 빌드가 다르면 GPU 연산의 마지막 자릿수가 달라지고, 그 차이가 학습 중에 커지기 때문이다. 예를 들어 이전 모델을 환경 B 하나로 다시 만들었을 때 수집 데이터는 값이 모두 같았지만, BC와 PPO는 다른 모델이 됐다(학습 seed를 바꾼 것과 같은 정도의 차이).

**`results/final`과 똑같은 모델을 다시 만들기**

최종 모델을 만들 때처럼 0–3단계(평가 세트, 수집, BC)는 환경 A, 4–5단계(PPO, 평가)는 환경 B로 돌린다.

```bash
PY=<환경 A의 python> PPO_PY=<환경 B의 python> bash scripts/reproduce_pipeline.sh
```

위 표와 같은 환경이면 데이터셋, BC 모델, PPO checkpoint, 평가 결과가 모두 `results/final`과 바이트 단위로 같게 나온다(2026-11-27 확인). GPU 종류나 드라이버가 달라도 PPO가 다른 모델로 끝날 수 있지만, 성능은 비슷한 범위(테스트 약 96–97%)다.

- 결과는 `results/reproduce_<시각>/`에 생긴다. 같은 폴더(`OUT=...`)로 다시 실행하면 끝난 단계는 건너뛴다.
- 여유 디스크 약 40GB가 필요하다(수집 원본 약 30GB, 선별 데이터셋 약 4.4GB).
- 단계별 명령과 옵션은 [docs/REPRODUCE.md](./docs/REPRODUCE.md)에 있다.

---

## 📚 Project Materials

### 개인별 진행

### 0. Project Introduction

- [프로젝트 소개 및 전체 진행 방향](./project_materials/0_Project_Introduction.md)

### 1. SUMO Installation

- [Windows — SUMO 설치 및 환경 설정](./project_materials/1_SUMO_Windows_Setup.md)
- [macOS (Apple Silicon) — SUMO 설치 및 환경 설정](./project_materials/1_SUMO_macOS_AppleSilicon_Setup.md)

### 2. Road Environment Construction

- [도로 및 모델 POMDP 구조 이해](./project_materials/2_Road_Environment_Construction.md)

---

### 팀별 진행 (1-2명으로 구성, 자율 building)

### 3. Model Training — Behavior Cloning

- [모방학습 기반 자율주행 의사결정 모델 학습](./project_materials/3_Model_Training_BC.md)

### 팀별 진행 (자율 building)

### 4. Model Training — Reinforcement Learning

- [강화학습 기반 자율주행 의사결정 모델 학습](./project_materials/4_Model_Training_RL.md)

---
### 팀별 진행 (3-5명으로 구성, 자율 building)

### 5. Project Preparation

- [프로젝트 진행을 위한 코드 소개](./project_materials/5_Project_Preparation.md)

### 이후. 팀별 프로젝트 진행

#### 평가 기준

- [프로젝트 평가 기준 소개](./project_materials/6_Project_Evaluation.md)
- [동료 평가 표](https://docs.google.com/document/d/1czmE5ofrKckau2rMwK4-wXax-f5QrUia/edit?usp=sharing&ouid=107893289049883287038&rtpof=true&sd=true)

#### 프로젝트 컨설팅

- [팀별 컨설팅 진행 소개](./project_materials/7_Project_Consulting.md)
---
> 수업 자료는 `project_materials/` 폴더에서도 확인 가능합니다.
