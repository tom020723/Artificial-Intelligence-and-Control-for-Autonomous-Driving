# Phase 1 신호 대응 검증 자료

최종 판정: **동일 설정 seed 3개 재현성 미달**. seed 0/1은 통과했지만 seed 2는
Ring 두 경로에서 100건 실패하고 직접 경로에서 먼 정차를 보였다.

- [실험 보고서](../../docs/phase1_experiment_report.md)
- [상세 결과 및 명령](../../docs/phase1_results.md)
- [진행 기록](../../docs/phase1_progress.md)
- [파일 크기 및 SHA256 manifest](manifest.json)

## 모델 구분

| run 접미사 (`parallel_1_20261113_` 이후) | 역할 |
|---|---|
| `120614_157702` | 최초 기준, 과거 27 ID 평가 |
| `122617_060732` | clearance reward 0.20 후보 |
| `124659_843781` | clearance reward 0.02, seed 0 |
| `130626_876501` | clearance reward 0.02, seed 1 |
| `132650_331439` | clearance reward 0.02, seed 2 |
| `134618_825826` | clearance reward 0.05 후보 |
| `140556_054790` | reward 0.02 + scale 2.0, penalty 0 후보 |
| `142521_414407` | 최종 후보 seed 0, 8,192,000 step |
| `144440_871432` | 최종 후보 seed 1, 8,192,000 step |
| `150435_751583` | 참고용 seed 2, 32,735,232 step, 다른 PPO 설정 |
| `184459_502151` | 최종 후보 동일 설정 seed 2, 8,192,000 step, 실패 |

각 폴더에는 모델, 저장 설정, 학습 로그, 남아 있는 Python 소스 snapshot,
원시 평가 CSV/JSON, 수정 후 trace를 포함한다. 생성 SUMO XML·TensorBoard·그림과
총보상이 비어 있던 이전 `traces/`는 제외했다. 원본 `results/` 자료는 삭제하지 않았다.
JSON에 포함된 원래 절대 경로는 실행 당시 출처이며 이 게시본의 실행 경로가 아니다.
manifest는 이 README와 manifest 자체를 제외한 복사된 자료의 해시다.

## 게시본 평가

프로젝트 루트에서 SUMO/libsumo와 프로젝트 의존성이 있는 Python 환경으로 실행한다.

```text
python scripts/evaluate_quick.py experiments/phase1_20261113/parallel_1_20261113_184459_502151
python scripts/evaluate_quick.py experiments/phase1_20261113/parallel_1_20261113_184459_502151 --offset-set separate
```

평가는 run 폴더에 결과를 쓰므로 보존된 자료를 바꾸지 않으려면 run을 별도 작업 폴더에
복사한 뒤 그 경로로 실행한다. 재평가로 자료를 덮어쓰면 manifest와 달라진다.
평가기 구현은 현재 checkout에서 import하므로 게시 브랜치의 코드를 사용한다.
저장된 source snapshot이 자동으로 실행되는 것은 아니다. 최초 기준 모델의 27 ID
결과와 현재 23개 고유 경로 결과는 분모와 경로 샘플링 변경을 고려해 해석한다.

검토 때 기록된 실행 환경은 Python 3.11.9, PyTorch 2.11.0+cu128, CUDA,
libsumo 1.28.0이다. 장비·버전이 달라지면 학습 결과의 동일성을 보장하지 않는다.
게시 준비에서는 기존 자료를 검사하고 관련 테스트를 실행했으며 장기 학습은 반복하지 않았다.
