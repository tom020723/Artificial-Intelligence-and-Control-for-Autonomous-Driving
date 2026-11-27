# 코드 제출 패키지

전체 프로젝트 소스와 문서, 최종 BC/PPO actor 가중치 및 실행에 필요한 설정을 포함합니다.

- 최종 PPO: `results/final/PPO/eval_runs/iter060/model.pt`
- 최종 BC: `results/final/BC/training/model.pt`
- 중간 체크포인트, critic, 학습·평가 로그, 평가 결과, 캐시, 영상은 포함하지 않습니다.
- `suite_*`의 XML·JSON·SQLite는 동일 조건 평가를 위한 입력이므로 보존했습니다.
- 학습 데이터 원본과 Python 가상환경은 포함하지 않습니다.

SUMO를 설치하고 실행 파일을 PATH에 등록한 뒤 Python 의존성을 설치하세요.
원본 requirements에 빠져 있던 headless 수집 의존성 `libsumo`를 이 패키지의 requirements에 추가했습니다.

```bash
python -m pip install -r requirements.txt
python scripts/evaluate_diverse_start_lanes.py evaluate results/final/suite_diverse_test --model results/final/PPO/eval_runs/iter060/model.pt --output results/submission_eval --smoke 1 --workers 1
python collect_pomdp_data.py --model results/final/PPO/eval_runs/iter060/model.pt --traffic-vph 4700 --episodes 1 --randomize-departures --name submission_ppo
```

코드는 GPU 없이도 위 모델을 CPU로 실행할 수 있습니다. 상세 실행 방법은 `docs/REPRODUCE.md`를 참고하세요.
기존 학습 곡선·보고서 그림을 다시 생성하는 명령은 생략한 로그·평가 결과가 필요하므로,
새 학습·평가 결과를 생성한 뒤 실행해야 합니다. 코드의 재학습 기능은 그대로 포함되어 있습니다.
GitHub의 원본 파일은 변경하지 않았습니다. 파일별 SHA-256과 원본 커밋은 `SUBMISSION_MANIFEST.json`에 있습니다.
