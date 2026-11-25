# 보관된 파일

최종 과정(`docs/REPRODUCE.md`)에 쓰지 않는 이전 실험 파일이다. 원래 경로를 그대로 유지했고 git 기록도 이어진다(`git log --follow`). 이 폴더의 스크립트는 경로가 바뀌어 그대로는 실행되지 않을 수 있다.

| 파일 | 내용 |
|---|---|
| `scripts/train_bc_ppo.py`, `scripts/plot_ppo_training.py` | 첫 PPO(`results/PPO_20261124_100850`). `scripts/finetune_bc_ppo.py`로 대체 |
| `scripts/run_bc_adaptive_idm.py`, `scripts/run_bc_adaptive_dagger.py` | 적응형 교통량 BC·DAgger 실험 |
| `scripts/run_bc_bumper_full.py`, `scripts/check_bc_shards.py` | BC49 범퍼 거리 실험의 일괄 실행과 데이터 점검 |
| `scripts/evaluate_quick_bc.py`, `scripts/evaluate_quick_phase1.py`, `scripts/evaluate_quick_phase2.py` | BC·Phase 1·2 빠른 평가. 최종 과정이 쓰던 함수는 `utils/bc_policy.py`, `utils/evaluation_summary.py`, `scripts/evaluate_fixed_routes.py`로 옮김 |
| `scripts/run_bc51_fixed_start.py`, `scripts/run_bc51_dagger.py` | BC51 가운데 출발 평가와 DAgger 실험. 평가 컨텍스트와 액터 불러오기는 `utils/bc_policy.py`로 옮김 |
| `train_bc_actor.py` | 한 파일 데이터셋용 BC 학습기. `scripts/train_bc_shards.py`로 대체했고, 쓰던 학습 함수는 그 파일로 옮김 |
| `tests/test_evaluate_quick_bc.py`, `tests/test_clearance_reference_plots.py` | 위 스크립트의 테스트. 옮긴 함수의 테스트는 `tests/test_bc_evaluation.py` |
| `docs/BC.md`, `docs/bc_observation.md` | 49차원·43차원 관측 시절의 BC 설명. 최종은 `docs/final_BC_trianing.md` |
| `docs/exp_train.md`, `docs/parallel_train.md` | Phase 1·2 PPO 학습 기록 |
| `docs/analysis/bc49_*.md` | BC49 실험 분석 |
