# 7시간 BC49 ring 합류 실험

- 기존 43개 관측 유지. 43~48에 `[ring, 자차 합류거리, 차량1 속도, 차량1 합류거리, 차량2 속도, 차량2 합류거리]` 추가.
- 합류 지점: 현재 ring 차선이 연결되는 공통 외부 도로의 시작점. 거리 API는 경로상 거리를 사용한다.
- 다른 도로의 접근 차량은 지도 연결성을 이용해 해당 합류 차선으로 연결되는 접근 차선과 내부 차선에서 찾는다. 이미 합류 지점을 지난 차량 제외. 자차 반경 50m 내에서만 감지하고, 합류 지점까지 남은 거리 순으로 2대를 선택한다.
- 거리 /50m, 속도 /도로 최대속도. clipping 없음. 없는 차량은 `[0,-1]`, ring 밖은 `[0,-1,0,-1,0,-1]`.
- SUMO의 getJunctionFoes, opened, priority, 안전한 출발 여부를 observation에 넣지 않는다.
- 기존 문제 사례를 native seed 66000으로 재현했을 때, 대기 사례는 접근 차량 슬롯이 채워지고 출발 사례는 빈 슬롯으로 나타났다. 원래 43개 항목과 native 행동은 유지됐다.

## 규모 및 자원

기존 8,002,544개 데이터 수집은 약 650초(episode wall 합 613.61초), 학습 약72초. 새 49차원 비례 유입 pilot은 118,378개/약9.35초, 약12,662 samples/s.

수집은 단일 시뮬레이션, 총 요청 유입3000대/시. 앞서 제안된 차선 비례 유입을 기본 가정으로 사용: A1 5/19, A2 4/19, A3 4/19, A4 3/19, A5 3/19. 중앙 및 ring 경유 수요를 모두 포함하고 각 출발 도로 내부의 경로 혼합 비율은 유지한다. 평가는 같은 유입 분포를 사용한다. 따라서 이전 가중치 기반 결과와는 관측·유입 분포 두 조건이 달라진다.

수집 상한128,000,000개 또는3시간. 최소20GB 여유 디스크 확보. 고정 seed99000부터 독립 simulation episode를 수집한다. 긴 정차·episode 반복으로 과도한 JSON 재기록이 발생하지 않게 shard manifest와 요약 진행 상태만 쓴다. 예상 저장공간 약6~8GB. 데이터는 합치지 않고 episode NPZ shard로 유지한다.

학습은 임의 초기화 seed0, hybrid actor256→256/ReLU, 기존 acceleration MSE + sqrt inverse-frequency lane CE, lr0.0003, batch4096 유지. 전체 데이터 GPU 적재 대신 shard 단위 전송, whole-episode train/validation 분리. 최대100epoch 또는 전체 마감에서 평가 여유30분을 제외한 시간까지. validation loss가25epoch 동안 개선되지 않으면 조기 종료할 수 있다. 남은 시간이 있어도 의미 없이 계산을 채우지는 않는다.

검증 loss 최저 및 마지막 checkpoint를 각각 0/1000/2000/3000 veh/h, 23경로×50신호 offset으로 평가한다. 4개 교통량의 평균 완주율, 동률이면 평균 충돌률로 selected_model.pt를 별도 선택한다. 원래 model.pt/last_model.pt는 보존한다. 전체7시간 이내 종료를 위해 여유를 둔 절대 마감을 사용한다. 실행 source는 작업 시작 시 별도로 복사·hash 저장해 중간 작업 파일 변경의 영향을 막는다.

## 확인

- 관련 unit/integration tests23개 통과.
- single-road BC49 dataset validation 통과.
- shard 학습2epoch 및 정규화 export parity 통과.
- ring5경로에 대해 BC49 모델의0/3000 veh/h replica inference 통과.
- 추가 legacy parallel suite6개 중1개는 Phase2 시작 차량 삽입 실패(기본 `_find_departures` 경로). 신규 실험은 `independent_episode_rng=True`, `randomize_departures=True`를 사용하며 해당 실행 경로는 별도 smoke test로 통과했다. legacy 실패를 수정한 것으로 간주하지 않는다.

runtime plan/status/log는 results/bc49_ring_7h_pipeline_*/에서 확인한다. 스케줄러 task-spooler 작업번호는 실행 후 보고한다.
