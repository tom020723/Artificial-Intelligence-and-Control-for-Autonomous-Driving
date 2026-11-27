# 수집 데이터 컬럼 (`data/nohyeong_phase1.csv`)

| 필수 포함 항목 | 형태 | 의미 | 포함 여부 (o / x) |
|---|---|---|---|
| episode | Integer | Episode 번호 | |
| t | Integer | Episode 내부 timestep | |
| observation | Vector (19-dim) | 시점 t에서 AV Policy가 실제로 입력받는 부분 관측 oₜ | |
| action_raw | Vector (2-dim) | Policy가 출력한 원본 Action [가감속, 차선변경] | |
| action.accel_raw | Float | 환경에 적용된 가감속 Action 값 | |
| action.lane_change_raw | Float | 환경에 적용된 차선변경 raw 값 | |
| action.lane_change | Integer | 실제 명령으로 변환된 차선변경 값 {-1, 0, +1} | |
| action.lane_change_applied | Boolean | 차선변경 명령이 실제 적용되었는지 여부 | |
| reward | Float | 시점 t에서 받은 Reward rₜ | |
| terminated | Boolean | 충돌 / 도착 등 환경 자체의 종료 여부 | |
| truncated | Boolean | 최대 Episode Step 등 시간 제한에 의한 종료 여부 | |
| state | Dictionary | SUMO 내부의 privileged simulator state sₜ | |
| detected_vehicles | List | Observation 시점에 감지된 주변 차량 정보 | |
| event.collided | Boolean | 해당 step에서 충돌 발생 여부 | |
| event.arrived | Boolean | 목적지 도착 여부 | |
| event.lane_before | Integer / String | Action 적용 전 AV 차선 | |
| event.lane_after | Integer / String | Action 적용 후 AV 차선 | |
| next_observation | Vector (19-dim) | 다음 시점의 부분 관측 oₜ₊₁ | |
| next_state | Dictionary | 다음 시점의 privileged simulator state sₜ₊₁ | |
| **팀별 추가 항목** | **형태** | **의미** | |
| event.red_light_violation | Boolean | 적신호에 정지선을 통과했는지 여부 | o |
| event.wrong_lane | Boolean | 경로 밖 차로로 가서 종료됐는지 여부 (원인 불명 소실 포함) | o |
| episode_info | Dictionary | 자차 경로 ID·시드·SUMO 시드·시작 시각·신호 시작 시점. 각 에피소드 첫 행(t=0)에만 기록 | o |
| new_routes | Dictionary | `{경로 ID: 경유 도로 목록}`. 그 경로가 처음 나온 행에만 기록 | o |
| state.road_context.observed_signals | List of Dictionary | `state` 칸 내부. 자차 적용 신호의 ID·연결 번호·신호 색 숫자 코드(적 0, 황 0.5, 비보호 녹 0.75, 보호 녹 1)·정지선까지 거리(m) | o |
| state.vehicles[].route_id | String | `state` 칸 내부. 각 차량이 타고 있는 경로 ID | o |
