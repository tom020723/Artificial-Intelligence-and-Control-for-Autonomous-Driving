# 노형오거리 데이터 컬럼

이 문서는 `collect_pomdp_data.py`에 [최종 BC 모델](../results/final/BC/training/model.pt) 또는 [최종 PPO 모델(iter060)](../results/final/PPO/eval_runs/iter060/model.pt)을 `--model`로 지정하고, `--traffic-vph 4700`으로 수집하는 제출 데이터를 기준으로 한다. 두 모델 모두 `observation`과 `next_observation`이 52차원이다. [모델별 수집 명령](REPRODUCE.md#제출용-주행-데이터-수집-pr-7-통합)

## 1. 기존 컬럼 — 차원 변경

| 기존 항목 | 기존 | 변경 | 한마디 |
|---|---|---|---|
| `observation`, `next_observation` | Vector (19-dim) | Vector (52-dim) | 정책이 보는 숫자 19개 → 52개 |

| 인덱스 | 개수 | 내용 | 기존 대비 |
|---|---|---|---|
| 0 | 1 | 자차 속도 / 도로 설정의 기준 제한속도 | 동일 |
| 1–20 | 20 | 차로 5개(좌2·좌1·현재·우1·우2)별 [앞차 거리, 앞차 상대속도, 뒤차 거리, 뒤차 상대속도] | 차로 3개 → 5개 |
| 21–25 | 5 | 차로 5개의 물리적 연결성 | 차로 3개 → 5개 |
| 26–30 | 5 | 차로 5개의 전방 차량 밀도 | 차로 3개 → 5개 |
| 31–34 | 4 | 전방 신호 [적, 황, 녹, 정지선 거리/200m] | 신규 |
| 35–38 | 4 | 다음 도로 분기 [유효 여부, 거리/650m, sin(각도), cos(각도)] | 신규 |
| 39–42 | 4 | 두 번째 도로 분기 (35–38과 같은 형식) | 신규 |
| 43–48 | 6 | 링 합류 [링 여부, 합류점 거리, 상대 차량 2대의 속도·합류점 거리] | 신규 |
| 49–50 | 2 | 분기 1·2 진출 도로 입구의 좌우 거리 | 신규 |
| 51 | 1 | 출발 직후 진행도 | 신규 |

**`observation` 값의 의미**

속도 정규화에는 도로 설정의 고정 기준값(`ROAD["speed_limit"]`, 13.89 m/s)을 쓴다.

| 인덱스 | 값 | 의미 |
|---|---|---|
| 0 | 0 ~ 1 | 0 = 정지, 1 = 기준 제한속도 이상. 비율은 최대 1로 제한 |
| 1–20 앞차 거리 (현재 차로) | 0 ~ 1 | 범퍼 사이 거리/50m. 1 = 50m 안에 앞차 없음 |
| 1–20 뒤차 거리 (현재 차로) | −1 ~ 0 | −(범퍼 사이 거리/50m). −1 = 50m 안에 뒤차 없음 |
| 1–20 앞차·뒤차 거리 (옆 차로 4개) | −1 ~ 1 | 부호 있는 범퍼 간격. 차가 자차 옆에 겹쳐 있으면 앞차 거리는 음수, 뒤차 거리는 양수. 없으면 앞차 1, 뒤차 −1 |
| 1–20 상대속도 | −1 ~ 1 | (상대 차량 속도 − 자차 속도)/제한속도. 앞차가 없으면 1, 뒤차가 없으면 −1 |
| 21–25 | −1 | 그 차로가 없음 |
| | 0 ~ 1 미만 | 50m 안에서 끝나는 차로. 남은 거리/50m |
| | 1 | 다음 도로로 이어짐 |
| 26–30 | 0 ~ 1 | 전방 50m 차량 수 / 수용 대수. 0 = 비어 있음, 1 = 꽉 참 또는 차로 없음 |
| 31–33 | 0 또는 1 | 적·황·녹 중 현재 색만 1. 셋 다 0이면 100m 안에 신호 없음 |
| 34 | 0 이상 | 정지선까지 거리/200m. 신호가 없으면 0 |
| 35, 39 | 0 또는 1 | 1 = 안내할 분기가 있음, 0 = 없음(나머지 세 값도 0) |
| 36, 40 | 0 이상 | 분기 지점까지 거리/650m |
| 37–38, 41–42 | −1 ~ 1 | 진출 도로 방향. sin > 0 = 왼쪽, sin < 0 = 오른쪽, cos = 1이면 직진 |
| 43 | 0 또는 1 | 1 = 링(원형 도로) 합류 구간 |
| 44 | −1 또는 0 이상 | 자차의 합류점까지 거리/50m. −1 = 링 구간 아님 |
| 45, 47 | 0 이상 | 상대 차량 1·2의 속도/제한속도. 차량이 없으면 0 |
| 46, 48 | −1 또는 0 이상 | 상대 차량 1·2의 합류점까지 거리/50m. −1 = 차량 없음 |
| 49, 50 | 실수 (범위 제한 없음) | 분기 1·2 진출 도로 입구가 자차 진행 방향 기준 좌우로 떨어진 거리/50m. 양수 = 입구가 오른쪽, 음수 = 왼쪽. 해당 분기가 없으면 0 |
| 51 | 0 ~ 1 | 첫 도로에서는 차로 위치/20m (최대 1), 그 뒤로는 1 |
| 전체 | 모두 0 | 자차가 사라진 행 (충돌·도착 직후) |

차량 거리(1–20)는 지금 도로뿐 아니라 이어진 다음·이전 도로와 교차로 내부 차로의 차량까지 포함해 범퍼 간격으로 잰다.

## 2. 기존 컬럼 — 내부 필드 추가

| 추가 항목 | 형태 | 의미 | 한마디 |
|---|---|---|---|
| `event.red_light_violation` | Boolean | 적신호에 정지선을 통과했는지 여부 | 빨간불에 지나갔는지 |
| `event.wrong_lane` | Boolean | 경로 밖 차로로 가서 종료됐는지 여부 (원인 불명 소실 포함) | 잘못된 차로로 가서 끝났는지 |
| `event.collision_partner_id` | String 또는 null | 충돌 상대 차량 ID | 누구와 부딪혔는지 |
| `event.collision_partner_type` | Integer 또는 null | 충돌 상대 차량 종류 | 어떤 종류의 차와 부딪혔는지 |
| `event.collision_partner_speed` | Float 또는 null | 충돌 순간 상대 차량 속도 (m/s) | 상대가 얼마나 빨랐는지 |
| `event.ego_is_collider` | Boolean 또는 null | 자차가 뒤에서 들이받은 쪽인지 | 자차가 박았는지, 받혔는지 |
| `state.road_context.all_signals` | Vector (5-dim) 또는 null | 진입로 A1~A5별 신호 | 다섯 방향 신호 색 |
| `state.road_context.signal_remaining_s` | Float 또는 null | 현재 현시가 바뀔 때까지 남은 시간 (s) | 몇 초 뒤 바뀌는지 |
| `state.vehicles[].route_id` | String 또는 null | 각 차량이 타고 있는 경로 ID | 각 차가 어느 경로로 가는지 |
| `state.road_context.observed_signals` | List of Dictionary 또는 null | 자차가 지금 보는(100m 이내) 신호의 `tls_id`·`link_index`·`state`(신호 색 숫자 코드)·`distance`(정지선까지 m) | 앞 신호 색과 정지선 거리 |

**`event.red_light_violation`**

| 값 | 의미 |
|---|---|
| true | 이번 step에 적신호(`r`)인 정지선을 넘음. 에피소드 종료 |
| false | 위반이 감지되지 않음. 황색 통과는 false |

**`event.wrong_lane`**

| 값 | 의미 |
|---|---|
| true | 경로로 이어지지 않는 차로 끝에 멈춤, 또는 충돌·도착 없이 자차가 사라짐. 에피소드 종료 |
| false | 해당 없음 |

**충돌 상대 정보 (`event.collision_*`, `event.ego_is_collider`)**

| 상태 | 값 |
|---|---|
| `event.collided = false`인 행 | 네 항목 모두 null |
| `event.collided = true`인 행 | SUMO에서 자차의 충돌 상세를 찾으면 네 항목 모두 기록. 조회가 실패하거나 해당 상세가 없으면 모두 null |

| `collision_partner_id` | 의미 |
|---|---|
| `f_<출발>_<도착>.<번호>` | 교차로를 지나는 배경 차량. 예: `f_A2_A3.51` = A2에서 A3로 가는 흐름의 51번째 차량 (0번부터 셈) |
| `lf_<원형 도로>.<번호>` | 원형 도로(우회전 전용 길)를 지나는 배경 차량. 예: `lf_RING_A1_A3.7` |

SUMO가 붙인 차량 ID를 그대로 쓴다. `state.vehicles[].id`와 같은 값이므로, 이 ID로 충돌 직전 행(`state`)에서 상대 차량의 위치·차로·속도·경로를 찾을 수 있다.

| `collision_partner_type` | 의미 |
|---|---|
| -1 | 기타·조회 실패 |
| 0 | Krauss |
| 1 | IDM |
| 2 | EIDM |
| 3 | ACC |

| `ego_is_collider` | 의미 |
|---|---|
| true | 자차가 앞차를 들이받음 |
| false | 뒤차가 자차를 들이받음 |

**`state.road_context.observed_signals`**

자차 경로에 적용되는 신호 중 정지선 거리가 100m 이내인 가장 가까운 신호를 최대 하나 담는다. 보이는 신호가 없으면 `[]`, 신호 조회에 실패하면 null이다. 자차가 사라진 뒤의 `next_state`는 `{"ego_alive": false}`이며 `road_context` 필드는 생략된다.

| 항목 | 형태 | 의미 |
|---|---|---|
| `tls_id` | String | 신호 제어기 ID (예: 중앙 교차로 `nC`) |
| `link_index` | Integer | 그 제어기에서 자차가 지나갈 연결 번호 |
| `state` | Float | 신호 색 숫자 코드 (아래 표) |
| `distance` | Float | 자차 앞부분부터 정지선까지 경로상 거리 (m). 정지선 위면 0 |

| `state` 값 | SUMO 신호 문자 | 의미 |
|---|---|---|
| 0 | `r` | 적색. 정지 |
| 0.5 | `y` | 황색 |
| 0.75 | `g` | 비보호 녹색. 진행 가능하지만 우선권이 있는 차량에 양보 (U턴, 합류하는 연결 등) |
| 1 | `G` | 보호 녹색. 우선권을 갖고 진행 |
| −1 | 그 밖의 문자 | 현재 신호 프로그램에서는 나오지 않음 |

`all_signals`와 같은 기준(적 0, 황 0.5, 녹 1)에, 비보호 녹색만 0.75로 구분한다.

**`state.road_context.all_signals`**

중앙 교차로의 진입로 순서는 `[A1, A2, A3, A4, A5]`이며, 시계 방향 표기로는 `[8시, 2시, 5시, 10시, 3시]`다. 한 칸에는 해당 진입로에서 출발하는 중앙 교차로 제어 연결의 신호 중 가장 허용적인 색을 기록한다. 녹색이 하나라도 있으면 1, 녹색 없이 황색이 있으면 0.5, 둘 다 없으면 0이다. 보호 녹색과 비보호 녹색은 여기서 모두 1이다. 자차가 지나갈 연결의 신호는 `observed_signals`에서 확인한다.

| 각 칸의 값 | 의미 |
|---|---|
| 0 | 적색 |
| 0.5 | 황색 |
| 1 | 녹색 |

| 예시 | 의미 |
|---|---|
| `[1, 1, 1, 0, 0]` | 8시·2시·5시 진입로에 녹색인 연결이 각각 하나 이상 있음. 같은 진입로의 다른 연결은 적색일 수 있음 |
| `[1, 0, 0, 0, 1]` | 8시·3시 진입로에 녹색인 연결이 각각 하나 이상 있음 |
| `[0, 0, 0, 0, 0]` | 다섯 진입로의 요약이 모두 적색 |
| null | 신호 조회 실패 |

**`state.road_context.signal_remaining_s`**

| 값 | 의미 |
|---|---|
| 0 ~ 20 | 중앙 교차로의 현재 현시가 끝날 때까지 남은 초. 그룹 신호의 녹색 현시는 17~20초, 황색 현시는 3초, 전환 현시는 2초 |
| null | 신호 조회 실패 |

제출 데이터에는 `--traffic-vph 4700`으로 켠 120초 그룹 신호 주기를 사용한다. 연속된 두 녹색 현시에 모두 포함된 연결은 그 사이 황색·전환 현시에도 녹색을 유지할 수 있다.

**`state.vehicles[].route_id`**

| 값 | 의미 |
|---|---|
| `ego_d_<출발>_<도착>` | 자차 경로. 교차로를 바로 지남 (예: `ego_d_A3_A1`) |
| `ego_r_<원형도로>_<출발>_<도착>` | 자차 경로. 원형 도로를 한 번 이상 거침 |
| `ego_d_A1_A1`처럼 출발 = 도착 | 자차 U턴 경로 |
| 그 밖의 문자열 | 배경 차량 경로. 경유 도로는 `new_routes`에서 확인 |
| null | 조회 실패 |

## 3. 기존 컬럼 — 의미·형태 변경

| 기존 항목 | 기존 | 변경 | 한마디 |
|---|---|---|---|
| `reward` | 충돌·도착 보상, 속도 보상, 근접·막힘·차로 변경 감점 | 최종 BC는 속도·신호·주행 상황을 반영한 보상, 최종 PPO는 사건 중심 보상 | 모델별 계산식은 아래 표 참조 |
| `terminated` | 충돌, 도착, 자차 소실 | 기존 + 적신호 위반, 경로 밖 차로 끝에 정지 | 끝나는 조건 2개 추가 |
| `event.collided` | SUMO 차량 타입별 기본 충돌 기준 | 범퍼 사이 0.2m 이하를 충돌로 판정 | 충돌 기준 통일 |
| `event.lane_before`, `event.lane_after` | Integer / String | Integer | 정수로 확정 |

**`reward`**

수집기는 모델에 저장된 설정으로 환경을 구성하고, `env.step()`이 반환한 보상을 기록한다. 아래 PPO 열은 기본 스텝 보상 `−0.0001`에 더하는 사건별 값이다. 예를 들어 차로 변경 감점이 없는 PPO 도착 행은 `0.9999`다.

| 상황 | 최종 BC에서 기록하는 값 | 최종 PPO 사건 보상 |
|---|---|---|
| 적신호 위반 | −5 | −2 |
| 충돌 | −5 | −2 |
| 목적지 도착 | +2 | +1 |
| 경로 밖 차로 끝에 정지 | 해당 스텝의 일반 보상 − 2 | −1 |
| 충돌·도착 없이 자차 소실 | −2 | −1 |
| 시간 초과만 발생 | 해당 스텝의 일반 보상 | −1 |
| 진행 중 | 속도 보상 `0.1 × 자차 속도 / 기준 제한속도`에 차량 간격·막힘·차로 변경·신호 관련 보상과 감점을 반영 | 적신호 정차 보상을 조건에 따라 더함 |

최종 PPO는 실행된 차로 변경 명령마다 `0.01`을 추가로 뺀다. 보상 설정은 `metadata.json`의 `effective_config.REWARD`에 기록되며, PPO 기본 스텝 보상과 사건 계수는 [`env/bc_ppo_reward.py`](../env/bc_ppo_reward.py)에 정의되어 있다.

**`terminated`**

| 값 | 의미 |
|---|---|
| true | 충돌, 도착, 적신호 위반, 경로 밖 차로 끝에 정지, 자차 소실 중 하나로 종료 |
| false | 진행 중 |

**종료 상황별 값** — 에피소드 마지막 행 기준. 빈 칸은 false

| 종료 상황 | `terminated` | `truncated` | `event.collided` | `event.arrived` | `event.red_light_violation` | `event.wrong_lane` | `analysis.failure_reason` |
|---|---|---|---|---|---|---|---|
| 충돌 | true | | true | | | | 1 |
| 목적지 도착 | true | | | true | | | 0 |
| 적신호 위반 | true | | | | true | | 2 |
| 경로 밖 차로 끝에 정지 | true | | | | | true | 3 |
| 충돌·도착 없이 자차 소실 | true | | | | | true | 5 |
| 1000 step 도달 | | true | | | | | 4 |
| (진행 중인 행) | | | | | | | 0 |

현재 환경에서 종료 사건은 한 행에 하나만 true가 된다. 적신호 위반은 행동 후 자차가 남아 있을 때만 검사하므로, 충돌·도착으로 자차가 사라진 행에서는 false다. 1000번째 step에 도착하면 `terminated`와 `truncated`가 둘 다 true다.

**`event.collided`**

| 값 | 의미 |
|---|---|
| true | 이번 step에 다른 차량과 범퍼 사이 0.2m 이하로 붙음. 자차와 상대 차량 모두 제거되고 종료 |
| false | 충돌 없음 |

**`event.lane_before`, `event.lane_after`**

자차가 몇 번 차로에 있었는지를 행동 적용 전(`lane_before`)과 후(`lane_after`)로 기록한다. 차로 변경 명령이 실행되면 두 값이 달라진다. 예: `lane_before = 1`, `lane_after = 2`면 왼쪽으로 한 차로 옮김. 차로 변경 명령(`action.lane_change`)과 비교해 명령대로 움직였는지 확인할 때 쓴다. 단, 차로 수가 다른 다음 도로나 교차로 내부 차로로 넘어갈 때는 명령 없이도 번호가 바뀔 수 있다.

| 값 | 의미 |
|---|---|
| 0 | 가장 오른쪽 차로 |
| 1, 2, … | 왼쪽으로 갈수록 1씩 증가 |
| 빈 칸 | 자차가 사라져 차로가 없음 (`lane_after`만 해당) |

## 4. 신규 컬럼

| 신규 항목 | 형태 | 의미 | 한마디 |
|---|---|---|---|
| `episode_info` | Dictionary | 에피소드 정보. 각 에피소드 첫 행(t=0)에만 기록 | 이번 주행의 경로와 시드 |
| `episode_info.route_id` | String | 자차 경로 ID | 이 주행의 경로 이름 |
| `episode_info.seed` | Integer | `env.reset(seed=...)` 입력값 | 경로를 뽑은 시드 |
| `episode_info.sumo_seed` | Integer | SUMO 실행 시드 | 시뮬레이터 시드 |
| `episode_info.sim_time_start_s` | Float | 에피소드 첫 행의 SUMO 시각 (s) | 언제부터 기록했는지 |
| `episode_info.signal_offset_s` | Float 또는 null | 신호 주기 안의 시작 시점 (s). `--randomize-signals`를 안 쓰면 null | 신호 몇 초 지점에서 시작했는지 |
| `new_routes` | Dictionary | `{경로 ID: 경유 도로 목록}`. 그 경로가 처음 나온 행에만 기록 | 경로가 지나는 도로 |

**`episode_info`**

에피소드 하나에 한 번만 기록하는 값이다. 한 에피소드 안의 모든 행·모든 차량에 공통이므로 첫 행(t=0)에만 넣고, t > 0인 행은 빈 칸이다.

| 항목 | 값 | 의미 |
|---|---|---|
| `route_id` | 예: `ego_d_A3_A1` | 이번 에피소드의 자차 경로. 2절 `route_id` 규칙과 같음 |
| `seed` | 0 이상 정수 | 수집기가 에피소드마다 다르게 주는 번호(`--seed` + 에피소드 번호). 자차 경로를 뽑는 데 쓰임 |
| `sumo_seed` | 0 ~ 99999 | SUMO가 배경 차량의 출발 차로, 운전 편차 같은 무작위 요소에 쓰는 시드. 에피소드마다 새로 뽑음 |
| `sim_time_start_s` | 0 이상 | 웜업 후 자차가 들어와 첫 행을 기록한 시각 |
| `signal_offset_s` | 0 ~ 신호 주기 길이, 또는 null | 학습과 같게 신호 주기의 임의 시점에서 시작했을 때 그 시점 |

`seed`와 `sumo_seed`는 차량별 값이 아니라 그 에피소드의 시뮬레이션 전체에 대한 값이다. `sumo_seed`는 수집기의 `--seed`와 에피소드 번호로만 정해지므로, 같은 `--seed`·수집 옵션·정책으로 다시 실행하면 워커 수와 상관없이 같은 번호의 에피소드가 재현된다.

**`new_routes`**

| 값 | 의미 |
|---|---|
| `{"ego_d_A3_A1": ["A3_in1", "A3_in", …]}` | 이 행에서 처음 나온 경로와 그 경로가 지나는 도로 순서 |
| 빈 칸 | 이 행에서 새로 나온 경로가 없음 |

## 5. 후처리 컬럼 (수집 데이터로 계산)

| 후처리 항목 | 형태 | 의미 | 필요 컬럼 |
|---|---|---|---|
| `analysis.failure_reason` | Integer | 실패 원인 | `event.collided`, `event.arrived`, `event.red_light_violation`, `event.wrong_lane`, `terminated`, `truncated`, `next_state.ego_alive` |
| `analysis.zone_type` | Integer | 현재 위치 유형 | `state.ego_lane_id`, `state.ego_road_id`, `state.ego_lane_pos`, `new_routes` + 구간표 |
| `analysis.planned_movement` | Integer | 예정 진행 방향 | `episode_info.route_id`, `new_routes`, `state.ego_road_id` + 방향표 |
| `analysis.merge_diverge_distance` | Float | 다음 합류·분기 지점까지 거리 (m) | `state.ego_lane_id`, `state.ego_road_id`, `state.ego_lane_pos`, `new_routes` + 구간표 |

| `failure_reason` | 의미 |
|---|---|
| 0 | 실패 없음 (진행 중·도착) |
| 1 | 충돌 |
| 2 | 적신호 위반 |
| 3 | 차로 오진입 |
| 4 | 시간 초과 |
| 5 | 기타·원인 불명 |

| `zone_type` | 의미 |
|---|---|
| -1 | 분류 불가 |
| 0 | 일반 구간 |
| 1 | 교차로 접근 |
| 2 | 교차로 내부 |
| 3 | 합류 구간 |
| 4 | 분기 구간 |

| `planned_movement` | 의미 |
|---|---|
| -1 | 판별 불가 |
| 0 | 남은 진행 없음 |
| 1 | 직진 |
| 2 | 좌회전 |
| 3 | 우회전 |
| 4 | 유턴 |

| `merge_diverge_distance` | 의미 |
|---|---|
| 0 이상 | 경로상 다음 합류·분기 지점까지 남은 거리 (m) |
| 빈 칸 | 남은 합류·분기 지점이 없거나 계산 불가 |

## 6. 변경 없는 기존 컬럼

| 항목 | 형태 |
|---|---|
| `episode` | Integer |
| `t` | Integer |
| `action_raw` | Vector (2-dim) |
| `action.accel_raw` | Float |
| `action.lane_change_raw` | Float |
| `action.lane_change` | Integer |
| `action.lane_change_applied` | Boolean |
| `truncated` | Boolean |
| `detected_vehicles` | List |
| `event.arrived` | Boolean |

| 항목 | 값 | 의미 |
|---|---|---|
| `episode` | 0 이상 정수 | 에피소드 번호 |
| `t` | 0 ~ 999 | 에피소드 안의 step 번호. 1 step = 0.5초 |
| `action_raw` | [가감속, 차로 변경], 각 −1 ~ 1 | 정책이 출력한 원래 값 |
| `action.accel_raw` | −1 ~ 1 | +1 = 최대 가속, −1 = 최대 감속, 0 = 속도 유지 |
| `action.lane_change_raw` | −1 ~ 1 | 차로 변경 원래 값 |
| `action.lane_change` | +1 | 왼쪽 변경 명령 (`lane_change_raw` ≥ 0.5) |
| | 0 | 유지 (−0.5 < `lane_change_raw` < 0.5) |
| | −1 | 오른쪽 변경 명령 (`lane_change_raw` ≤ −0.5) |
| `action.lane_change_applied` | true | 명령이 SUMO에 적용됨. 유지 명령(0)도 현재 차로 고정으로 적용되므로 true |
| | false | 변경 명령이 차로 없음·출발 직후 제한·안전 조건 등으로 실행되지 않음, 또는 교차로 내부라 명령을 적용하지 않음 |
| `truncated` | true | 1000 step(500초)에 도달해 끊김 |
| | false | 시간 제한 전 |
| `detected_vehicles` | `[{id, x, y, rel_pos, speed, lane_offset}, …]` | 관측 범위(좌우 2차로, 앞뒤 50m) 안의 배경 차량. `rel_pos` > 0 = 앞, `lane_offset` > 0 = 왼쪽 |
| | `[]` | 감지된 차량 없음, 자차가 교차로 내부, 또는 자차 없음 |
| `event.arrived` | true | 목적지 도착. 종료 |
| | false | 미도착 |

## 7. 파일 구성

제출은 과제 기본 형식인 `.jsonl.gz` + `.npz`로 한다. CSV는 선택이며, 필요하면 일부 에피소드만 샘플로 만든다.

| 파일 | 제출 | 내용 |
|---|---|---|
| `<수집명>.jsonl.gz` | ✅ | 전체 transition (1~4번 항목 포함). `next_observation`·`next_state`는 에피소드 마지막 행에만 저장하고, 나머지 행은 다음 행의 `observation`·`state`가 곧 `next_*`다 |
| `<수집명>.npz` | ✅ | 학습용 배열 (교수님 기본 항목) |
| `<수집명>.metadata.json` | ✅ | 실행 인자, 설정, 코드 버전. `--traffic-vph`로 신호 주기를 켜면 `effective_config.NOHYEONG.all_signals_green`은 `false`다. 차로 연결별 신호 프로그램은 `.sumo/highway.net.xml`에 저장된다 |
| `<수집명>.sumo/` | ✅ | 사용한 지도 파일 |
| `<수집명>.csv` | 선택 (샘플) | `--csv`로 생성. `next_*` 모든 행에 채움, list·dict는 JSON 문자열, Boolean은 `true`/`false`, null은 빈 칸. 행마다 전체 차량 목록이 압축 없이 들어가 10⁷개 기준 약 570GB라 전체 제출에는 쓰지 않는다 |

10⁷ transition 기준 예상 용량: `.jsonl.gz` 약 30GB, `.npz` 수 GB. 에피소드 약 5만 개가 필요하므로 여러 번 나눠 수집한다. 회차마다 `--name`을 바꾸고, `--seed`는 한 회차 에피소드 수 이상 간격으로 준다 (예: 5,000개씩이면 `--seed 0`, `5000`, `10000`, …). 자차 경로 시드가 `--seed + 에피소드 번호`라 간격이 좁으면 회차끼리 같은 경로 시드가 겹친다. `episode` 번호는 회차(파일)마다 0부터 다시 시작한다.

설계 근거와 세부 정의는 [data_columns_rationale.md](data_columns_rationale.md)에 있다.
