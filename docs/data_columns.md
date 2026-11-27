# 데이터컬럼 정리

| 팀별 추가 항목 | 형태 | 의미 | 한마디 |
|---|---|---|---|
| `event.red_light_violation` | Boolean | 적신호에 정지선을 통과했는지 여부 (3.2) | 빨간불에 지나갔는지 |
| `event.wrong_lane` | Boolean | 경로 밖 차로로 가서 종료됐는지 여부. 원인 불명 소실 포함 (3.2) | 잘못된 차로로 가서 끝났는지 |
| `episode_info` | Dictionary | 자차 경로 ID와 재현용 시드·시작 시각·신호 시작 시점. 각 에피소드 첫 행(t=0)에만 기록 (3.1) | 이번 주행의 경로와 시드 |
| `new_routes` | Dictionary | `{경로 ID: 경유 도로 목록}`. 그 경로가 처음 나온 행에만 기록 (3.1, 3.7) | 처음 나온 경로가 지나는 도로 |
| `state.road_context.observed_signals[]` | List of Dictionary | 기존 `state` 안에 추가. 지금 보이는 자차 적용 신호의 제어기·연결 번호·신호 색 숫자 코드·정지선까지 거리 (3.3) | 앞 신호 색과 정지선까지 거리 |
| `state.vehicles[].route_id` | String | 기존 `state.vehicles` 각 차량에 추가. 그 차량이 타고 있는 경로 ID (3.7) | 각 차가 어느 경로로 가는지 |
| `analysis` | Dictionary (후처리) | 수집한 컬럼으로 후처리한 정보: 실패 원인(3.2), 현재 구간·예정 진행(3.4), 합류·분기까지 거리(3.6), 상호작용 후보(3.7) | 저장된 값으로 나중에 계산한 것 |

## 기존 필수 항목의 변경 사항

교수님 기본 코드(환경 `b4fd1d7`, 수집 `da542dc`) 대비 **이름은 같지만 내용이 바뀐 항목**이다. 위 표의 추가 항목과 함께 이 데이터의 모든 변경점이다.

| 기존 항목 | 기존 | 변경 | 한마디 |
|---|---|---|---|
| `observation`, `next_observation` | Vector (19-dim, 교수님 표). 교수님 9/4 코드부터 31-dim | Vector (51-dim). 구성은 아래 표 | 정책이 보는 숫자 19(31)개 → 51개 |
| `state` | `ego_*` 8개 + `vehicles[]` | 기존 키 유지 + `road_context.observed_signals[]`(3.3), `vehicles[].route_id`(3.7) 추가 | 기존 그대로에 두 가지 추가 |
| `reward` | 충돌·도착 보상, 속도 보상, 근접·막힘·차로 변경 감점 | 팀 보상 함수. 적신호 위반·차로 실패 감점, 적신호 정차·청신호 통과 관련 항 등 추가. 실제 계수는 `metadata.json`의 `effective_config.REWARD` | 보상 계산식이 다름 |
| `terminated` | 충돌, 도착, 자차 소실 | 기존 + **적신호 위반**, **경로 밖 차로 끝에 정지** 시 종료 | 끝나는 조건 2개 추가 |
| `truncated` | 1000 step(500초) | 변경 없음 | 그대로 |
| `episode`, `t`, `action_raw`, `action.*`, `detected_vehicles`, `event.collided`, `event.arrived`, `event.lane_before`, `event.lane_after` | — | 변경 없음 | 그대로 |

**CSV 변환 규칙** (기본 코드는 CSV를 만들지 않으며, 제출용으로 `--csv`에서 생성)
- 컬럼 순서: 교수님 필수 19개 → `event.red_light_violation`, `event.wrong_lane` → `episode_info`, `new_routes`.
- `next_observation`, `next_state`: JSONL에서는 에피소드 마지막 행에만 있지만 CSV에서는 모든 행에 채운다.
- list·dict 값(`observation`, `state` 등)은 JSON 문자열 한 칸으로 저장한다.
- 값이 없으면 빈 칸이다. 예: 도착으로 자차가 사라진 마지막 행의 `event.lane_after`, t>0 행의 `episode_info`.

### observation 51차원 구성 (Phase 1 모델 설정 기준)

| 인덱스 | 개수 | 내용 | 교수님 코드 (19 / 31차원) |
|---|---|---|---|
| 0 | 1 | 자차 속도 / 최대속도 | 있음 |
| 1–20 | 20 | 차로 5개(좌2·좌1·현재·우1·우2)마다 [앞차 거리, 앞차 상대속도, 뒤차 거리, 뒤차 상대속도] | 차로 3개 12칸 / 차로 5개 20칸 (같음) |
| 21–29 | 9 | 차로 9개(좌4 ~ 우4)가 경로로 이어지는지 (3.5) | 차로 3개 / 5개 |
| 30–34 | 5 | 차로 5개의 전방 차량 밀도 | 차로 3개 / 5개 (같음) |
| 35–39 | 5 | 신호: [없음/기타, 녹색(G·g), 황색, 적색] one-hot 4개 + 정지선 거리/100m | 없음 |
| 40–48 | 9 | 차로 9개가 앞으로 3개 도로까지 차로 변경 없이 이어지는지 (3.5) | 없음 |
| 49 | 1 | 경로상 다음 진출 제한 지점까지 거리/300m | 없음 |
| 50 | 1 | 그 지점 전까지 필요한 차로 변경 여유 | 없음 |

모든 값은 −1~1로 잘린다. 자차가 사라진 행(충돌·도착 직후)은 51개 모두 0이다. 구성은 모델 폴더의 관측 설정을 따르며, 다른 모델로 수집하면 `metadata.json`의 `effective_config.OBSERVATION`을 확인한다. observation의 신호 one-hot은 `G`와 `g`를 구분하지 않으므로, 양보형 녹색 여부는 `state.road_context.observed_signals[].state`로 본다.

### 3.1 Route — 첫 transition에 함께 저장

필요한 이유: 자차의 실제 경로를 확인하고 위치·상황별 분석의 기준으로 사용한다.

| 항목 | 형태 | 의미 | 한마디 |
|---|---|---|---|
| `episode_info.route_id` | String | 자차 Route ID. 경유 edge 목록은 같은 행의 `new_routes`에 있음 | 이 주행의 경로 이름 |
| `episode_info.seed` | Integer | 해당 에피소드의 `env.reset(seed=...)` 입력값 | 경로를 뽑은 주사위 번호 |
| `episode_info.sumo_seed` | Integer | 실제 SUMO 실행에 사용한 시드 | 시뮬레이터 안쪽 주사위 번호 |
| `episode_info.sim_time_start_s` | Float | 첫 observation/state를 수집한 실제 시뮬레이션 시각 | 기록을 시작한 시각 |
| `episode_info.signal_offset_s` | Float 또는 null | `--randomize-signals` 사용 시 에피소드 시작 시점의 신호 주기 내 위치(s). 미사용이면 null | 신호가 주기의 몇 초 지점에서 시작했는지 |

`episode_info`는 각 에피소드의 **첫 transition(t=0)에만** 들어간다. `next_state`를 마지막 step에만 넣는 것과 같은 방식이며, 에피소드 번호는 기존 `episode` 컬럼을 그대로 쓴다. 실제 환경은 SUMO 시드를 별도로 뽑고 준비 운행·자차 삽입 대기가 있으므로 재현용 시드와 시작 시각도 함께 남긴다. 실제 시각은 `sim_time_start_s + t × step_length_s`로 복원한다. SUMO 시드는 `_start_sumo()`가 `self.sumo_seed`에 보관한 값이며 실제 `--seed` 인자와 같다.

**수집 파일 구성**

| 파일 | 의미 |
|---|---|
| `<수집명>.jsonl.gz` | 메인. 기본 transition과 추가 state·event, `episode_info`, `new_routes` |
| `<수집명>.npz` | 메인. 학습용 고정 크기 배열 (교수님 기본 항목) |
| `<수집명>.csv` | 제출용. `.jsonl.gz`를 `next_*` 복원 후 평평하게 변환 |
| `<수집명>.metadata.json` | 실제 적용 설정(`effective_config`), 코드 스냅샷(커밋 해시 등), `step_length_s`, `lookahead_edges` |

출발·목적 도로와 경유 목록은 Route 정의에서 구한다. `origin_arm`, `destination_arm`, `route_edges`, 매 step의 자차 `road_context.route_id`는 추가하지 않는다. Route ID 이름을 해석할 수 있어도 실제 경로 정의를 기준으로 한다.

`route_index`는 기본 추가에서 제외한다. 현재 경로처럼 일반 edge가 반복되지 않으면 기존 road ID와 경로 목록으로 계산할 수 있다. 내부 edge는 Route 목록에 직접 없으므로 지도 연결과 전후 transition으로 위치를 복원한다. 반복 경로나 재경로 탐색을 도입하면 명시적 index 저장을 재검토한다.

### 3.2 Failure reason — 사건은 수집, 대표 코드는 후처리

필요한 이유: 원본 사건을 보존하면서 집계 규칙을 수집 후에도 검토할 수 있게 한다.

| 항목 | 형태 | 의미 | 한마디 |
|---|---|---|---|
| `event.collided` | Boolean | 기존 충돌 플래그 유지 | 부딪혔는지 |
| `event.arrived` | Boolean | 기존 도착 플래그 유지 | 목적지에 도착했는지 |
| `event.red_light_violation` | Boolean | 환경 `info`의 적신호 위반 판정 원본 | 빨간불에 지나갔는지 |
| `event.wrong_lane` | Boolean | 환경 `info`의 차로 실패 판정 원본. 원인 불명 소실도 포함될 수 있음 | 잘못된 차로로 가서 끝났는지 |

`failure_reason`, `timeout`, `unexplained_disappearance`는 새 원시 컬럼으로 저장하지 않는다. 위 사건, 기존 `terminated` / `truncated`, `next_state.ego_alive`로 요약한다. 상태 조회 실패도 `ego_alive=false`로 표현될 수 있으므로 소실 자체를 확정적인 차로 오진입 증거로 쓰지 않는다.

#### 현재 환경의 판정 한계

- 적신호 위반은 행동 전 `signals_before`가 있고 행동 후 자차가 살아 있을 때, 이동거리가 정지선까지 거리보다 커졌으며 해당 신호가 `r`인 경우에 판정한다.
- 같은 step에 적신호를 통과한 뒤 충돌 제거·도착·기타 소실로 자차가 없어지면 위반을 검사하지 못해 false일 수 있다. false를 독립적으로 검증된 준법 라벨로 해석하지 않는다.
- 황색 통과는 현재 구현에서 적신호 위반이 아니다. 신호 조회가 꺼져 있거나 대상이 없으면 판정 범위도 제한된다.
- 현재 `wrong_lane`은 생존 상태에서 경로 단절 차로 끝에 정지한 경우와, 충돌·도착 없이 자차가 사라진 경우를 함께 포함한다. 후자의 실제 원인은 확정하지 않는다.

#### 후처리 코드와 우선순위

**필요 컬럼:** `event.collided`, `event.arrived`, `event.red_light_violation`, `event.wrong_lane`, `terminated`, `truncated`, `next_state.ego_alive`

| `analysis.failure_reason` | 의미 |
|---|---|
| `0` | 실패 없음. 진행 중·정상 도착 |
| `1` | 충돌 |
| `2` | 환경이 감지한 적신호 위반 |
| `3` | 생존 상태에서 판정된 차로 실패 |
| `4` | 시간 제한에 따른 미완주 종료 |
| `5` | 기타·원인 불명 종료 또는 소실 |

대표 원인의 우선순위는 환경의 보상 분기에 맞춰 **신호 위반(2), 충돌(1), 정상 도착(0), 원인 불명 소실(5), 생존 상태의 차로 실패(3), 시간 초과(4)** 순서로 적용한다. 개별 원본 Boolean은 그대로 보존한다. 동시 사건이 있는 정상 도착은 앞선 실패 판정이 우선한다.

시간 초과는 현재 환경에서 `truncated=true`이고 `terminated=false`, 미도착·다른 사건 없음·행동 후 생존인 경우로 요약한다. 마지막 허용 step의 정상 도착은 코드 0이다. 다른 truncation 원인을 도입하면 이 규칙을 함께 변경한다. 예상하지 못한 종료는 코드 5로 둔다.

환경 주석은 소실 원인으로 잘못된 차로 진입을 설명하지만, 그것만으로 개별 소실을 확정하지 않는다. 코드 5에 실제 차로 오진입이 포함될 수 있어 코드 3만으로 전체 차로 실패율을 단정하지 않는다.

### 3.3 Signal state — 신호와 정지선 거리

필요한 이유: 정책 판단 시점의 실제 신호와 양보형 녹색 여부를 확인하고, 속도와 함께 제동 시점·정지 위치를 분석한다.

| 항목 | 형태 | 의미 | 한마디 |
|---|---|---|---|
| `state.road_context.observed_signals[]` | List of Dictionary 또는 null | 시점 t에서 관측 범위 안의 자차 적용 신호 목록 | 지금 보이는 신호 목록 |
| `observed_signals[].tls_id` | String | 신호 제어기 ID | 어느 신호 제어기인지 (노형오거리는 `nC` 하나) |
| `observed_signals[].link_index` | Integer | 제어 연결 인덱스. 신호 문자열에서 이 연결의 위치 | 어느 진입 차로→진출 도로 연결의 신호인지 |
| `observed_signals[].state` | Float | 신호 색 숫자 코드. 적(`r`) 0, 황(`y`) 0.5, 비보호 녹(`g`) 0.75, 보호 녹(`G`) 1, 그 밖 −1 | 신호 색 |
| `observed_signals[].distance` | Float | 해당 제어 정지선까지의 경로상 거리(m). 자차 앞부분 기준, 선 위는 0 | 정지선까지 몇 m 남았는지 |

`r`=적색, `y`=황색, `G`=우선권 녹색, `g`=양보형 녹색이다. 그 외 `u`, `o`, `O`, `s` 등 SUMO 문자도 원문으로 저장하며 다른 색으로 임의 통합하지 않는다.

신호 조회는 기존 `_upcoming_signals()`·`_observed_signals()`를 그대로 쓰며, `get_privileged_state()`가 state를 만들 때 함께 호출한다. 따라서 `state`에는 행동 전 시점 t의 신호가, `next_state`에는 t+1의 신호가 자연스럽게 들어간다.

`link_index`가 필요한 이유: 노형오거리 전체를 제어기 `nC` 하나가 제어하고 그 아래 연결이 30개 있어, `tls_id`만으로는 어느 신호인지 알 수 없다. `(tls_id, link_index)`로 `.net.xml`에서 진입 차로와 정지선 위치를 찾고, 같은 녹색이 직진(`G`)인지 유턴(`g`)인지 연결별로 해석한다. 현재 지도에서는 이 키가 진입 차로 하나에 대응함을 확인했다. 지도를 바꾸면 이 유일성을 다시 확인한다.

별도의 `stop_line_id`, `distance_m`, `stop_line` 객체는 만들지 않는다. 행동 후 신호 목록에서 대상이 사라졌다는 사실만으로 위반을 판단하지 않으며, 위반 판정은 3.2를 따른다.

범위는 모델의 관측 설정을 따른다(Phase 1 모델: 1개·100m). 실제 값은 `metadata.json`의 `signal_count`·`signal_visibility`에 기록된다. 조회 성공 시 범위 내 대상이 없으면 `[]`, 조회 실패는 null이다. 빈 목록을 도로 전체의 무신호로 해석하지 않는다.

`all_signals`, 매 step의 `phase_name`, `program_id`, 전체 신호 문자열은 추가하지 않는다. 현재 고정식 프로그램은 실제 시뮬레이션 시각·프로그램·offset으로 복원한다. 신호 현시를 별도로 무작위 초기화하는 실행에서는 초기 phase와 잔여시간 또는 동등한 시간 offset도 에피소드별로 남겨야 한다. 프로그램 변경·수동 제어를 도입하면 변경 이력 저장을 추가한다.

### 3.4 Decision zone — 후처리

필요한 이유: 상황별 성능을 분류한다. 수작업 구간 매핑이 원시 수집 시작을 막지 않도록 분리한다.

| 후처리 항목 | 형태 | 의미 | 한마디 |
|---|---|---|---|
| `analysis.zone_type` | Integer | 지도·차로·위치와 필요한 Route 조건으로 구한 위치 유형 | 지금 어떤 구간에 있는지 |
| `analysis.planned_movement` | Integer | Route와 현재 진행 위치로 구한 예정 진행 | 다음에 직진·좌회전·우회전·유턴 중 무엇을 할지 |

**필요 컬럼:** `state.ego_lane_id`, `state.ego_road_id`, `state.ego_lane_pos`, `new_routes`(자차 경유 도로) + 구간표(지도 `.sumo/`에서 한 번 작성)

| `zone_type` | 의미 |
|---|---|
| `-1` | 매핑 미완료·위치 분류 불가 |
| `0` | 일반 구간 |
| `1` | 교차로 접근 구간 |
| `2` | 지정된 중앙 교차로·유턴 진행 구간 |
| `3` | 합류 구간 |
| `4` | 분기 구간 |

내부 edge를 모두 코드 2로 분류하지 않는다. 실제 지도와 진행 연결별 구간을 정의하고, 겹치는 위치는 해당 Route의 역할에 따라 분류한다. 접근 범위는 분석 설정으로 고정하고 버전을 남긴다. 신호 거리와 접근 구간은 중복되는 부분이 있지만, 관측 범위 제한이나 무신호 구간이 있어 항상 같은 값은 아니다.

**필요 컬럼:** `episode_info.route_id`, `new_routes`(자차 경유 도로), `state.ego_road_id` + 방향표(진입 도로×진출 도로 → 직진·좌·우·유턴, 한 번 작성)

| `planned_movement` | 의미 |
|---|---|
| `-1` | 경로 정보 누락·매핑 미완료 등으로 판별 불가 |
| `0` | 남은 진행 없음(목적지 구간) |
| `1` | 직진 |
| `2` | 좌회전 |
| `3` | 우회전 |
| `4` | 유턴 |

현재 진행 중인 대상 연결이 있으면 그 유형을 사용하고, 진입 전이면 다음 대상 연결을 사용한다. 연결 진출부를 통과하면 다음 대상으로 갱신한다. 경로와 매핑이 완전하면 -1은 정상적인 주행 분류가 아니라 데이터·분석 문제를 나타낸다. `zone_id`, `movement_id`도 원시 수집 컬럼으로 추가하지 않는다.

### 3.5 Lane connectivity — 기존 observation 재사용

필요한 이유: 정책이 실제로 본 차로 연결 정보를 분석한다. 별도의 더 엄격한 정의를 원시 수집에 추가하지 않는다.

| 기존 정보 | 형태 | 의미 | 한마디 |
|---|---|---|---|
| `observation`의 기본 연결성 구간 | Vector | `_lane_connectivity()` 결과. 다음 edge 연결과 차로 잔여거리·관측 범위를 반영 | 이 차로가 다음 도로로 이어지는지 |
| `observation`의 추가 연결성 구간 | Vector | 활성화 시 `_lookahead_connectivity_observation()` 결과. 설정된 여러 route edge까지 추가 차로 변경 없는 연결 여부 | 이 차로로 몇 도로 앞까지 이어지는지 |
| 실제 적용 설정·코드 스냅샷 | Dictionary 및 코드 파일 | 관측 구간의 인덱스·차로 오프셋·정규화·결측 표현 복원. 레이아웃 요약은 선택 | 이 데이터를 만든 설정과 코드 |
| `metadata.lookahead_edges` | Integer | 실제 추가 연결성 평가에 사용한 route edge 수 | 몇 도로 앞까지 확인했는지 |

`_lane_follows_route()`는 다음 edge 연결을 확인한다. `_lane_connectivity()`는 남은 차로 거리까지 반영하는 동적 값으로, 단순 Boolean이 아니다. 추가 연결성은 여러 edge를 보며 기본 연결성과 정의가 다르다. 주석의 ‘두 개’에 고정하지 않고 실제 `lookahead_edges` 설정을 기록한다.

`connected_without_lane_change`, 매 step의 `valid_lane_ids`는 추가하지 않는다. 정적 차로 연결표가 필요하면 분석 결과를 메타데이터/분석 보조 파일에 `(episode, route_id, edge)`별로 한 번 저장한다. 일반 edge가 반복되면 route index도 키에 포함한다. 내부 차로 처리와 조회 실패 시 중립값도 기존 관측 구현대로 설명한다.

연결성 값만으로 안전한 차로 변경 가능 여부나 주행 실패를 확정하지 않는다. 기존 observation 저장으로 정책 입력이 보존되므로 새 컬럼에 동일 결과를 복사하지 않는다.

### 3.6 Distance to merge/diverge — 후처리 (수집 안 함)

**필요 컬럼:** `state.ego_lane_id`, `state.ego_road_id`, `state.ego_lane_pos`, `new_routes`(자차 경유 도로) + 3.4의 구간표(합류·분기 기준점 위치)

`next_merge_diverge`, 지점 ID·유형·거리와 모호한 ‘두 기능 모두’ 코드는 수집하지 않는다. 분석에서 필요할 때 지도·Route·차로 위치로 계산하며, 기준점·분류 규칙은 후처리 설정에 정의한다. 합류·분기 구간별 분류는 3.4를 사용한다.

### 3.7 Interaction vehicles — 상대 Route만 추가, 후보 분류는 후처리

필요한 이유: 비보호 유턴 등에서 상대 차량의 진행 경로를 나중에 확인할 수 있도록 한다.

| 항목 | 형태 | 의미 | 한마디 |
|---|---|---|---|
| `state.vehicles[].route_id` | String 또는 null | 해당 차량의 실제 `getRouteID` 결과. 조회 실패는 null | 각 차가 타고 있는 경로 이름 |
| `new_routes` | Dictionary | `{route_id: edge 목록}`. 그 에피소드에서 해당 Route ID를 처음 본 transition에만 기록 | 처음 나온 경로가 지나는 도로 목록 |

기존 전체 차량 위치·속도·차로와 `detected_vehicles`를 유지하고, 차량별 Route ID를 추가하며 경로 정의는 같은 `episode`의 `new_routes`에서 찾는다. 배경 차량 Route ID는 에피소드마다 재사용될 수 있으므로 다른 에피소드의 정의를 쓰지 않는다. 자차도 기존 차량 목록에 포함되므로 같은 규칙을 적용한다. 자차 에피소드 Route와의 일관성을 확인하되 별도의 자차 매 step Route 컬럼을 만들지 않는다.

`neighbor_vehicles`, 관계 코드, 상대속도·간격, 충돌 구역 거리·점유 라벨은 수집하지 않는다. 앞뒤 차량 거리·상대속도는 이미 observation에 있고, 차량 ID 추적이 실제로 필요해지면 그 목적에 맞춰 확장한다. 개별 Route 조회가 실패해도 다른 차량 목록 전체를 null로 만들지 않는다.

**후처리 필요 컬럼:** `state.vehicles[]`(`id`, `route_id`, `lane_id`, `lane_pos`, `speed`), `new_routes`(각 차량 경유 도로), `episode_info.route_id` + 지도 `.sumo/`의 교차로 연결

후처리는 Route·차로·지도 연결을 이용해 합류·교차·유턴의 잠재 충돌 후보를 구한다. Route는 일반 edge 순서이므로 미래 진출 차로까지 항상 하나로 정해지지는 않는다. 가능한 연결이 여러 개면 잠재 후보로 보존하고, 확정 의도나 실제 충돌로 단정하지 않는다.

차체를 고려한 구역 점유까지 분석하려면 차량 길이·폭 또는 실제 차량 타입별 고정 치수를 함께 보관해야 한다. 이 치수와 구역 계산 규칙은 분석을 시작하기 전에 확보한다. Route ID만으로 정확한 충돌 시각이나 안전한 진입 간격까지 알 수 있는 것은 아니다.

**기본 설정은 배경 차량 OFF라 상대 차량 표본이 생기지 않는다.** 실제 유턴 분석용 수집에는 배경 교통을 활성화하고 실행 조건을 기록한다. 상대 Route는 simulator-side 정보이며 정책이 상대 의도를 알고 있다는 뜻이 아니다. 반대편 차량에 대응하는 정책 관측·학습·평가는 별도로 필요하다.
