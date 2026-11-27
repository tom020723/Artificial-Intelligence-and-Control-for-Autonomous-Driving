# 제출용 주행 데이터 규격

`collect_pomdp_data.py`에 최종 BC 또는 PPO를 지정해 수집하는 데이터다.
[실행 방법](REPRODUCE.md#제출용-주행-데이터-수집)을 참고한다.
모델의 관측 구성과 학습·보상 설명은 [모델 보고서](report/report_ko.pdf)에 있다.

## 파일 구성

출력 위치는 기본 `data/`이며 `--out-dir`, 이름은 `--name`으로 지정한다.

| 파일 | 내용 |
|---|---|
| `<이름>.jsonl.gz` | 관측·행동·보상·환경 상태·사건을 기록한 압축 JSONL |
| `<이름>.npz` | 고정 크기 학습 배열과 감지 차량 좌표 |
| `<이름>.metadata.json` | 실행 인자, 적용 설정, Git 커밋·미커밋 diff |
| `<이름>.sumo/` | 사용한 도로·신호·경로 파일 |
| `<이름>.csv` | `--csv`를 지정한 경우 생성. 중첩 값은 JSON 문자열 |

JSONL의 float는 `--ndigits`(기본 4)로 반올림한다. `action_raw`와 보상은 별도로
반올림하지 않으며, NPZ의 관측·행동·보상은 float32로 저장한다.

## JSONL transition

| 필드 | 형식 | 의미 |
|---|---|---|
| `episode`, `t` | 정수 | 파일 안 에피소드 번호, 에피소드 안 step 번호. 각각 0부터 시작 |
| `observation` | 길이 52 배열 | 최종 모델의 행동 전 관측 |
| `action_raw` | 길이 2 배열 | 정책이 출력한 가감속·차로 변경 입력. −1~1로 제한 |
| `action.accel_raw`, `action.lane_change_raw` | 실수 | 환경에서 처리한 raw 행동 |
| `action.lane_change` | −1, 0, 1 | 오른쪽 변경, 유지, 왼쪽 변경 명령 |
| `action.lane_change_applied` | Boolean | 차로 명령을 환경에 적용했는지 여부 |
| `reward` | 실수 | `env.step()`이 반환한 값. 모델의 저장된 보상 설정을 사용 |
| `terminated`, `truncated` | Boolean | 환경 사건에 의한 종료, 최대 step 도달 여부 |
| `state` | 객체 | 행동 전 simulator-side 상태. 정책 입력에는 쓰지 않음 |
| `detected_vehicles` | 객체 배열 | 행동 전 관측 범위 안 차량. `id`, `x`, `y`, `rel_pos`, `speed`, `lane_offset` |
| `event` | 객체 | 행동 후 충돌·도착·신호 위반 등의 판정 |
| `next_observation`, `next_state` | 배열, 객체 | 행동 후 관측·상태. 에피소드 마지막 행에만 직접 저장 |
| `episode_info` | 객체 | 에피소드 첫 행에만 저장하는 경로·seed·시작 정보 |
| `new_routes` | 객체 | `{route_id: 경유 edge 목록}`. 해당 경로가 에피소드에서 처음 확인된 행에 저장 |

중간 행의 `next_observation`, `next_state`는 다음 행의 `observation`, `state`다.
`collect_pomdp_data.iter_transitions()`로 읽으면 생략된 값을 복원한다.
CSV에는 모든 행의 `next_*`를 복원하며, null은 빈 칸으로 기록한다.

## 환경 상태와 신호

자차가 존재할 때 `state`에는 `ego_alive`, `ego_x`, `ego_y`, `ego_speed`, `ego_lane`,
`ego_lane_id`, `ego_road_id`, `ego_lane_pos`, `vehicles`, `road_context`가 있다.
자차가 없거나 상태 조회에 실패하면 `{"ego_alive": false}`만 기록한다.
`vehicles`의 각 항목은 `id`, `x`, `y`, `speed`, `lane`, `lane_id`, `road_id`, `lane_pos`, `route_id`다.
차로 번호 0은 가장 오른쪽이며, 왼쪽으로 갈수록 증가한다.

| `road_context` 필드 | 의미 |
|---|---|
| `observed_signals` | 자차가 관측하는 신호의 `tls_id`, `link_index`, `state`, `distance`(m). 최종 모델은 100m 안의 신호를 최대 하나 관측. 없음은 `[]`, 조회 실패는 null |
| `all_signals` | 중앙 교차로 진입로 `[A1, A2, A3, A4, A5]` 순서의 요약 신호. 조회 실패는 null |
| `signal_remaining_s` | 중앙 교차로 현재 현시가 바뀔 때까지 남은 초. 조회 실패는 null |

`observed_signals[].state`는 적색 0, 황색 0.5, 비보호 녹색 0.75, 보호 녹색 1,
그 밖의 신호 문자는 −1이다. `distance`는 자차 앞부분에서 정지선까지의 경로상 거리다.
`all_signals`는 각 진입로의 가장 허용적인 색(적 0, 황 0.5, 녹 1)을 기록하므로,
같은 진입로의 모든 연결이 그 색이라는 뜻은 아니다.
실제 연결별 신호 프로그램은 `.sumo/highway.net.xml`에서 확인한다.

## 사건과 종료

| `event` 필드 | 의미 |
|---|---|
| `collided` | SUMO가 자차 충돌을 보고했는지 여부 |
| `arrived` | 도착 여부. 충돌한 행에서는 false |
| `red_light_violation` | 정지선 통과 당시 신호가 적색인지 여부 |
| `wrong_lane` | 경로로 이어지지 않는 차로 끝 정차, 또는 충돌·도착 없이 자차 소실 |
| `lane_before`, `lane_after` | 행동 전·후 차로 번호. 자차가 없으면 `lane_after`는 null |
| `collision_partner_id` | 충돌 상대 차량 ID |
| `collision_partner_type` | 상대 차종: Krauss 0, IDM 1, EIDM 2, ACC 3, 기타 −1 |
| `collision_partner_speed` | 충돌 상세에 기록된 상대 속도(m/s) |
| `ego_is_collider` | SUMO 충돌 상세에서 자차가 collider인지 여부. 법적 과실이나 충돌 방향의 판정은 아님 |

충돌하지 않았거나 충돌 상세를 찾지 못하면 충돌 상대 네 필드는 모두 null이다.
차로 번호는 도로·교차로 내부 차로 전환 때도 달라질 수 있어 실제 차로 변경 횟수와 같지 않다.
`lane_change_applied`도 명령 적용 여부이며, 물리적으로 차로를 옮겼다는 별도 측정값은 아니다.
모델 수집은 규칙 기반 ego 제한을 해제하지만, 없는 차로처럼 실행 불가능한 명령은 적용되지 않는다.

최종 모델 설정은 0.5초/step, 최대 1,000 step이다.
충돌·도착·신호 위반·차로 실패·자차 소실은 `terminated`를,
최대 step 도달은 `truncated`를 설정한다. 마지막 step에 종료 사건이 발생하면 둘 다 true일 수 있다.
사건 필드는 환경의 원본 Boolean이며, 항상 하나만 true라고 가정하지 않는다.

## 에피소드 메타데이터

| `episode_info` 필드 | 의미 |
|---|---|
| `route_id` | 자차의 실제 경로 ID |
| `seed` | `env.reset()`에 전달한 `--seed + episode` |
| `sumo_seed` | SUMO 실행에 사용한 seed(0~99,999) |
| `sim_time_start_s` | 예열 후 첫 행을 기록한 SUMO 시각 |
| `signal_offset_s` | 신호 주기의 무작위 시작 시점(초). 사용하지 않으면 null |

`new_routes`는 조회에 성공한 경로만 기록하고, 실패하면 뒤의 행에서 다시 조회한다.
에피소드마다 이미 기록한 경로 집합을 초기화한다.
여러 회차로 수집할 때 `--name`을 다르게 지정한다. `episode`는 파일마다 다시 0부터 시작한다.

## NPZ 배열

`observation`, `action_raw`, `lane_change`, `reward`, `next_observation`, `terminated`,
`truncated`, `episode`, `t`를 행 순서로 저장한다.
`detected_xy`는 모든 행의 감지 차량 `(x, y)`를 이어 붙인 `(M, 2)` 배열이고,
`detected_offsets`로 각 행의 시작·끝 위치를 찾는다. pickle 없이 읽을 수 있다.
전체 환경 상태와 충돌 상대 정보는 JSONL에서 읽는다.

수집기는 `analysis.failure_reason`, 구간 유형, 예정 회전 방향 등의 후처리 칼럼을 생성하지 않는다.
