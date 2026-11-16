"""출발·도착 조합을 한 대씩 주행하는 도로 테스트. 학습 모델은 사용하지 않는다.

    python test_road_routes.py                         # 25개 조합, GUI에서 매번 ▶
    python test_road_routes.py --nogui                  # 자동 검사
    python test_road_routes.py --pair A1 A4             # 특정 조합만 재생
    python test_road_routes.py --pair A1 A4 --depart-lane 4
    python test_road_routes.py --auto-start --delay 20  # GUI 자동 재생

각 조합은 시각 0부터 시작한다. 배경 차량 없이 SUMO 기본 제어로 신호와
경로를 따라가며, 순간이동은 비활성화한다. 실제 도착한 경우만 PASS다.
현재 설정으로 별도 도로를 생성하고 결과를 results/road_test_*/에 저장한다.
"""
import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

# SUMO_HOME 자동 감지 및 SUMO Python 도구 경로 설정.
from env.sumo_env import SumoHighwayEnv  # noqa: F401
from env import road_builder
from env.road_config import ROAD, EGO, TRAFFIC, NOHYEONG
from env.mdp_config import SIMULATION
from sumolib import checkBinary
import traci


BASE = Path(__file__).resolve().parent
LABELS = {"passed": "PASS", "no_route": "경로 없음", "timeout": "시간 초과",
          "collision": "충돌", "teleport": "순간이동", "removed": "차량 소실",
          "error": "실행 오류", "aborted": "중단", "not_run": "미실행"}


def build_test_road(folder):
    """학습 설정과 env/sumo 파일을 보존하고 테스트 도로만 생성한다."""
    config, traffic = copy.deepcopy(NOHYEONG), copy.deepcopy(TRAFFIC)
    config["all_signals_green"] = False
    config["warmup_seconds"] = 0
    for demand in config.get("ring_segments", []) + config.get("loop_routes", []):
        demand["vph"] = 0
    traffic.update(background_enabled=False, vehs_per_hour=0)
    traffic.setdefault("prefill", {})["enabled"] = False
    original = road_builder.NOHYEONG
    try:
        road_builder.NOHYEONG = config
        cfg = road_builder.build(ROAD, EGO, traffic, str(folder / "sumo"))
    finally:
        road_builder.NOHYEONG = original
    (folder / "settings.json").write_text(json.dumps(
        {"road": ROAD, "ego": EGO, "nohyeong": config, "traffic": traffic,
         "controller": "SUMO default (no learned policy)", "signals": "normal cycle"},
        ensure_ascii=False, indent=2) + "\n")
    return cfg


def departure_lane(net, edges):
    """뒤쪽 연결부터 추적하여 필요한 차선 변경 수가 적은 출발 차선을 고른다."""
    costs = None
    connections = net.findall("connection")
    for source, target in reversed(list(zip(edges, edges[1:]))):
        previous = {}
        for link in connections:
            if link.get("from") != source or link.get("to") != target:
                continue
            src, dst = int(link.get("fromLane")), int(link.get("toLane"))
            cost = 0 if costs is None else min(abs(dst - lane) + value
                                             for lane, value in costs.items())
            previous[src] = min(previous.get(src, float("inf")), cost)
        if not previous:
            raise ValueError(f"차선 연결 없음: {source} → {target}")
        costs = previous
    if not costs:
        raise ValueError("두 개 이상의 도로 구간이 필요합니다.")
    return min(costs, key=lambda lane: (costs[lane], lane))


def run_case(cfg, row, args, log_file):
    vehicle = f"test_{row['origin']}_{row['destination']}"
    command = [checkBinary("sumo" if args.nogui else "sumo-gui"), "-c", cfg,
               "--step-length", str(SIMULATION["step_length"]),
               "--seed", str(args.seed), "--no-step-log", "true",
               "--time-to-teleport", "-1", "--collision.action", "remove",
               "--log", str(log_file)]
    if not args.nogui:
        command += ["--delay", str(args.delay)]
        if args.auto_start:
            command += ["--start"]
    connected = False
    seen, tracking = False, False
    last_lane = None
    row.update(elapsed_seconds=0., waiting_seconds=0., lane_trace=[])
    try:
        traci.start(command)
        connected = True
        # speedMode/laneChangeMode를 바꾸지 않고 SUMO 기본 운전을 사용한다.
        traci.vehicle.add(vehicle, row["route_id"], typeID="ego_type",
                          departLane=str(row["depart_lane"]), departSpeed="0")
        while traci.simulation.getTime() < args.max_seconds:
            traci.simulationStep()
            row["elapsed_seconds"] = traci.simulation.getTime()
            if vehicle in traci.simulation.getStartingTeleportIDList():
                row.update(status="teleport", reason="순간이동 발생")
                break
            if vehicle in traci.simulation.getCollidingVehiclesIDList():
                row.update(status="collision", reason="충돌 발생")
                break
            if vehicle in traci.simulation.getArrivedIDList():
                if last_lane is not None and last_lane.rsplit("_", 1)[0] == row["route"][-1]:
                    row.update(status="passed", reason="목적지 도착", arrival_lane=last_lane)
                else:
                    row.update(status="error", reason="예상 목적지 도로에서 도착하지 않음")
                break
            if vehicle not in traci.vehicle.getIDList():
                if seen:
                    row.update(status="removed", reason="도착 없이 차량이 사라짐")
                    break
                continue
            seen = True
            lane = traci.vehicle.getLaneID(vehicle)
            if lane != last_lane:
                row["lane_trace"].append({"time": row["elapsed_seconds"], "lane": lane})
            last_lane = lane
            if traci.vehicle.getSpeed(vehicle) < .1:
                row["waiting_seconds"] += SIMULATION["step_length"]
            if not args.nogui and not tracking:
                view = traci.gui.getIDList()[0]
                traci.gui.trackVehicle(view, vehicle)
                traci.gui.setZoom(view, SIMULATION.get("gui_zoom", 1200))
                tracking = True
        else:
            row.update(status="timeout", reason=f"{args.max_seconds:g}초 이내 미도착")
        if row["status"] != "passed" and last_lane:
            row["reason"] += f" (마지막 차선: {last_lane})"
    except KeyboardInterrupt:
        row.update(status="aborted", reason="사용자가 중단함")
    except (traci.TraCIException, traci.FatalTraCIError, OSError) as exc:
        row.update(status="error", reason=str(exc))
    finally:
        if connected:
            try:
                traci.close()
            except (traci.TraCIException, traci.FatalTraCIError):
                pass
        row["warnings"] = [line for line in log_file.read_text().splitlines()
                           if line.startswith("Warning:")] if log_file.exists() else []
    return row


def save_results(folder, rows):
    (folder / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    fields = ["origin", "destination", "status", "reason", "route_id", "route",
              "route_kind", "depart_lane", "elapsed_seconds", "waiting_seconds", "arrival_lane", "warnings"]
    with (folder / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "route": " ".join(row["route"]),
                             "warnings": " | ".join(row.get("warnings", []))})


def main():
    parser = argparse.ArgumentParser(description="5거리 출발·도착 25개 조합 도로 테스트")
    parser.add_argument("--nogui", action="store_true", help="화면 없이 전체 자동 검사")
    parser.add_argument("--pair", nargs=2, metavar=("FROM", "TO"), help="예: --pair A1 A4")
    parser.add_argument("--depart-lane", type=int, help="출발 차선 고정. 기본은 경로에 맞춰 자동 선택")
    parser.add_argument("--auto-start", action="store_true", help="GUI에서 ▶ 없이 자동 재생")
    parser.add_argument("--delay", type=float, default=SIMULATION.get("gui_delay", 100), help="GUI 스텝당 지연(ms)")
    parser.add_argument("--max-seconds", type=float, default=300, help="조합당 시뮬레이션 제한시간(초)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="새 결과 폴더 경로")
    args = parser.parse_args()
    if ROAD.get("type") != "nohyeong":
        parser.error("이 테스트는 nohyeong 도로용입니다.")
    arms = [arm["id"] for arm in NOHYEONG["arms"]]
    if args.pair and any(arm not in arms for arm in args.pair):
        parser.error("출발·도착은 " + ", ".join(arms) + " 중에서 선택하세요.")
    if not 0 < args.max_seconds < float("inf") or not 0 <= args.delay < float("inf"):
        parser.error("제한시간은 양수, GUI 지연은 0 이상이어야 합니다.")
    if not 0 <= args.seed < 2**31:
        parser.error("시드는 0 이상, 2**31 미만이어야 합니다.")
    if args.depart_lane is not None and args.depart_lane < 0:
        parser.error("출발 차선은 0 이상이어야 합니다.")
    folder = args.output_dir or BASE / "results" / datetime.now(ZoneInfo("Asia/Seoul")).strftime("road_test_%Y%m%d_%H%M%S_%f")
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    cfg = build_test_road(folder)
    net = ET.parse(folder / "sumo" / "highway.net.xml").getroot()
    routes = {r.get("id"): r.get("edges").split() for r in
              ET.parse(folder / "sumo" / "highway.ego.rou.xml").getroot().findall("route")}
    pairs = [tuple(args.pair)] if args.pair else [(a, b) for a in arms for b in arms]
    rows = []
    edge_nodes = {edge.get("id"): (edge.get("from"), edge.get("to"))
                  for edge in net.findall("edge") if edge.get("function") != "internal"}
    for origin, destination in pairs:
        route_id = f"ego_d_{origin}_{destination}"
        route = routes.get(route_id, [])
        route_kind = "central" if any(edge_nodes[a][1] == edge_nodes[b][0] == "nC"
                                       for a, b in zip(route, route[1:])) else "bypass"
        rows.append({"origin": origin, "destination": destination, "route_id": route_id,
                     "route": route, "route_kind": route_kind if route else "",
                     "status": "not_run", "reason": "", "warnings": []})
    print("SUMO 기본 운전 / 배경 차량 없음 / 정상 신호 주기 / 순간이동 금지", flush=True)
    print(f"결과 폴더: {folder}", flush=True)
    if not args.nogui and not args.auto_start:
        print("각 조합의 창에서 ▶를 누르세요. 종료 후 다음 조합 창이 열립니다.", flush=True)
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{len(rows)}] {row['origin']} → {row['destination']}", flush=True)
        if not row["route"]:
            row.update(status="no_route", reason="현재 도로망에서 연결 경로를 생성할 수 없음")
        else:
            try:
                row["depart_lane"] = departure_lane(net, row["route"]) if args.depart_lane is None else args.depart_lane
                lanes = net.find(f"edge[@id='{row['route'][0]}']").findall("lane")
                if row["depart_lane"] >= len(lanes):
                    raise IndexError(f"출발 도로의 차선 범위는 0~{len(lanes)-1}")
                kind = "중앙 교차로 경유" if row["route_kind"] == "central" else "우회도로 경유"
                print(f"  경로: {' → '.join(row['route'])} / 출발 차선: {row['depart_lane']} / {kind}", flush=True)
                run_case(cfg, row, args, folder / f"{row['origin']}_{row['destination']}.log")
            except ValueError as exc:
                row.update(status="no_route", reason=str(exc))
            except IndexError as exc:
                row.update(status="error", reason=str(exc))
        print(f"  {LABELS[row['status']]}: {row['reason']}", flush=True)
        save_results(folder, rows)
        if row["status"] == "aborted" or (not args.nogui and row["status"] == "error"):
            break
    by_pair = {(r["origin"], r["destination"]): r["status"] for r in rows}
    print("\n출발 \\ 도착  " + "  ".join(f"{arm:>7}" for arm in arms))
    for origin in arms:
        print(f"{origin:>10}  " + "  ".join(f"{LABELS[by_pair[(origin, dest)]]:>7}" if (origin, dest) in by_pair else f"{'-':>7}" for dest in arms))
    passed = sum(r["status"] == "passed" for r in rows)
    missing = sum(r["status"] == "no_route" for r in rows)
    print(f"\n전체 {len(rows)}개: 통과 {passed}, 경로 없음 {missing}, 기타 {len(rows)-passed-missing}")
    warnings = sorted({warning for row in rows for warning in row.get("warnings", [])})
    if warnings:
        print("SUMO 경고 (도착 성공과 별도로 확인 필요):")
        for warning in warnings:
            print("  " + warning)
    print(f"상세 결과: {folder / 'results.csv'}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
