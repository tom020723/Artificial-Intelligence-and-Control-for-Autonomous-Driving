"""Find a sustainable background demand split by simulation feedback.

usage (repository root):
    python scripts/tune_background_demand.py <suite_dir> <start_shares.json|-> <out.json> [seconds=3600] [seeds=3] [tune|check]

<suite_dir> is a suite made by scripts/evaluate_fixed_routes.py prepare (its sumo/ folder is used).
"-" starts from env.background_demand.FLOW_SHARES. "check" only reports, without changing shares.
TUNE_LIMITS="p95_wait,max_wait,p95_depart_delay" (seconds) overrides the health limits.

Background-only SUMO runs with Poisson arrivals. A flow is unhealthy when its
vehicles wait too long (p95 waiting > one signal cycle + 10 s, or any wait > two
cycles + 10 s; the cycle is read from the suite's network), cannot be
inserted (depart delay), or do not finish. Unhealthy flows are scaled down 8 %
and the run repeats until every flow is healthy for all seeds.
"""
import json, shutil, subprocess, sys, collections, tempfile, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

suite, start, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
seconds = int(sys.argv[4]) if len(sys.argv) > 4 else 3600
n_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 3
mode = sys.argv[6] if len(sys.argv) > 6 else "tune"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sumolib import checkBinary  # noqa: E402
from env.background_demand import FLOW_SHARES  # noqa: E402

SUMO = checkBinary("sumo")
# Longest signal cycle in the suite's network (125 s sequential plan, 120 s grouped plan, ...).
CYCLE = max(sum(float(phase.get("duration")) for phase in logic.findall("phase"))
            for logic in ET.parse(suite / "sumo/highway.net.xml").getroot().findall("tlLogic"))
WAIT_P95, WAIT_MAX, DELAY_P95 = (float(x) for x in os.environ.get(
    "TUNE_LIMITS", f"{CYCLE + 10},{2 * CYCLE + 10},20").split(","))
print(f"signal cycle {CYCLE:g} s; limits: p95 wait {WAIT_P95:g} s, max wait {WAIT_MAX:g} s, "
      f"p95 depart delay {DELAY_P95:g} s", flush=True)
work = Path(tempfile.mkdtemp(prefix="tune_"))
shutil.copytree(suite / "sumo", work / "sumo")
rou = work / "sumo" / "highway.rou.xml"
template = rou.read_text(encoding="utf-8")


def run(shares, seed):
    tree = ET.ElementTree(ET.fromstring(template))
    for flow in tree.getroot().findall("flow"):
        flow.set("period", f"exp({shares[flow.get('id')] / 3600:.6f})")
        flow.set("end", str(seconds))
    path = work / "sumo" / f"rou_{seed}.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    trip = work / f"trip_{seed}.xml"
    subprocess.run([SUMO, "-n", str(work / "sumo/highway.net.xml"), "-r", str(path),
                    "--step-length", "0.5", "--seed", str(seed), "--time-to-teleport", "-1",
                    "--no-warnings", "true", "--no-step-log", "true", "--end", str(seconds + 500),
                    "--tripinfo-output", str(trip), "--tripinfo-output.write-unfinished", "true",
                    "--max-depart-delay", "100000"], check=True, capture_output=True)
    stats = collections.defaultdict(lambda: dict(wait=[], delay=[], done=0, n=0))
    for t in ET.parse(trip).getroot().findall("tripinfo"):
        s = stats[t.get("id").rsplit(".", 1)[0]]
        if float(t.get("depart")) < 600:        # warm-up: network still filling
            continue
        s["n"] += 1
        s["done"] += float(t.get("arrival")) >= 0
        s["wait"].append(float(t.get("waitingTime")))
        s["delay"].append(float(t.get("departDelay")))
    return stats


def evaluate(shares):
    with ThreadPoolExecutor(n_seeds) as pool:
        results = list(pool.map(lambda seed: run(shares, seed), range(1, n_seeds + 1)))
    report = {}
    for fid in shares:
        wait = np.concatenate([np.array(r[fid]["wait"] or [0.0]) for r in results])
        delay = np.concatenate([np.array(r[fid]["delay"] or [0.0]) for r in results])
        n = sum(r[fid]["n"] for r in results)
        done = sum(r[fid]["done"] for r in results)
        expected = shares[fid] * (seconds - 600) / 3600 * n_seeds
        report[fid] = dict(vph=shares[fid], p95_wait=float(np.percentile(wait, 95)), max_wait=float(wait.max()),
                           p95_delay=float(np.percentile(delay, 95)), done=done / max(n, 1),
                           inserted=n / max(expected, 1))
    return report


def unhealthy(r):
    return (r["p95_wait"] > WAIT_P95 or r["max_wait"] > WAIT_MAX or r["p95_delay"] > DELAY_P95
            or r["done"] < 0.999 or r["inserted"] < 0.9)


shares = {k: float(v) for k, v in (FLOW_SHARES if str(start) == "-" else json.loads(start.read_text())).items()}
for iteration in range(40):
    report = evaluate(shares)
    bad = [fid for fid, r in report.items() if unhealthy(r)]
    print(f"iter {iteration:2d} total {sum(shares.values()):6.0f} vph  unhealthy {len(bad)}: "
          + " ".join(f"{fid.replace('lf_RING_', 'R:').replace('f_', '')}({report[fid]['p95_wait']:.0f}/{report[fid]['max_wait']:.0f}/{report[fid]['p95_delay']:.0f})" for fid in bad), flush=True)
    if not bad or mode == "check":
        break
    for fid in bad:
        shares[fid] *= 0.92
print("\nfinal per-flow (vph, p95 wait, max wait, p95 depart delay, finished):")
for fid, r in sorted(report.items()):
    print(f"  {fid:16s} {shares[fid]:5.0f}  {r['p95_wait']:5.0f} {r['max_wait']:5.0f} {r['p95_delay']:5.0f}  {r['done']:.3f}{'   <-- unhealthy' if unhealthy(r) else ''}")
out.write_text(json.dumps({k: round(v, 1) for k, v in shares.items()}, indent=1))
print("total", round(sum(shares.values())), "->", out)
shutil.rmtree(work, ignore_errors=True)
