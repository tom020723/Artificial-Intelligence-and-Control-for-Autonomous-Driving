"""Watch one seed of the mixed-traffic BC benchmark in SUMO GUI."""
import argparse
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_fixed_routes import prepare_road, read_json
from scripts.evaluate_mixed_traffic import DEFAULT_SECONDS, DEFAULT_SEEDS, run_road


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="Fixed-demand suite providing the calibrated road and flow weights")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vph", type=int, help="Total demand including ego (defaults to the suite demand)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS.start)
    parser.add_argument("--seconds", type=int, default=DEFAULT_SECONDS)
    parser.add_argument("--delay", type=int, default=80, help="GUI delay in milliseconds per step")
    parser.add_argument("--spawn-protect-m", type=float, default=0.,
                        help="Initial lane-change restriction in meters (default: 0, disabled)")
    args = parser.parse_args()
    if not math.isfinite(args.spawn_protect_m) or args.spawn_protect_m < 0:
        parser.error("--spawn-protect-m must be finite and nonnegative")
    suite = args.suite.resolve(strict=True)
    model = args.model.resolve(strict=True)
    spec = read_json(suite / "suite.json")
    vph = spec["traffic_vph"] if args.vph is None else args.vph
    if spec.get("adaptive_traffic") or min(vph, args.seconds) < 1 or min(args.seed, args.delay) < 0:
        parser.error("Use a fixed-demand suite, positive traffic and time, and nonnegative seed and delay")
    with TemporaryDirectory(prefix="sumo-mixed-play-") as work:
        cfg, flows, lanes = prepare_road(suite, Path(work), spec)
        run_road(cfg, flows, lanes, suite, model, vph, args.seconds, args.seed,
                 gui_delay=args.delay, spawn_protect_m=args.spawn_protect_m)


if __name__ == "__main__":
    main()
