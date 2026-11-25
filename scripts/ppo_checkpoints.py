"""Prepare saved PPO checkpoints for the BC evaluators and pick the best one.

The fixed-route and diverse start-lane evaluators read a BC training folder
(configuration files plus ``model.pt``). The PPO actor has the same network and
observation as its initial BC, so ``export`` copies the BC folder's
configuration next to each checkpoint:

    python scripts/ppo_checkpoints.py export results/PPO_x --iterations 50 60 70 80 90 100
      -> results/PPO_x/eval_runs/iter050/{config.json, ..., model.pt}

After evaluating every exported checkpoint on the validation suite into
``eval_runs/iterNNN/<evaluation>``, ``select`` reports the success rates and
writes the best one to ``results/PPO_x/selection.json``:

    python scripts/ppo_checkpoints.py select results/PPO_x --evaluation evaluation_validation
"""

import argparse
import json
from pathlib import Path
import shutil

BC_FILES = ("config.json", "mdp_config.py", "normalization.npz", "road_config.py", "train.py")


def export(run, iterations):
    initial = Path(json.loads((run / "run.json").read_text())["initial_model"])
    bc = initial.parent
    if not all((bc / name).is_file() for name in BC_FILES):
        raise FileNotFoundError(f"{bc} is not a BC training folder with {', '.join(BC_FILES)}")
    exported = []
    for iteration in iterations:
        checkpoint = run / f"model_iter{iteration:03d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{checkpoint} does not exist (checkpoints are saved every 10 iterations)")
        target = run / "eval_runs" / f"iter{iteration:03d}"
        target.mkdir(parents=True, exist_ok=True)
        for name in BC_FILES:
            shutil.copy2(bc / name, target / name)
        shutil.copy2(checkpoint, target / "model.pt")
        exported.append(target)
        print(target / "model.pt", flush=True)
    return exported


def select(run, evaluation):
    rows = []
    for folder in sorted((run / "eval_runs").glob("iter[0-9][0-9][0-9]")):
        summary = folder / evaluation / "evaluation_summary.json"
        if summary.is_file():
            overall = json.loads(summary.read_text())["overall"]
            rows.append(dict(iteration=int(folder.name[4:]), success_rate=overall["success_rate"],
                             episodes=overall["episodes"], model=str(folder / "model.pt")))
    if not rows:
        raise FileNotFoundError(f"No {evaluation}/evaluation_summary.json under {run / 'eval_runs'}")
    if len({row["episodes"] for row in rows}) != 1:
        raise ValueError("Checkpoints were evaluated on different numbers of cases")
    # Ties go to the later checkpoint.
    best = max(rows, key=lambda row: (row["success_rate"], row["iteration"]))
    for row in rows:
        print(f"iter{row['iteration']:03d}  {row['success_rate']:.2%}{'  <- best' if row is best else ''}")
    (run / "selection.json").write_text(json.dumps(dict(evaluation=evaluation, best=best, candidates=rows),
                                                   indent=2) + "\n")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("run", type=Path)
    exporter.add_argument("--iterations", type=int, nargs="+", required=True)
    selector = commands.add_parser("select")
    selector.add_argument("run", type=Path)
    selector.add_argument("--evaluation", default="evaluation_validation",
                          help="Evaluation folder name inside each eval_runs/iterNNN")
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    if args.command == "export":
        if any(i < 1 for i in args.iterations):
            parser.error("Iterations must be positive")
        export(run, args.iterations)
    else:
        select(run, args.evaluation)


if __name__ == "__main__":
    main()
