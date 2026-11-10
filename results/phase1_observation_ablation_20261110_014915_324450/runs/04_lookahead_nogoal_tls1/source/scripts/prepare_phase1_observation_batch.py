"""Prepare four immutable, independent 300k Phase 1 observation experiments.

Only writes a new results/ batch. This command does not enqueue training jobs.
"""
import ast
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    ("01_lookahead_goal_tls2", True, 2, 59),
    ("02_lookahead_nogoal_tls2", False, 2, 54),
    ("03_lookahead_goal_tls1", True, 1, 54),
    ("04_lookahead_nogoal_tls1", False, 1, 49),
)


def replace_values(path, constants=None, dictionaries=None):
    """Replace literal values in a copied source file, preserving unrelated code."""
    constants, dictionaries = constants or {}, dictionaries or {}
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    replacements = []
    found = set()
    for statement in ast.parse(text).body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in constants:
            replacements.append((statement.value, repr(constants[target.id])))
            found.add(target.id)
        if target.id in dictionaries:
            for key, value in zip(statement.value.keys, statement.value.values):
                if key.value in dictionaries[target.id]:
                    replacements.append((value, repr(dictionaries[target.id][key.value])))
                    found.add((target.id, key.value))
    expected = set(constants) | {(name, key) for name, values in dictionaries.items() for key in values}
    assert found == expected, (path, found, expected)
    # All current experiment values are single-line ASCII literals.
    for node, value in sorted(replacements, key=lambda item: (item[0].lineno, item[0].col_offset), reverse=True):
        assert node.lineno == node.end_lineno
        index = node.lineno - 1
        line = lines[index].encode("utf-8")
        lines[index] = (line[:node.col_offset] + value.encode("utf-8") + line[node.end_col_offset:]).decode("utf-8")
    path.write_text("".join(lines))
    ast.parse(path.read_text())


def main():
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    batch = ROOT / "results" / f"phase1_observation_ablation_{stamp}"
    batch.mkdir()
    entries = []
    protected = ["exp_train.md", "docs/train_phase1.md", "env/road_config.py", "env/mdp_config.py",
                 "env/sumo_env.py", "env/sumo/highway.net.xml", "train.py"]
    protected_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in protected}
    for name, destination, signals, dimension in VARIANTS:
        source = batch / "prepared" / name
        source.mkdir(parents=True)
        for folder in ("env", "algorithms", "utils", "scripts"):
            shutil.copytree(ROOT / folder, source / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "sumo"))
        for filename in ("train.py", "test.py", "requirements.txt"):
            shutil.copy2(ROOT / filename, source / filename)
        replace_values(source / "env/road_config.py", constants={
            "ROUTE_ONLY_EXPERIMENT": True, "SIGNAL_EXPERIMENT": True})
        replace_values(source / "env/mdp_config.py", dictionaries={"OBSERVATION": {
            "lookahead_connectivity_enabled": True, "destination_enabled": destination,
            "signal_count": signals}})
        replace_values(source / "train.py", constants={
            "TOTAL_TIMESTEPS": 300000, "RESULTS_DIR": str(batch / "runs"), "RUN_NAME": name})
        run = batch / "runs" / name
        run.mkdir(parents=True)
        manifest = dict(variant=name, source_dir=str(source), run_dir=str(run), target_steps=300000,
                        expected_actual_steps=301056, updates=147, observation_dim=dimension,
                        destination_enabled=destination, signal_count=signals,
                        lookahead_connectivity_enabled=True, all_green=True, background=False,
                        lane_change_penalty=0.02, invalid_action_penalty=0.2,
                        seed=0, from_scratch=True, initial_checkpoint=None, previous_training_steps=0,
                        observation_layout={"original_blocks": [0, 35], "signals": [35, 35 + 5 * signals],
                                            "lookahead": [35 + 5 * signals, 44 + 5 * signals],
                                            "destination": [44 + 5 * signals, dimension] if destination else None})
        launch = source / "launch.sh"
        launch.write_text('''#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SOURCE_DIR"
export TZ=Asia/Seoul PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/lair15/anaconda3/bin/conda run --no-capture-output -n sumo python -u "$SOURCE_DIR/scripts/check_phase1_observation_variant.py" "$SOURCE_DIR" --verify-only
# Every experiment starts from random weights; no --init-model argument.
exec /home/lair15/anaconda3/bin/conda run --no-capture-output -n sumo python -u "$SOURCE_DIR/train.py"
''')
        manifest["source_hashes"] = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*")) if p.is_file()
        }
        (source / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (run / "fresh_initialization.json").write_text(json.dumps(manifest, indent=2) + "\n")
        entries.append(manifest)
    summary = dict(batch_dir=str(batch), mode="four independent fresh Phase 1 experiments",
                   sequential=True, task_spooler_slots_observed=1, variants=entries,
                   protected_files_sha256=protected_hashes, jobs=[])
    (batch / "batch.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(str(batch))


if __name__ == "__main__":
    main()
