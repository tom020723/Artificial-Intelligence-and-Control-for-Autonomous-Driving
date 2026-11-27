"""Create a results directory for a BC training run."""

from datetime import datetime
from itertools import count
from pathlib import Path


def create_bc_result_dir(out_dir: Path | None = None) -> Path:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=False)
        return out_dir

    results = Path(__file__).resolve().parents[1] / "results"
    stem = datetime.now().strftime("BC_%Y%m%d_%H%M%S")
    for index in count(1):
        name = stem if index == 1 else f"{stem}_{index}"
        path = results / name
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return path
