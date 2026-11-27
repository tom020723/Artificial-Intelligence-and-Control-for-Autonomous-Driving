# Benchmark definition

`scripts/evaluate_fixed_routes.py prepare` builds a suite's `reference/` settings
from the repository code plus the three files here, so reproduction needs no
earlier experiment output.

| Suite file | Built from |
|---|---|
| `reference/road_config.py` | `env/road_config.py` + `road_context.py` |
| `reference/mdp_config.py` | `env/mdp_config.py` + `mdp_context.py` |
| `reference/train.py` | `train.py` (reward and hyperparameters) |
| traffic curriculum (`suite.json` `settings`) | `traffic_curriculum.json`, with `max_vph` set to `--traffic-vph` and an 800 s snapshot warm-up |

- `road_context.py`: background traffic (four car-following models and their
  shares, 0.2 m collision gap), live signals, ring-road demand, and a
  center-lane ego departure.
- `mdp_context.py`: the base observation (bc49 layout with split-guidance
  lateral distances). The ego action settings stay as in `env/mdp_config.py`,
  with every rule-based lane-change restriction off.
- `traffic_curriculum.json`: how the background inflow is split between the
  central roads and the ring roads.

These values match the 2026-11-19 reference run
(`results/eval_fixed_routes_20261119_214714/reference`) that the final models
were built from. `docs/REPRODUCE.md` describes how this was checked.
