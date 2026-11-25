"""Evaluate a supervised PPO-compatible BC actor at 0/1000/2000/3000 veh/h.

python scripts/evaluate_quick_bc.py results/bc_actor_lc10000_seed0 --num-workers 4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from env import road_builder
from env.background_demand import set_weighted_od_flow_rates
from env.collision_detection import configure_collision_gaps
from env.ego_routes import read_ego_route_groups
from env.parallel_traffic import TrafficCurriculum
from env.traffic_snapshots import TrafficSnapshotBank
from scripts.evaluate import _read_log
from scripts.evaluate_quick_phase1 import STEP, TICKS, _evaluate_worker, summarize
from scripts.evaluate_quick_phase2 import _plot_overview, _plot_level
from test import load_saved_configs

LEVELS = (0, 1000, 2000, 3000)


def configure_ego_collision_gap(config, gap_m):
    """Override collision detection only; preserve minGap used by observations."""
    setting = configure_collision_gaps(config, ego_gap_m=gap_m)['ego_type']
    return dict(ego_collision_gap_m=gap_m, ego_min_gap_m=setting['min_gap_m'],
                ego_collision_min_gap_factor=setting['collision_min_gap_factor'],
                background_collision_rules='unchanged')


def plot_bc_training(fig, grid, run, evaluated_epoch=None):
    """BC epoch axis; no PPO returns, value losses, or KL panels."""
    rows = _read_log(run / 'training_log.csv')
    panels = (
        ('Supervised BC loss', ('train_loss', 'val_loss'), ('Train', 'Validation')),
        ('Validation acceleration MSE', ('val_accel_mse',), ('Acceleration',)),
        ('Validation weighted lane cross entropy', ('val_lane_ce',), ('Lane CE',)),
        ('Validation lane macro F1', ('val_lane_macro_f1',), ('Macro F1',)),
        ('Validation lane-change precision', ('val_left_precision', 'val_right_precision'), ('Left', 'Right')),
        ('Validation lane-change recall', ('val_left_recall', 'val_right_recall'), ('Left', 'Right')),
    )
    best_path = run / 'result.json'
    best_epoch = evaluated_epoch if evaluated_epoch is not None else (json.loads(best_path.read_text()).get('best_epoch') if best_path.exists() else None)
    for i, (title, keys, labels) in enumerate(panels):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        plotted = False
        for key, label in zip(keys, labels, strict=True):
            points = [(float(row['epoch']), float(row[key])) for row in rows
                      if row.get('epoch') and row.get(key) not in (None, '')]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, label=label)
                plotted = True
        if best_epoch is not None and plotted:
            ax.axvline(best_epoch, color='#64748b', linestyle=':', linewidth=1, label='Evaluated checkpoint')
        if plotted:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'No BC metric logged', ha='center', va='center', transform=ax.transAxes)
        ax.set(title=title, xlabel='Epoch')
        if i >= 3:
            ax.set_ylim(0, 1.03)
        ax.grid(alpha=.25)


def prepare_context(run, context, data_dir=None, *, max_vph=3000):
    """Restore collection conditions without relabelling BC as PPO training."""
    config = json.loads((run / 'config.json').read_text())
    if config.get('algorithm') != 'supervised_bc_on_ppo_hybrid_actor':
        raise ValueError('Use a train_bc_actor.py result folder; legacy regression BC is unsupported')
    data_dir = Path(data_dir) if data_dir else (Path(config['dataset_directory']) if config.get('dataset_directory') else Path(config['data']).parent)
    collection = json.loads((data_dir / 'config.json').read_text())
    if collection.get('observation', {}).get('layout') not in ('bc43', 'bc49'):
        raise ValueError('The dataset must use the bc43 observation layout')
    spawn_protect=float(collection.get('expert_control_timing', {}).get('spawn_protect',0.0))
    context.mkdir(parents=True, exist_ok=False)
    for name in ('model.pt', 'road_config.py', 'mdp_config.py', 'train.py'):
        shutil.copy2(run / name, context / name)
    # Recorded road geometry stays in the BC run snapshot. Collection restores
    # cyclic signals and traffic templates disabled by the default training flags.
    with (context / 'road_config.py').open('a') as file:
        file.write('\n# BC evaluation context (not a PPO-trained run).\n')
        file.write('ROUTE_ONLY_EXPERIMENT = False\nSIGNAL_EXPERIMENT = True\n')
        file.write('TRAFFIC = ' + repr(collection['traffic']) + '\n')
        file.write("NOHYEONG['all_signals_green'] = False\nNOHYEONG['warmup_seconds'] = 0\n")
        file.write("for demand, vph in zip(NOHYEONG['ring_segments'] + NOHYEONG['loop_routes'], _RING_BACKGROUND_VPH, strict=True):\n    demand['vph'] = vph\n")
    with (context / 'mdp_config.py').open('a') as file:
        file.write('\nOBSERVATION = ' + repr(collection['observation']) + '\n')
        file.write('SIMULATION = ' + repr(collection['simulation']) + '\n')
        # The expert's collection timing (spawn_protect) restricted the expert only.
        # The BC ego runs without any rule-based lane-change restriction.
        file.write("\n# The ego executes its lane commands without rule-based restrictions.\n")
        file.write("from copy import deepcopy as _copy_action\nACTION = _copy_action(ACTION)\n"
                   "ACTION['lane_change_safety'] = dict(ACTION.get('lane_change_safety', {}), "
                   "enabled=False, spawn_protect_only=False, spawn_protect=0.0)\n")
    with (context / 'train.py').open('a') as file:
        file.write('\nHPARAMS.update(' + repr(dict(hidden_sizes=tuple(config['hidden_sizes']),
                        activation=config['activation'], policy_type='hybrid')) + ')\n')
    reference = collection['traffic']['vehs_per_hour']
    curriculum = TrafficCurriculum(max_vph=max_vph, total_vph=True, snapshot_warmup_seconds=250,
                                    ring_reference_vph=reference)
    from dataclasses import asdict
    saved = dict(evaluation_only=True, bc_spawn_protect_m=0.0, expert_spawn_protect_m=spawn_protect, training_algorithm=config['algorithm'],
                 traffic_curriculum=asdict(curriculum), randomize_departures=True,
                 independent_episode_rng=True, seed=0)
    (context / 'evaluation_config.json').write_text(json.dumps(saved, indent=2) + '\n')
    return saved, curriculum, data_dir.resolve()


def evaluate(run, out, num_workers, data_dir=None, checkpoint=None, ego_collision_gap_m=None,
             background_collision_gap_m=None, levels=LEVELS, *,
             context_builder=None, training_plotter=None, policy_label='BC'):
    start = perf_counter()
    build_context = context_builder or prepare_context
    saved, curriculum, data_dir = build_context(run, out / 'context', data_dir, max_vph=max(3000, *levels))
    model = out / 'context/model.pt'
    checkpoint = Path(checkpoint).resolve() if checkpoint else (run / 'model.pt').resolve()
    if checkpoint != (run / 'model.pt').resolve():
        shutil.copy2(checkpoint, model)
    evaluated_epoch = None
    result_path = run / 'result.json'
    if training_plotter is None and result_path.exists():
        training_result = json.loads(result_path.read_text())
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        for filename, epoch_key in [('last_model.pt','epochs_completed'),('model.pt','best_epoch')]:
            candidate = run / filename
            if candidate.exists() and hashlib.sha256(candidate.read_bytes()).hexdigest() == digest:
                evaluated_epoch = training_result.get(epoch_key)
                break
    details = out / 'details'
    details.mkdir()
    road, mdp, _ = load_saved_configs(str(model))
    rows = {vph: [] for vph in levels}
    summaries = {}
    with TemporaryDirectory(prefix='sumo-quick-bc-') as directory:
        root = Path(directory)
        config = road_builder.build(road['ROAD'], road['EGO'], road['TRAFFIC'], str(root / 'base'))
        ego_gap = ego_collision_gap_m if ego_collision_gap_m is not None else road['EGO'].get('collision_gap_m')
        bg_gap = background_collision_gap_m if background_collision_gap_m is not None else road['TRAFFIC'].get('collision_gap_m')
        collision_settings = dict(ego_collision_gap_m=ego_gap, background_collision_gap_m=bg_gap,
            vehicle_types=configure_collision_gaps(config, ego_gap_m=ego_gap, background_gap_m=bg_gap))
        saved['collision_detection'] = collision_settings
        (out / 'context/evaluation_config.json').write_text(json.dumps(saved, indent=2) + '\n')
        shutil.copy2(root / 'base/highway.rou.xml', out / 'context/evaluation_vehicle_types.rou.xml')
        if road['TRAFFIC'].get('demand_profile') == 'weighted_od':
            set_weighted_od_flow_rates(root/'base/highway.rou.xml', max(levels))
        elif road['TRAFFIC'].get('origin_lane_proportional'):
            from collect_background_data import rebalance_origin_flows
            rebalance_origin_flows(root/'base', 3000)
        direct, ring = read_ego_route_groups(root / 'base/highway.ego.rou.xml')
        if (len(direct), len(ring)) != (18, 5):
            raise ValueError('Expected 18 non-ring and 5 ring routes')
        routes = direct + ring
        bank = TrafficSnapshotBank(config, curriculum, road['TRAFFIC']['vehs_per_hour'],
                                   tuple(v for v in levels if v), STEP, 0, refresh=False)
        try:
            chunks = np.array_split(TICKS, max(2, num_workers))
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as pool:
                jobs = {
                    pool.submit(_evaluate_worker, model, config, root / f'vph_{level}_{i}', saved, level,
                                [(j, route, int(tick)) for j, route in enumerate(routes) for tick in chunk],
                                snapshot_path=bank.path): level
                    for level in levels for i, chunk in enumerate(chunks)
                }
                for job in as_completed(jobs):
                    level = jobs[job]
                    rows[level].extend(job.result())
                    if len(rows[level]) == len(routes) * len(TICKS):
                        rows[level].sort(key=lambda r: (r['route'], r['signal_offset_s']))
                        summaries[level] = dict(all=summarize(rows[level]), by_route={
                            route: summarize([r for r in rows[level] if r['route'] == route]) for route in routes})
                        print(f'{level:4} veh/h: {summaries[level]["all"]["statuses"].get("arrival", 0)}/'
                              f'{len(rows[level])} arrivals', flush=True)
        finally:
            bank.close()
    summary = dict(algorithm=saved['training_algorithm'], model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                   source_run=str(run), source_checkpoint=str(checkpoint), evaluated_epoch=evaluated_epoch,
                   source_dataset=str(data_dir) if data_dir is not None else None, traffic_levels_vph=list(levels),
                   traffic_vph_definition='Total requested background demand, including ring traffic',
                   traffic_demand_profile=road['TRAFFIC'].get('demand_profile', 'default'),
                   randomize_departures=saved.get('randomize_departures', True), deterministic=True,
                   observation_neighbor_distance_mode=mdp['OBSERVATION'].get('neighbor_distance_mode', 'legacy'),
                   signal_offsets_s=[tick * STEP for tick in TICKS], episodes_per_level=len(routes) * len(TICKS),
                   elapsed_seconds=round(perf_counter() - start, 2), by_traffic_vph=summaries)
    summary['collision_detection'] = collision_settings
    if 'ppo_training' in saved:
        summary['ppo_training'] = saved['ppo_training']
    summary['red_wait_clearance_definition'] = dict(
        statistic='Last qualifying red-light stop in each episode; one sample per episode',
        ego_stop_speed_max_mps=0.5, leader_stop_speed_max_mps=0.5,
        selection='Nearer of stop line and stopped leader; grouped by recorded reference',
        legacy_mixed_key='red_wait_clearance', split_key='red_wait_clearance_by_reference')
    # Save numeric results before rendering so they survive plotting failures.
    (out / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'route_evaluation.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[levels[0]][0]))
        writer.writeheader()
        writer.writerows(row for level in levels for row in rows[level])
    with (out / 'red_wait_clearance_by_reference.csv').open('w', newline='') as file:
        fields=['traffic_vph','route','reference','n','min_m','q10_m','q1_m','median_m','q3_m','q90_m','max_m','mean_m']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for level in levels:
            for route, stats in [('ALL',summaries[level]['all']), *summaries[level]['by_route'].items()]:
                for reference, values in stats['red_wait_clearance_by_reference'].items():
                    writer.writerow(dict(traffic_vph=level,route=route,reference=reference,**values))
    if training_plotter is None:
        def training_plotter(fig, grid, source):
            plot_bc_training(fig, grid, source, evaluated_epoch=evaluated_epoch)
    _plot_overview(run, out, routes, levels, summaries,
                   training_plotter=training_plotter, policy_label=policy_label)
    for level in levels:
        _plot_level(details, level, routes, rows[level], summaries[level], policy_label=policy_label)
    print(f'Complete: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='BC actor result folder or an explicit .pt checkpoint (including selected_model.pt)')
    parser.add_argument('--traffic-vph', type=int, nargs='+', default=LEVELS,
                        help='Traffic levels to evaluate (default: 0 1000 2000 3000)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-dir', type=Path, help='New output directory; default: RUN/analysis/quick_bc')
    parser.add_argument('--data-dir', type=Path, help='Collection directory if the dataset was relocated')
    parser.add_argument('--ego-collision-gap-m', type=float,
                        help='Ego collision detection gap in metres; preserves observation minGap and background rules')
    parser.add_argument('--background-collision-gap-m', type=float,
                        help='Background/expert collision detection gap in metres; default: saved traffic configuration')
    args = parser.parse_args()
    run = args.run.resolve()
    checkpoint = run if run.is_file() and run.suffix == '.pt' else None
    if checkpoint is not None:
        run = run.parent
    if not all((run / name).is_file() for name in ('model.pt', 'config.json', 'road_config.py', 'mdp_config.py', 'train.py')):
        parser.error('Need the BC actor checkpoint, config.json and saved environment/training snapshots')
    if not 1 <= args.num_workers <= len(TICKS):
        parser.error(f'num-workers must be 1..{len(TICKS)}')
    for name in ('ego_collision_gap_m', 'background_collision_gap_m'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(name.replace('_', '-') + ' must be finite and nonnegative')
    if any(v < 0 for v in args.traffic_vph) or len(set(args.traffic_vph)) != len(args.traffic_vph):
        parser.error('--traffic-vph requires distinct nonnegative values')
    out = (args.out_dir or run / 'analysis/quick_bc').resolve()
    if out.exists():
        parser.error('Output directory already exists; choose a new --out-dir')
    try:
        evaluate(run, out, args.num_workers, args.data_dir, checkpoint=checkpoint,
                 ego_collision_gap_m=args.ego_collision_gap_m,
                 background_collision_gap_m=args.background_collision_gap_m, levels=args.traffic_vph)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
