"""Collect a full BC49 physical-gap dataset, train fresh, then evaluate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-workers', type=int, default=12, help='Collection workers')
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error('--num-workers must be positive')
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    control = root / 'results' / f'bc49_bumper_full_pipeline_{tag}'
    control.mkdir()
    data = root / 'data' / f'background_bc49_bumper_3000_noacc_{tag}'
    run = root / 'results' / f'BC_{tag}'
    source = control / 'source'
    source.mkdir()
    for name in ['env', 'algorithms', 'utils', 'scripts']:
        shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['train.py', 'train_bc.py', 'train_bc_actor.py', 'test.py', 'collect_background_data.py']:
        shutil.copy2(root / name, source / name)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*.py')}
    (control / 'source_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    plan = dict(data=str(data), run=str(run), target_samples=110000000,
                max_collection_episodes=10000, collection_workers=args.num_workers, collection_seed=99000, training_seed=0,
                traffic_vph=3000, origin_lane_proportional=True, expert_types=['car_krauss','car_idm','car_eidm'],
                background_types=['car_krauss','car_idm','car_eidm','car_acc'],
                observation_dim=49, neighbor_distance_mode='bumper', collision_gap_m=.2,
                init_model=None, epochs=100, patience=100, training_wall_limit_s=21600,
                evaluation='best validation checkpoint; 1150 episodes at each of 0/1000/2000/3000 veh/h')
    (control / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    def update(stage, **extra):
        record = {**plan, **extra, 'stage':stage, 'updated_utc':datetime.now(timezone.utc).isoformat()}
        p = control / 'status.json'
        temp = p.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(p)
        print(json.dumps(record), flush=True)
    def execute(stage, arguments, timeout):
        log = control / f'{stage}.log'
        update(stage, log=str(log))
        with log.open('w') as file:
            subprocess.run([sys.executable, '-u', *arguments], cwd=source,
                           stdout=file, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    try:
        execute('collecting', ['collect_background_data.py', '--num-workers', str(args.num_workers),
                              '--out-dir', str(data)], 43200)
        manifest = json.loads((data / 'manifest.json').read_text())
        if manifest['samples'] < plan['target_samples']:
            raise ValueError('Collection ended below the requested sample target')
        execute('validating', ['scripts/check_bc_shards.py',str(data)], 3600)
        execute('training', ['scripts/train_bc_shards.py',str(data),'--out-dir',str(run),
                '--epochs','100','--wall-seconds','21600','--batch-size','4096','--lr','0.0003',
                '--seed','0','--patience','100','--device','cuda'], 22500)
        config = json.loads((run / 'config.json').read_text())
        if json.loads((run / 'result.json').read_text())['epochs_completed'] != 100:
            raise RuntimeError('Training did not reach 100 epochs')
        assert config['init_model'] is None
        assert config['observation_neighbor_distance_mode'] == 'bumper'
        shutil.copy2(run / 'model.pt', run / 'selected_model.pt')
        execute('evaluating', ['scripts/evaluate_quick_bc.py',str(run / 'selected_model.pt'),
                '--num-workers','4','--ego-collision-gap-m','0.2','--background-collision-gap-m','0.2',
                '--out-dir',str(run / 'analysis/bumper_full')], 600)
        result = json.loads((run / 'analysis/bumper_full/evaluation_summary.json').read_text())
        update('complete', samples=manifest['samples'],
               retained_samples=json.loads((data / 'manifest.json').read_text())['samples'],
               training=json.loads((run / 'result.json').read_text()),
               evaluation={level:values['all']['statuses'] for level,values in result['by_traffic_vph'].items()})
    except BaseException as exc:
        update('failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
