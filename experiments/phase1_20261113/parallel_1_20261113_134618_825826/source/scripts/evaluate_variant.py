"""Evaluate a saved Phase 1 observation variant on 27 route IDs and five seeds.

Usage (from the project root): python scripts/evaluate_variant.py <run_directory>
Loads the run's source/ and model.pt; writes results into its analysis/ directory.
Requires all-green signals, no background traffic, and fresh_initialization.json.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
args = parser.parse_args()
RUN = args.run.resolve()
OUT = RUN / 'analysis'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(RUN / 'source'))

import numpy as np
import traci
from env.road_config import ROAD, EGO, TRAFFIC, NOHYEONG
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.road_builder import build
from env.sumo_env import SumoHighwayEnv
from train import HPARAMS, REWARD
from algorithms.ppo import PPO
from utils.evaluator import evaluate_policy

class RecordedEnv(SumoHighwayEnv):
    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.trace = []
        self.last_info = {}
        self.start_lane = traci.vehicle.getLaneID(self.ego_id)
        return result

    def step(self, action):
        lane = traci.vehicle.getLaneID(self.ego_id)
        observation = self._get_obs()
        connectivity_start = 1 + 4 * len(self.obs_offsets)
        lookahead_start = connectivity_start + len(self.connectivity_offsets) + len(self.obs_offsets) + 5 * self.signal_count
        before = dict(step=self.step_count, lane=lane,
                      position=traci.vehicle.getLanePosition(self.ego_id),
                      remaining=traci.lane.getLength(lane)-traci.vehicle.getLanePosition(self.ego_id),
                      speed=traci.vehicle.getSpeed(self.ego_id),
                      route_index=traci.vehicle.getRouteIndex(self.ego_id),
                      connected=self._lane_follows_route(lane),
                      connectivity_obs=observation[connectivity_start:connectivity_start+len(self.connectivity_offsets)].tolist(),
                      lookahead_obs=observation[lookahead_start:lookahead_start+len(self.connectivity_offsets)].tolist(),
                      action=np.asarray(action).tolist())
        result = super().step(action)
        assert result[0].shape == (self.obs_dim,) and np.isfinite(result[0]).all()
        before.update(reward=result[1], terminated=result[2], truncated=result[3], info=result[4])
        self.trace.append(before)
        self.last_info=result[4]
        return result

def main():
    assert NOHYEONG['all_signals_green'] and not TRAFFIC['background_enabled']
    cfg=build(ROAD,EGO,TRAFFIC,str(OUT/'sumo'))
    routes={r.get('id'):r.get('edges').split() for r in
            ET.parse(OUT/'sumo/highway.ego.rou.xml').getroot().findall('route')
            if r.get('id').startswith(('ego_d_','ego_r_'))}
    assert len(routes)==27 and sum(r.startswith('ego_d_') for r in routes)==22
    env=RecordedEnv(cfg,ROAD,EGO,SIMULATION,OBSERVATION,ACTION,REWARD,traffic=TRAFFIC)
    manifest=json.loads((RUN/'fresh_initialization.json').read_text())
    assert env.obs_dim==manifest['observation_dim'] and env.lookahead_connectivity_enabled
    assert env.destination_enabled==manifest['destination_enabled'] and env.signal_count==manifest['signal_count']
    agent=PPO(obs_dim=env.obs_dim,act_dim=env.action_space.shape[0],**HPARAMS)
    agent.load(str(RUN/'model.pt'))
    agent.policy.eval()
    rows=[]; traces={}
    try:
        for route in sorted(routes):
            env._choose_ego_route=lambda route=route:route
            for seed in range(5):
                metrics=evaluate_policy(agent,env,n_episodes=1,deterministic=True,seed=seed).as_dict()
                info=env.last_info
                status=next((name for key,name in [('red_light_violation','red_light'),('collided','collision'),
                                                      ('arrived','arrival'),('wrong_lane','wrong_lane')] if info.get(key)),
                            'timeout' if env.trace[-1]['truncated'] else 'unknown')
                row=dict(route=route,seed=seed,status=status,start_lane=env.start_lane,
                         final_lane=env.trace[-1]['lane'],**metrics)
                rows.append(row)
                traces[f'{route}_seed{seed}']=env.trace
                with (OUT/'route_evaluation.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                print(f"{route} seed={seed}: {status}, return={metrics['ep_return']:.2f}, "
                      f"steps={metrics['ep_length']:.0f}, changes={metrics['lane_changes']:.0f}",flush=True)
    finally:
        env.close()
        (OUT/'route_traces.json').write_text(json.dumps(traces,separators=(',',':')))
    def group(selected):
        return dict(episodes=len(selected),statuses=dict(Counter(r['status'] for r in selected)),
                    **{k:float(np.mean([r[k] for r in selected])) for k in
                       ['success_rate','wrong_lane_rate','timeout_rate','ep_return','lane_changes','mean_speed','ep_length','green_passes']})
    summary=dict(model=str(RUN/'model.pt'),model_sha256=hashlib.sha256((RUN/'model.pt').read_bytes()).hexdigest(),
                 source='saved source/',observation_dim=env.obs_dim,destination_enabled=env.destination_enabled,
                 signal_count=env.signal_count,lookahead=True,seeds=list(range(5)),all_green=True,background=False,
                 deterministic=True,departure=EGO['depart_lane'],
                 groups=dict(all=group(rows),direct_od=group([r for r in rows if r['route'].startswith('ego_d_')]),
                             explicit_ring=group([r for r in rows if r['route'].startswith('ego_r_')])))
    (OUT/'evaluation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
