import sys, json, tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
import torch
import test as runner
from env import road_builder
from env.sumo_env import SumoHighwayEnv
from utils.evaluator import evaluate_policy

torch.set_num_threads(1)
model=Path(__file__).resolve().parents[1]/"model.pt"
road,mdp,training=runner.load_saved_configs(str(model))
runner.HPARAMS=training["HPARAMS"]
import numpy as np, traci
records=[]
with tempfile.TemporaryDirectory(prefix="sumo-trace-") as temp:
 cfg=road_builder.build(road["ROAD"],road["EGO"],road["TRAFFIC"],temp)
 env=SumoHighwayEnv(cfg,road["ROAD"],road["EGO"],mdp["SIMULATION"],mdp["OBSERVATION"],mdp["ACTION"],training["REWARD"],traffic=road["TRAFFIC"],gui=False)
 agent=runner.load_agent(str(model),env.observation_space.shape[0],env.action_space.shape[0])
 try:
  for seed in [0,1,8]:
   np.random.seed(seed);obs,_=env.reset(seed=seed);steps=[]
   while True:
    action=agent.predict(obs,deterministic=True)
    lane=traci.vehicle.getLaneID("ego")
    row={"step":len(steps),"lane":lane,"speed":traci.vehicle.getSpeed("ego"),"remaining":traci.lane.getLength(lane)-traci.vehicle.getLanePosition("ego"),"connectivity":obs[21:26].tolist(),"signals":env._upcoming_signals(),"action":action.tolist()}
    obs,reward,term,trunc,info=env.step(action)
    row.update(reward=reward,wrong_lane=info.get("wrong_lane"),red_violation=info.get("red_light_violation"),lane_applied=info.get("lane_change_applied"))
    steps.append(row)
    if term or trunc:break
   records.append({"seed":seed,"route":env.ego_route_id,"steps":steps})
   print("TRACE",seed,env.ego_route_id,"return",sum(x["reward"] for x in steps))
   for row in steps[-7:]:print(json.dumps(row))
 finally:env.close()
Path(__file__).with_name("failure_traces.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
