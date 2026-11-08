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
records=[]
with tempfile.TemporaryDirectory(prefix="sumo-final-eval-") as temp:
 cfg=road_builder.build(road["ROAD"],road["EGO"],road["TRAFFIC"],temp)
 env=SumoHighwayEnv(cfg,road["ROAD"],road["EGO"],mdp["SIMULATION"],mdp["OBSERVATION"],mdp["ACTION"],training["REWARD"],traffic=road["TRAFFIC"],gui=False)
 agent=runner.load_agent(str(model),env.observation_space.shape[0],env.action_space.shape[0])
 try:
  for seed in range(50):
   metrics=evaluate_policy(agent,env,n_episodes=1,seed=seed)
   records.append({"seed":seed,"route":env.ego_route_id,**metrics.as_dict()})
   if (seed+1)%10==0: print("evaluated",seed+1,flush=True)
 finally: env.close()
summary={k:sum(r[k] for r in records)/len(records) for k in metrics.as_dict()}
result={"model":str(model.relative_to(root)),"training_steps":200704,"episodes":50,"seeds":[0,49],"background_vehicles":False,"summary_episode_means":summary,"episodes_detail":records}
Path(__file__).with_name("final_evaluation_seed0_49.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2),flush=True)
