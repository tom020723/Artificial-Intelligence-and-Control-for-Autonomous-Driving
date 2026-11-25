"""Memory-bounded BC training on episode NPZ shards, with a wall-time limit."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.bc_results import create_bc_result_dir
from utils.networks import HybridActorCritic
from train import HPARAMS


def lane_class(actions):
    """PPO order: left +1 => 0, keep 0 => 1, right -1 => 2."""
    if not np.isin(actions[:,1], [-1,0,1]).all():
        raise ValueError('Hybrid BC requires lane command labels in {-1,0,+1}')
    return (1-actions[:,1]).astype(np.int64)


def export_policy(model, mean, std):
    """Fold normalization into first Linear for unchanged raw-observation PPO."""
    state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    mean, std=mean.detach().cpu(), std.detach().cpu()
    weight=state['trunk.0.weight']/std.unsqueeze(0)
    state['trunk.0.bias']-=weight@mean
    state['trunk.0.weight']=weight
    return state


def initialize_from_raw_policy(model, state, mean, std):
    """Keep the raw-input actor unchanged when adopting new dataset statistics."""
    state = {key: value.detach().clone() for key, value in state.items()}
    weight = state['trunk.0.weight']
    state['trunk.0.bias'] = state['trunk.0.bias'] + weight @ mean
    state['trunk.0.weight'] = weight * std.unsqueeze(0)
    model.load_state_dict(state)


def epoch(model, x, accel, lane, weights, batch_size, optimizer=None, *, accel_weights=None):
    train=optimizer is not None
    model.train(train)
    order=torch.randperm(len(x),device=x.device) if train else torch.arange(len(x),device=x.device)
    squared=x.new_zeros(())
    ce=x.new_zeros(())
    denominator=x.new_zeros(())
    confusion=torch.zeros((3,3),dtype=torch.int64,device=x.device)
    with torch.set_grad_enabled(train):
        for idx in order.split(batch_size):
            mu,_,logits,_=model(x[idx])
            errors=(mu.squeeze(-1)-accel[idx]).square()
            losses=F.cross_entropy(logits,lane[idx],weight=weights,reduction='none')
            accel_loss = errors.mean()
            if accel_weights is not None:
                batch_weights = accel_weights[idx]
                accel_loss = (errors * batch_weights).sum() / batch_weights.sum()
            loss=accel_loss+losses.sum()/weights[lane[idx]].sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite BC loss')
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
            squared+=errors.detach().sum()
            ce+=losses.detach().sum()
            denominator+=weights[lane[idx]].sum()
            pred=logits.detach().argmax(-1)
            confusion+=torch.bincount(lane[idx]*3+pred,minlength=9).reshape(3,3)
    cm=confusion.cpu().numpy()
    precision=np.diag(cm)/np.maximum(cm.sum(axis=0),1)
    recall=np.diag(cm)/np.maximum(cm.sum(axis=1),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    mse=float(squared)/len(x)
    lane_loss=float(ce/denominator)
    return dict(loss=mse+lane_loss,accel_mse=mse,lane_ce=lane_loss,
                lane_macro_f1=float(f1.mean()),left_precision=float(precision[0]),
                left_recall=float(recall[0]),right_precision=float(precision[2]),
                right_recall=float(recall[2]),confusion=cm.tolist())



def load(path):
    with np.load(path,allow_pickle=False) as d:
        return d['observation'],d['action_raw']


def run_pass(model, paths, mean, std, weights, batch_size, optimizer=None, deadline=None,
             *, yellow_index=None, yellow_weight=1.0):
    total_n=0; squared=0.;ce=0.;denom=0.;conf=np.zeros((3,3),np.int64)
    completed=True
    for path in paths:
        if deadline is not None and time.monotonic()>=deadline:
            completed=False;break
        obs,act=load(path);labels=lane_class(act)
        x=(torch.as_tensor(obs,device=mean.device)-mean)/std
        y=torch.as_tensor(act[:,0],device=mean.device)
        z=torch.as_tensor(labels,device=mean.device)
        accel_weights = None
        if yellow_index is not None:
            yellow = torch.as_tensor(obs[:, yellow_index], device=mean.device)
            accel_weights = 1 + yellow * (yellow_weight - 1)
        result=epoch(model,x,y,z,weights,batch_size,optimizer,accel_weights=accel_weights)
        n=len(obs);d=float(weights[z].sum().item())
        total_n+=n;squared+=result['accel_mse']*n;ce+=result['lane_ce']*d;denom+=d
        conf+=np.asarray(result['confusion'])
    if not total_n:
        return None
    diag=conf.diagonal();precision=diag/np.maximum(conf.sum(0),1);recall=diag/np.maximum(conf.sum(1),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return dict(loss=squared/total_n+ce/denom,accel_mse=squared/total_n,lane_ce=ce/denom,
                lane_macro_f1=float(f1.mean()),left_precision=float(precision[0]),left_recall=float(recall[0]),
                right_precision=float(precision[2]),right_recall=float(recall[2]),confusion=conf.tolist(),
                samples=total_n,complete_pass=completed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_dir',type=Path);p.add_argument('--out-dir', type=Path, help='Default: results/BC_YYYYMMDD_HHMMSS')
    p.add_argument('--wall-seconds',type=float,default=21600);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=8192);p.add_argument('--lr',type=float,default=4.2e-4)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda');p.add_argument('--patience',type=int,default=100)
    p.add_argument('--hidden-sizes', nargs='+', type=int, default=HPARAMS['hidden_sizes'])
    p.add_argument('--init-model', type=Path)
    p.add_argument('--lane-weight-root', type=int, default=2,
                   help='Root of inverse lane class frequency (default: 2)')
    p.add_argument('--yellow-accel-weight-root', type=int,
                   help='Apply inverse-frequency weighting to acceleration loss on yellow signal steps')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.lane_weight_root < 1:
        p.error('--lane-weight-root must be at least 1')
    manifest=json.loads((a.data_dir/'manifest.json').read_text());collection=json.loads((a.data_dir/'config.json').read_text())
    yellow_index = None
    yellow_count = 0
    yellow_weight = 1.0
    if a.yellow_accel_weight_root is not None:
        observation = collection['observation']
        if (a.yellow_accel_weight_root < 1 or observation.get('layout') != 'bc49'
                or not observation.get('signal_enabled')):
            p.error('--yellow-accel-weight-root requires a positive root and the BC signal layout')
        # Speed, five lanes of neighbors, five connectivity flags, five densities, then red and yellow.
        yellow_index = 1 + 5 * 4 + 5 + 5 + 1
    paths=[a.data_dir/x for x in manifest['shards']];rng=np.random.default_rng(a.seed)
    permutation=rng.permutation(len(paths));nval=max(1,round(.05*len(paths)))
    val_indices=set(manifest.get("validation_indices", map(int, permutation[:nval])));train=[x for i,x in enumerate(paths) if i not in val_indices];val=[x for i,x in enumerate(paths) if i in val_indices]
    if len(train)==0:raise ValueError('Need independent train/validation episodes')
    # Merge per-shard moments without retaining all observations in RAM/GPU.
    count=0;mean=np.zeros(manifest['obs_dim'],np.float64);m2=np.zeros_like(mean);counts=np.zeros(3,np.int64)
    for path in train:
        obs,act=load(path);n=len(obs);local=obs.mean(0,dtype=np.float64);delta=local-mean
        m2+=np.square(obs-local).sum(0)+delta**2*count*n/(count+n)
        mean+=delta*n/(count+n);count+=n;counts+=np.bincount(lane_class(act),minlength=3)
        if yellow_index is not None:
            yellow = obs[:, yellow_index]
            if not np.isin(yellow, [0, 1]).all():
                raise ValueError('Expected a binary yellow observation')
            yellow_count += int(yellow.sum())
    if (counts==0).any():raise ValueError(f'Missing lane class: {counts}')
    if yellow_index is not None:
        if not 0 < yellow_count < count:
            raise ValueError('Weighting requires both yellow and non-yellow steps')
        yellow_weight = ((count - yellow_count) / yellow_count) ** (1 / a.yellow_accel_weight_root)
        print(f'Yellow acceleration weight: {yellow_weight:.6f} ({yellow_count:,}/{count:,} training steps)', flush=True)
    std=np.sqrt(np.maximum(m2/count,0));std=np.where(std<1e-6,1,std)
    mean=mean.astype(np.float32);std=std.astype(np.float32)
    w = np.power(count / (3 * counts), 1 / a.lane_weight_root)
    w /= w.mean()
    a.out_dir = create_bc_result_dir(a.out_dir)
    device=torch.device(a.device);mt=torch.tensor(mean,device=device);st=torch.tensor(std,device=device);wt=torch.tensor(w,dtype=torch.float32,device=device)
    model=HybridActorCritic(manifest['obs_dim'],hidden_sizes=a.hidden_sizes,activation=HPARAMS['activation'],init_log_std=HPARAMS['init_log_std'],lane_keep_bias=HPARAMS['lane_keep_bias']).to(device)
    if a.init_model:
        state = torch.load(a.init_model, map_location=device, weights_only=True)
        initialize_from_raw_policy(model, state, mt, st)
    else:
        with torch.no_grad():model.v_head.weight.zero_();model.v_head.bias.zero_()
    for param in model.v_head.parameters():param.requires_grad_(False)
    model.log_std.requires_grad_(False)
    optimizer=torch.optim.Adam([x for x in model.parameters() if x.requires_grad],lr=a.lr)
    config=dict(data=str(a.data_dir/'manifest.json'),dataset_directory=str(a.data_dir.resolve()),init_model=str(a.init_model) if a.init_model else None,
                algorithm='supervised_bc_on_ppo_hybrid_actor',observation_dim=manifest['obs_dim'],hidden_sizes=a.hidden_sizes,activation=HPARAMS['activation'],seed=a.seed,
                observation_neighbor_distance_mode=collection['observation'].get('neighbor_distance_mode','legacy'),
                collision_detection=collection.get('collision_detection'),
                epochs=a.epochs,wall_seconds=a.wall_seconds,batch_size=a.batch_size,lr=a.lr,train_samples=count,
                validation_samples=manifest['samples']-count,train_episodes=[i for i in range(len(paths)) if i not in val_indices],validation_episodes=sorted(val_indices),
                lane_class_order=['left','keep','right'],train_lane_counts=counts.tolist(),lane_weights=w.tolist(),
                lane_weight_root=a.lane_weight_root,
                yellow_accel_weight_root=a.yellow_accel_weight_root, yellow_accel_weight=yellow_weight,
                train_yellow_count=yellow_count if yellow_index is not None else None,
                loss='accel_mse + inverse_frequency_lane_ce',storage='episode_npz_shards',
                collection_manifest_sha256=hashlib.sha256((a.data_dir/'manifest.json').read_bytes()).hexdigest())
    (a.out_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n');np.savez(a.out_dir/'normalization.npz',mean=mean,std=std)
    for source in ['env/road_config.py','env/mdp_config.py','train.py','scripts/train_bc_shards.py','env/ring_observation.py']:
        shutil.copy2(source,a.out_dir/Path(source).name)
    with (a.out_dir/'mdp_config.py').open('a') as snapshot:
        snapshot.write('\nOBSERVATION = '+repr(collection['observation'])+'\nSIMULATION = '+repr(collection['simulation'])+'\n')
    history=[];best=float('inf');best_epoch=0;start=time.monotonic();deadline=start+a.wall_seconds
    fields=['epoch','train_loss','val_loss','val_accel_mse','val_lane_ce','val_lane_macro_f1','val_left_precision','val_left_recall','val_right_precision','val_right_recall']
    with SummaryWriter(str(a.out_dir/'tb')) as writer,(a.out_dir/'training_log.csv').open('w') as f:
        csvwriter=csv.DictWriter(f,fieldnames=fields);csvwriter.writeheader()
        for e in range(1,a.epochs+1):
            shuffled=[train[i] for i in rng.permutation(len(train))]
            tr=run_pass(model,shuffled,mt,st,wt,a.batch_size,optimizer,deadline,
                        yellow_index=yellow_index,yellow_weight=yellow_weight)
            if tr is None:break
            va=run_pass(model,val,mt,st,wt,a.batch_size)
            row=dict(epoch=e,train_loss=tr['loss'],val_loss=va['loss'],**{'val_'+k:va[k] for k in ['accel_mse','lane_ce','lane_macro_f1','left_precision','left_recall','right_precision','right_recall']})
            csvwriter.writerow(row);f.flush()
            for k,v in row.items():
                if k!='epoch':writer.add_scalar(k,v,e)
            history.append(dict(epoch=e,train=tr,validation=va))
            exported=export_policy(model,mt,st)
            torch.save(exported,a.out_dir/'last_model.pt')
            if va['loss']<best:
                best=va['loss'];best_epoch=e;torch.save(exported,a.out_dir/'model.pt')
                torch.save(dict(state_dict=model.state_dict(),obs_mean=mt,obs_std=st,epoch=e),a.out_dir/'bc_training_state.pt')
            result=dict(status='complete' if e==a.epochs or time.monotonic()>=deadline or e-best_epoch>=a.patience else 'training',best_epoch=best_epoch,best_val_loss=best,epochs_completed=e,wall_seconds=time.monotonic()-start,best_validation=history[best_epoch-1]['validation'],init_model=str(a.init_model) if a.init_model else None)
            (a.out_dir/'metrics.json').write_text(json.dumps(history,indent=2)+'\n');(a.out_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'Epoch {e}: train={tr["loss"]:.6f} val={va["loss"]:.6f} complete={tr["complete_pass"]} seconds={time.monotonic()-start:.1f}',flush=True)
            if time.monotonic()>=deadline or e-best_epoch>=a.patience:break
    # Export parity is verified on an independent validation shard.
    saved=torch.load(a.out_dir/'bc_training_state.pt',map_location=device,weights_only=True);model.load_state_dict(saved['state_dict']);model.eval()
    raw=torch.tensor(load(val[0])[0][:1024],device=device);raw_model=deepcopy(model);raw_model.load_state_dict(torch.load(a.out_dir/'model.pt',map_location=device,weights_only=True));raw_model.eval()
    with torch.no_grad():torch.testing.assert_close(model.det_action((raw-mt)/st),raw_model.det_action(raw),rtol=1e-4,atol=1e-5)
    result['status']='complete';result['raw_input_export_verified']=True
    (a.out_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
