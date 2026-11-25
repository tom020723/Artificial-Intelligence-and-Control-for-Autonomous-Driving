"""Supervised BC on the exact PPO hybrid actor; export a raw-observation PPO checkpoint.

Acceleration MSE + sqrt-inverse-frequency weighted lane cross entropy.
The critic and acceleration exploration std are not trained from demonstrations.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train_bc import load_data, split_episodes
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--init-model',type=Path,help='Continue BC from a raw-input PPO-compatible checkpoint')
    parser.add_argument('--out-dir', type=Path, help='Default: results/BC_YYYYMMDD_HHMMSS')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=8192)
    parser.add_argument('--lr',type=float,default=4.2e-4)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.patience<1 or not np.isfinite(args.lr) or args.lr<=0:
        parser.error('epochs/batch-size/patience/lr must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs,actions,episodes=load_data(args.data)
    labels=lane_class(actions)
    train,val=split_episodes(episodes,.2,args.seed)
    mean=obs[train].mean(axis=0)
    std=obs[train].std(axis=0)
    std=np.where(std<1e-6,1,std)
    counts=np.bincount(labels[train],minlength=3)
    if (counts==0).any():
        raise ValueError('Training requires demonstrations of all lane classes')
    weights=np.sqrt(len(train)/(3*counts))
    weights=weights/weights.mean()
    device=torch.device(args.device)
    model=HybridActorCritic(obs.shape[1],hidden_sizes=HPARAMS['hidden_sizes'],
                           activation=HPARAMS['activation'],init_log_std=HPARAMS['init_log_std'],
                           lane_keep_bias=HPARAMS['lane_keep_bias']).to(device)
    if args.init_model is not None:
        initialize_from_raw_policy(model,
            torch.load(args.init_model,map_location=device,weights_only=True),
            torch.as_tensor(mean,device=device),torch.as_tensor(std,device=device))
    # There are no value targets in BC. PPO will train its critic during fine-tuning.
    with torch.no_grad():
        model.v_head.weight.zero_()
        model.v_head.bias.zero_()
    for param in model.v_head.parameters():param.requires_grad_(False)
    model.log_std.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    mean_tensor=torch.tensor(mean,device=device)
    std_tensor=torch.tensor(std,device=device)
    x=torch.as_tensor((obs-mean)/std,device=device)
    accel=torch.as_tensor(actions[:,0],device=device)
    lane=torch.as_tensor(labels,device=device)
    train_data=(x[train],accel[train],lane[train])
    val_data=(x[val],accel[val],lane[val])
    weight_tensor=torch.as_tensor(weights,dtype=torch.float32,device=device)
    args.out_dir = create_bc_result_dir(args.out_dir)
    config={**vars(args),'data':str(args.data.resolve()),'out_dir':str(args.out_dir.resolve()),
            'init_model':str(args.init_model.resolve()) if args.init_model else None,
            'init_model_sha256':hashlib.sha256(args.init_model.read_bytes()).hexdigest() if args.init_model else None,
            'algorithm':'supervised_bc_on_ppo_hybrid_actor','observation_dim':obs.shape[1],
            'hidden_sizes':HPARAMS['hidden_sizes'],'activation':HPARAMS['activation'],
            'train_samples':len(train),'validation_samples':len(val),
            'train_episodes':np.unique(episodes[train]).tolist(),
            'validation_episodes':np.unique(episodes[val]).tolist(),
            'lane_class_order':['left','keep','right'],'train_lane_counts':counts.tolist(),
            'lane_weights':weights.tolist(),'loss':'accel_mse + sqrt_inverse_frequency_lane_ce',
            'checkpoint':'PPO state_dict; normalization folded into first layer; critic zero, std unchanged',
            'dataset_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest()}
    (args.out_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    np.savez(args.out_dir/'normalization.npz',mean=mean,std=std)
    for source in ['env/road_config.py','env/mdp_config.py','train.py','train_bc_actor.py']:
        shutil.copy2(source,args.out_dir/Path(source).name)
    print(json.dumps(config,indent=2),flush=True)
    best=float('inf'); best_epoch=0; history=[]; start=time.monotonic()
    writer=SummaryWriter(str(args.out_dir/'tb'))
    try:
        with (args.out_dir/'training_log.csv').open('w',newline='') as stream:
            fields=['epoch','train_loss','val_loss','val_accel_mse','val_lane_ce','val_lane_macro_f1',
                    'val_left_precision','val_left_recall','val_right_precision','val_right_recall']
            csv_writer=csv.DictWriter(stream,fieldnames=fields);csv_writer.writeheader()
            for i in range(1,args.epochs+1):
                training=epoch(model,*train_data,weight_tensor,args.batch_size,optimizer)
                validation=epoch(model,*val_data,weight_tensor,args.batch_size)
                row=dict(epoch=i,train_loss=training['loss'],val_loss=validation['loss'],
                         **{f'val_{key}':value for key,value in validation.items() if f'val_{key}' in fields and key!='loss'})
                csv_writer.writerow(row);stream.flush()
                for key,value in row.items():
                    if key!='epoch':writer.add_scalar(key,value,i)
                writer.flush()
                history.append(dict(epoch=i,train=training,validation=validation))
                (args.out_dir/'metrics.json').write_text(json.dumps(history,indent=2)+'\n')
                if validation['loss']<best:
                    best=validation['loss'];best_epoch=i
                    torch.save(export_policy(model,mean_tensor,std_tensor),args.out_dir/'model.pt')
                    torch.save(dict(state_dict=model.state_dict(),obs_mean=mean_tensor,obs_std=std_tensor,
                                    optimizer=optimizer.state_dict(),epoch=i),args.out_dir/'bc_training_state.pt')
                torch.save(export_policy(model,mean_tensor,std_tensor),args.out_dir/'last_model.pt')
                print(f'Epoch {i}/{args.epochs}: train={training["loss"]:.5f} val={validation["loss"]:.5f} '
                      f'LC macro F1={validation["lane_macro_f1"]:.3f} '
                      f'L recall={validation["left_recall"]:.3f} R recall={validation["right_recall"]:.3f}',flush=True)
                if i-best_epoch>=args.patience:
                    print('Early stopping',flush=True);break
    finally:
        writer.close()
    # Verify that exported raw-input PPO reproduces normalized BC predictions.
    saved=torch.load(args.out_dir/'bc_training_state.pt',map_location=device,weights_only=True)
    model.load_state_dict(saved['state_dict']);model.eval()
    exported=deepcopy(model)
    exported.load_state_dict(torch.load(args.out_dir/'model.pt',map_location=device,weights_only=True))
    with torch.no_grad():
        raw=torch.as_tensor(obs[val[:1024]],device=device)
        expected=model.det_action((raw-mean_tensor)/std_tensor)
        actual=exported.det_action(raw)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
    result=dict(status='complete',best_epoch=best_epoch,best_val_loss=best,
                epochs_completed=len(history),wall_seconds=time.monotonic()-start,
                best_validation=history[best_epoch-1]['validation'],raw_input_export_verified=True,
                driving_evaluation_performed=False)
    (args.out_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
