"""Validate each BC49 shard without merging a multi-hour collection in memory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET


def check(directory):
    m=json.loads((directory/'manifest.json').read_text());c=json.loads((directory/'config.json').read_text())
    obs_dim = 51 if c['observation'].get('guidance_lateral_scale') is not None else 49
    assert m['obs_dim'] == obs_dim
    net=ET.parse(directory/'sumo/highway.net.xml').getroot()
    vmax=max(float(l.get('speed')) for e in net.findall('edge') if e.get('function')!='internal' for l in e.findall('lane'))
    count=0;lane_counts=np.zeros(3,np.int64);ring_count=0;approach_count=0;stopped=0;digests={}
    for name in m['shards']:
        path=directory/name
        with np.load(path,allow_pickle=False) as d:
            x=d['observation'];a=d['action_raw'];n=len(x)
            excluded=c.get('filter',{}).get('excluded_vehicle_types',[])
            if excluded:
                assert not np.isin(d['vehicle_type'],excluded).any(), 'Excluded demonstrator remains in dataset'
            assert x.shape==(n,obs_dim) and a.shape==(n,2)
            assert np.isfinite(x).all() and np.isfinite(a).all()
            assert (abs(a)<=1).all() and np.isin(a[:,1],[-1,0,1]).all()
            assert np.isin(x[:,43],[0,1]).all()
            assert (x[:,[44,46,48]]>=-1).all()
            assert (x[:,[45,47]]>=0).all()
            outside=x[:,43]==0
            assert (x[outside,43:49]==[0,-1,0,-1,0,-1]).all()
            if obs_dim == 51:
                assert (x[:, 49:51][x[:, [35, 39]] == 0] == 0).all()
            for speed,distance in [(45,46),(47,48)]:
                assert (x[x[:,distance]==-1,speed]==0).all()
            both=x[:,48]>=0
            assert (x[both,46]>=0).all() and (x[both,46]<=x[both,48]+1e-5).all()
            assert (a[~d['lane_change_allowed'],1]==0).all()
            v=d['speed'];nxt=d['next_speed'];physical=a[:,0]*np.where(a[:,0]>=0,c['ego']['accel'],c['ego']['decel'])
            assert np.max(abs(np.clip(v+physical*c['simulation']['step_length'],0,vmax)-nxt))<1e-4
            count+=n;ring_count+=int((x[:,43]==1).sum());approach_count+=int((x[:,46]>=0).sum());stopped+=int((v<.1).sum())
            lane_counts+=np.bincount((1-a[:,1]).astype(int),minlength=3)
        digests[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert count==m['samples'] and ring_count>0 and approach_count>0
    report=dict(valid=True,samples=count,shards=len(m['shards']),observation_dim=obs_dim,
                lane_counts_left_keep_right=lane_counts.tolist(),ring_samples=ring_count,
                approach_vehicle_samples=approach_count,stopped_samples=stopped,sha256=digests)
    (directory/'quality_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sha256'}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);check(p.parse_args().directory)
