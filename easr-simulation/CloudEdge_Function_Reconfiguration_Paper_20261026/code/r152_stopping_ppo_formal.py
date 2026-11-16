"""R152 local matched-rollout formal experiment; immutable protocol and resume."""
import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import statistics as st
import time

import torch
import r151_stopping_ppo_smoke as S

ROOT=Path(__file__).resolve().parent/'outputs/r152_stopping_ppo_formal'
SEEDS=(17,29,43,59,71)
ARMS=('single','stop','forced')
CHECKPOINTS=(512,1024,2048,4096)


def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def save(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def profiles():
    out=[];ts=[[(0,1),(1,2),(2,3)],[(0,1),(0,2),(1,3),(2,3)],[(0,1),(0,2),(0,3)]]
    for i in range(84):
        rng=random.Random(1521000+i)
        p=dict(id=152000+i,family=('chain','diamond','tree')[i%3],work=[rng.uniform(.015,.16) for _ in range(4)],
            mem=[rng.choice([32,64,96]) for _ in range(4)],edges=[(u,v,rng.choice([.01,.1,1.])) for u,v in ts[i%3]],
            ram=[256,2048],cores=[1,3],speed=[1.,rng.uniform(1.5,3.)],mbps=rng.choice([10.,50.,100.]),rtt=rng.choice([.01,.05,.08]))
        initials=random.Random(1522000+i).sample(S.B.layouts(p),4)
        out.append(dict(profile=p,initials=initials,split='train' if i<48 else 'validation' if i<60 else 'confirmation'))
    return out


def states(ps,split):
    return [(b['profile'],tuple(map(tuple,l)),j) for b in ps if b['split']==split for j,l in enumerate(b['initials'])]


def observe(e,pool,step):
    rows=S.features(e,pool,step)
    past=[t-60. for t in S.B.flow(152300000+e.p['id'])]
    known=past+[t for t in e.arrivals if t<=e.now]
    rate=sum(e.now-60.<t<=e.now for t in known)/60.
    recent=sum(e.now-10.<t<=e.now for t in known)/10.
    return [r+[math.asinh(rate),math.asinh(recent)] for r in rows]


def rollout(p,old,arr,net,arm,stochastic=False):
    e=S.Engine(p,old,arr);ds=[];limit=1 if arm=='single' else 2
    for step in range(limit):
        pool=[e.layout]+[v for v in S.neighbors(p,e.layout)[1:] if v not in e.visited]
        x=observe(e,pool,step);forced=arm=='forced'
        with torch.no_grad():
            lp,v,ent=net(torch.tensor(x,dtype=torch.float32),forced)
            idx=int(torch.distributions.Categorical(logits=lp).sample()) if stochastic else int(lp.argmax())
        ds.append(dict(x=x,index=idx,logp=float(lp[idx]),value=float(v),reward=0.,forced=forced,
                       time=e.now,options=len(pool)))
        if idx==0:break
        e.change(pool[idx])
        if step+1<limit:
            e.advance(max(15.,e.now))
            if e.now>=60.:break
    result=e.finish();ds[-1]['reward']=-result['mean']/100.
    result.update(decisions=len(ds),indices=[d['index'] for d in ds],decision_times=[d['time'] for d in ds])
    return result,ds


def evaluate(net,arm,ss,prefix,n,attribution=False):
    rows=[]
    for p,old,j in ss:
        for stream in range(n):
            streamseed=prefix+p['id']*7+stream;arr=S.B.flow(streamseed)
            r,ds=rollout(p,old,arr,net,arm)
            rr=dict(profile=p['id'],initial_index=j,stream=stream,stream_seed=streamseed,**r)
            if attribution:
                keep=S.Engine(p,old,arr).finish();e=S.Engine(p,old,arr)
                if r['actions']:e.change(tuple(map(tuple,r['actions'][0]['new'])))
                first=e.finish();rr.update(keep_mean=keep['mean'],first_only_mean=first['mean'])
            rows.append(rr)
    return rows


def train(job):
    seed,arm=job;torch.set_num_threads(1);folder=ROOT/f'{arm}_{seed}';folder.mkdir(exist_ok=True)
    if (folder/'confirmation.json').exists():return
    ps=json.loads((ROOT/'manifest.json').read_text())['profiles'];ss=states(ps,'train');vs=states(ps,'validation')
    torch.manual_seed(seed);net=S.Policy(67);opt=torch.optim.Adam(net.parameters(),lr=.0003)
    first=0;elapsed=0.;steps=0;raw=[];valid=[];binding=sha(ROOT/'manifest.json');resume=folder/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['manifest_sha256']==binding
        net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer'])
        first=ck['episode'];steps=ck['steps'];elapsed=ck['compute_seconds'];raw=ck['training'];valid=ck['validation']
    t0=time.monotonic()
    for start in range(first,4096,8):
        batch=[]
        for ep in range(start,start+8):
            p,old,j=ss[random.Random(152400000+seed*10000+ep).randrange(len(ss))]
            torch.manual_seed(152500000+seed*10000+ep)
            r,ds=rollout(p,old,S.B.flow(152600000+seed*10000+ep),net,arm,True)
            batch.append(ds);steps+=len(ds)
            raw.append(dict(episode=ep,profile=p['id'],initial_index=j,**r))
        loss=S.update(net,opt,batch)
        if start+8 in CHECKPOINTS:
            vr=evaluate(net,arm,vs,152700000,2)
            val=dict(episode=start+8,mean=st.mean(r['mean'] for r in vr),p95=st.mean(r['p95'] for r in vr))
            valid.append(val);write(folder/f'validation_{start+8}.json',vr)
            ck=dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),episode=start+8,steps=steps,
                    compute_seconds=elapsed+time.monotonic()-t0,seed=seed,arm=arm,manifest_sha256=binding,
                    training=raw,validation=valid)
            save(folder/f'checkpoint_{start+8}.pt',ck);save(resume,ck)
            write(folder/'status.json',dict(phase='training',episode=start+8,total=4096,steps=steps,validation=valid,loss=loss))
            print('R152',arm,seed,'EP',start+8,'VALID',round(val['mean'],4),flush=True)
    # Validation selection is durable before opening confirmation flows.
    chosen=min(valid,key=lambda v:(v['mean'],v['episode']));write(folder/'selection.json',chosen)
    ck=torch.load(folder/f"checkpoint_{chosen['episode']}.pt",map_location='cpu',weights_only=False)
    net.load_state_dict(ck['state_dict']);save(folder/'selected.pt',ck)
    rr=evaluate(net,arm,states(ps,'confirmation'),152800000,3,True)
    for r in rr:r.update(seed=seed,arm=arm,selected_episode=chosen['episode'])
    write(folder/'confirmation.json',rr)
    write(folder/'status.json',dict(phase='completed',episode=4096,steps=steps,selected=chosen,
                                   compute_seconds=elapsed+time.monotonic()-t0,records=len(rr)))
    print('R152 DONE',arm,seed,'MEAN',round(st.mean(r['mean'] for r in rr),4),flush=True)


def summarize():
    rows=[r for seed in SEEDS for arm in ARMS for r in json.loads((ROOT/f'{arm}_{seed}'/'confirmation.json').read_text())]
    by={a:[r for r in rows if r['arm']==a] for a in ARMS};methods={};perseed=[]
    for a,rr in by.items():
        gains=[r['keep_mean']-r['mean'] for r in rr]
        gross=sum(max(0.,v) for v in gains);serious=[r for r in rr if r['mean']>r['keep_mean']+5.]
        methods[a]={k:st.mean(r[k] for r in rr) for k in ('mean','p95','changes','deployment','drain','keep_mean','first_only_mean')}
        methods[a].update(records=len(rr),initial_stop=sum(r['indices'][0]==0 for r in rr),
            second_stop=sum(len(r['indices'])==2 and r['indices'][1]==0 for r in rr),
            second_actions=sum(r['changes']==2 for r in rr),
            second_wins=sum(r['changes']==2 and r['mean']<r['first_only_mean'] for r in rr),
            over10pct_keep=sum(r['mean']>1.1*r['keep_mean'] for r in rr),over5s_keep=len(serious),
            max_extra_s_keep=max(r['mean']-r['keep_mean'] for r in rr),
            serious_loss_over_gross_gain=sum(r['mean']-r['keep_mean'] for r in serious)/gross if gross else None)
    for seed in SEEDS:
        perseed.append(dict(seed=seed,**{a:st.mean(r['mean'] for r in by[a] if r['seed']==seed) for a in ARMS}))
    comparisons={};pids=sorted({r['profile'] for r in rows})
    for base in ('forced','single'):
        pairs=[(st.mean(r['mean'] for r in by[base] if r['profile']==p),st.mean(r['mean'] for r in by['stop'] if r['profile']==p)) for p in pids]
        rng=random.Random(1529000);boots=[]
        for _ in range(5000):
            sample=rng.choices(pairs,k=len(pairs));boots.append(100*(1-sum(b for a,b in sample)/sum(a for a,b in sample)))
        boots.sort();improve=100*(1-methods['stop']['mean']/methods[base]['mean'])
        index={(r['seed'],r['profile'],r['initial_index'],r['stream']):r for r in by[base]}
        worse=[(r,index[r['seed'],r['profile'],r['initial_index'],r['stream']]) for r in by['stop']]
        wins=sum(r['stop']<r[base] for r in perseed)
        comparisons[base]=dict(improvement_pct=improve,config_bootstrap_ci95=[boots[124],boots[4874]],seed_wins=wins,
            p95_improvement_pct=100*(1-methods['stop']['p95']/methods[base]['p95']),
            over10pct=sum(r['mean']>1.1*b['mean'] for r,b in worse),
            over5s=sum(r['mean']>b['mean']+5 for r,b in worse),
            max_extra_s=max(r['mean']-b['mean'] for r,b in worse),
            performance_gate=improve>=5 and boots[124]>0 and wins>=4 and methods['stop']['p95']<=methods[base]['p95'])
    out=dict(status='completed_pending_audit',methods=methods,seeds=perseed,comparisons=comparisons,
             independent_confirmation_conditions=24*4*3,model_seeds=5,profiles=24,
             training=[dict(arm=a,seed=s,**json.loads((ROOT/f'{a}_{s}'/'status.json').read_text())) for s in SEEDS for a in ARMS])
    write(ROOT/'summary.json',out);print(json.dumps({k:out[k] for k in ('methods','seeds','comparisons')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    torch.set_num_threads(1);ROOT.mkdir(parents=True,exist_ok=True)
    bindings={p.name:sha(p) for p in (Path(__file__),Path(S.__file__),Path(S.B.__file__),Path(__file__).with_name('R152_PROTOCOL.md'))}
    manifest=dict(profiles=profiles(),seeds=SEEDS,arms=ARMS,episodes=4096,checkpoints=CHECKPOINTS,feature_dim=67,
                  bindings=bindings,torch=torch.__version__,workers=args.workers)
    manifest=json.loads(json.dumps(manifest));mpath=ROOT/'manifest.json'
    if mpath.exists():assert json.loads(mpath.read_text())==manifest
    else:write(mpath,manifest)
    ps=manifest['profiles'];audit=S.preflight(profiles())
    p,old,j=states(ps,'train')[0];a=S.Engine(p,old,[1.,2.,20.]);b=S.Engine(p,old,[1.,2.,25.,26.])
    a.advance(10.);b.advance(10.);pool=S.neighbors(p,old);assert observe(a,pool,1)==observe(b,pool,1)
    audit.update(no_future_observation=True,feature_dim=67);write(ROOT/'preflight.json',audit)
    print('R152 PREFLIGHT',audit,flush=True);write(ROOT/'status.json',dict(phase='running'))
    with cf.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as ex:
        list(ex.map(train,[(s,a) for s in SEEDS for a in ARMS]))
    summarize();write(ROOT/'status.json',dict(phase='completed_pending_audit'))


if __name__=='__main__':main()
