"""R153 fixed single-factor candidate WAN features; local durable execution."""
import argparse
import concurrent.futures as cf
import copy
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import statistics as st
import time
import traceback
import r152_stopping_ppo_formal as F

S=F.S;torch=F.torch
ROOT=Path(__file__).resolve().parent/'outputs/r153_candidate_wan'
VARIANTS=('base67','wan69')


def profiles():
    out=[b for b in F.profiles() if b['split']!='confirmation']
    ts=[[(0,1),(1,2),(2,3)],[(0,1),(0,2),(1,3),(2,3)],[(0,1),(0,2),(0,3)]]
    for i in range(24):
        rng=random.Random(1531000+i)
        p=dict(id=153000+i,family=('chain','diamond','tree')[i%3],work=[rng.uniform(.015,.16) for _ in range(4)],
            mem=[rng.choice([32,64,96]) for _ in range(4)],edges=[(u,v,rng.choice([.01,.1,1.])) for u,v in ts[i%3]],
            ram=[256,2048],cores=[1,3],speed=[1.,rng.uniform(1.5,3.)],mbps=rng.choice([10.,50.,100.]),rtt=rng.choice([.01,.05,.08]))
        old=random.Random(1532000+i).sample(S.B.layouts(p),4)
        out.append(dict(profile=p,initials=old,split='confirmation'))
    return out


def wan_features(e,pool):
    past=[t-60. for t in S.B.flow(152300000+e.p['id'])]
    known=past+[t for t in e.arrivals if t<=e.now]
    rate=sum(e.now-60.<t<=e.now for t in known)/60.
    out=[]
    for m,z in pool:
        data=[0.,0.]
        for u,v,b in e.p['edges']:
            a,c=z[m[u]],z[m[v]]
            if a!=c:data[a]+=b
        out.append([math.asinh(rate*v*8/e.p['mbps']) for v in data])
    return out


def observe(e,pool,step,variant):
    xs=F.observe(e,pool,step)
    return xs if variant=='base67' else [x+w for x,w in zip(xs,wan_features(e,pool))]


def make_net(seed,variant):
    torch.manual_seed(seed);net=S.Policy(67)
    if variant=='wan69':
        old=net.encoder[0];new=torch.nn.Linear(69,32)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:67].copy_(old.weight);new.bias.copy_(old.bias)
        net.encoder[0]=new
    return net


def rollout(p,old,arr,net,variant,stochastic=False):
    e=S.Engine(p,old,arr);ds=[]
    for step in range(2):
        pool=[e.layout]+[l for l in S.neighbors(p,e.layout)[1:] if l not in e.visited]
        x=observe(e,pool,step,variant)
        with torch.no_grad():
            lp,v,ent=net(torch.tensor(x,dtype=torch.float32))
            idx=int(torch.distributions.Categorical(logits=lp).sample()) if stochastic else int(lp.argmax())
        ds.append(dict(x=x,index=idx,logp=float(lp[idx]),value=float(v),reward=0.,forced=False,
                       time=e.now,options=len(pool)))
        if idx==0:break
        e.change(pool[idx])
        if step==0:
            e.advance(max(15.,e.now))
            if e.now>=60:break
    r=e.finish();ds[-1]['reward']=-r['mean']/100.
    r.update(decisions=len(ds),indices=[d['index'] for d in ds],decision_times=[d['time'] for d in ds])
    return r,ds


def evaluate(net,variant,ss,prefix,n,attribution=False):
    rows=[]
    for p,old,j in ss:
        for stream in range(n):
            streamseed=prefix+p['id']*7+stream;arr=S.B.flow(streamseed)
            r,ds=rollout(p,old,arr,net,variant)
            rr=dict(profile=p['id'],initial_index=j,stream=stream,stream_seed=streamseed,**r)
            if attribution:
                e=S.Engine(p,old,arr)
                if r['actions']:e.change(tuple(map(tuple,r['actions'][0]['new'])))
                rr.update(keep_mean=S.Engine(p,old,arr).finish()['mean'],first_only_mean=e.finish()['mean'])
            rows.append(rr)
    return rows


def train(job):
    seed,variant=job;torch.set_num_threads(1);folder=ROOT/f'{variant}_{seed}';folder.mkdir(exist_ok=True)
    if (folder/'confirmation.json').exists():return
    ps=json.loads((ROOT/'manifest.json').read_text())['profiles'];ss=F.states(ps,'train');vs=F.states(ps,'validation')
    net=make_net(seed,variant);opt=torch.optim.Adam(net.parameters(),lr=.0003)
    first=0;elapsed=0.;steps=0;raw=[];valid=[];binding=F.sha(ROOT/'manifest.json');resume=folder/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['manifest_sha256']==binding
        net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer'])
        first=ck['episode'];steps=ck['steps'];elapsed=ck['compute_seconds'];raw=ck['training'];valid=ck['validation']
    tick=time.monotonic()
    for start in range(first,4096,8):
        batch=[]
        for ep in range(start,start+8):
            p,old,j=ss[random.Random(152400000+seed*10000+ep).randrange(len(ss))]
            torch.manual_seed(152500000+seed*10000+ep)
            r,ds=rollout(p,old,S.B.flow(152600000+seed*10000+ep),net,variant,True)
            batch.append(ds);steps+=len(ds);raw.append(dict(episode=ep,profile=p['id'],initial_index=j,**r))
        loss=S.update(net,opt,batch)
        if start+8 in F.CHECKPOINTS:
            vr=evaluate(net,variant,vs,152700000,2)
            val=dict(episode=start+8,mean=st.mean(r['mean'] for r in vr),p95=st.mean(r['p95'] for r in vr))
            valid.append(val);F.write(folder/f'validation_{start+8}.json',vr)
            ck=dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),episode=start+8,steps=steps,
                    compute_seconds=elapsed+time.monotonic()-tick,seed=seed,variant=variant,manifest_sha256=binding,
                    training=raw,validation=valid)
            F.save(folder/f'checkpoint_{start+8}.pt',ck);F.save(resume,ck)
            F.write(folder/'status.json',dict(phase='training',episode=start+8,total=4096,steps=steps,validation=valid,loss=loss))
            print('R153',variant,seed,'EP',start+8,'/4096 VALID',round(val['mean'],4),flush=True)
    chosen=min(valid,key=lambda v:(v['mean'],v['episode']));F.write(folder/'selection.json',chosen)
    ck=torch.load(folder/f"checkpoint_{chosen['episode']}.pt",map_location='cpu',weights_only=False)
    net.load_state_dict(ck['state_dict']);F.save(folder/'selected.pt',ck)
    rr=evaluate(net,variant,F.states(ps,'confirmation'),153800000,3,True)
    for r in rr:r.update(seed=seed,variant=variant,selected_episode=chosen['episode'])
    F.write(folder/'confirmation.json',rr)
    F.write(folder/'status.json',dict(phase='completed',episode=4096,steps=steps,selected=chosen,
                                    compute_seconds=elapsed+time.monotonic()-tick,records=len(rr)))
    print('R153 MODEL_DONE',variant,seed,flush=True)


def summarize():
    rows={v:[r for seed in F.SEEDS for r in json.loads((ROOT/f'{v}_{seed}'/'confirmation.json').read_text())] for v in VARIANTS}
    methods={}
    for v,rr in rows.items():
        methods[v]={k:st.mean(r[k] for r in rr) for k in ('mean','p95','changes','deployment','drain','keep_mean','first_only_mean')}
        methods[v].update(records=len(rr),initial_stop=sum(r['indices'][0]==0 for r in rr),
            second_stop=sum(len(r['indices'])==2 and r['indices'][1]==0 for r in rr),
            over10pct_keep=sum(r['mean']>1.1*r['keep_mean'] for r in rr),
            over5s_keep=sum(r['mean']>r['keep_mean']+5 for r in rr),
            max_extra_s_keep=max(r['mean']-r['keep_mean'] for r in rr),
            serious_loss_over_gross_gain=sum(r['mean']-r['keep_mean'] for r in rr if r['mean']>r['keep_mean']+5)/max(1e-12,sum(max(0,r['keep_mean']-r['mean']) for r in rr)))
    seeds=[dict(seed=s,**{v:st.mean(r['mean'] for r in rr if r['seed']==s) for v,rr in rows.items()}) for s in F.SEEDS]
    pids=sorted({r['profile'] for r in rows['base67']});pairs=[tuple(st.mean(r['mean'] for r in rows[v] if r['profile']==p) for v in VARIANTS) for p in pids]
    rng=random.Random(1539000);boot=[]
    for _ in range(5000):
        sample=rng.choices(pairs,k=len(pairs));boot.append(100*(1-sum(b for a,b in sample)/sum(a for a,b in sample)))
    boot.sort();imp=100*(1-methods['wan69']['mean']/methods['base67']['mean']);wins=sum(r['wan69']<r['base67'] for r in seeds)
    summary=dict(status='completed_pending_user_analysis',methods=methods,seeds=seeds,improvement_pct=imp,
        config_bootstrap_ci95=[boot[124],boot[4874]],seed_wins=wins,
        performance_gate=imp>=5 and boot[124]>0 and wins>=4 and methods['wan69']['p95']<=methods['base67']['p95'],
        distinct_conditions=288,statistical_config_groups=24)
    F.write(ROOT/'summary.json',summary);print('R153_ALL_FINISHED',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);a=parser.parse_args()
    torch.set_num_threads(1);ROOT.mkdir(parents=True,exist_ok=True)
    try:
        bindings={p.name:F.sha(p) for p in (Path(__file__),Path(F.__file__),Path(S.__file__),Path(S.B.__file__),Path(__file__).with_name('R153_PROTOCOL.md'))}
        manifest=json.loads(json.dumps(dict(profiles=profiles(),seeds=F.SEEDS,variants=VARIANTS,episodes=4096,
             checkpoints=F.CHECKPOINTS,feature_dims=[67,69],bindings=bindings,workers=a.workers,torch=torch.__version__)))
        mpath=ROOT/'manifest.json'
        if mpath.exists():assert json.loads(mpath.read_text())==manifest
        else:F.write(mpath,manifest)
        p,old,j=F.states(manifest['profiles'],'train')[0];e=S.Engine(p,old,[1.,2.,20.]);q=S.Engine(p,old,[1.,2.,30.,31.])
        e.advance(10.);q.advance(10.);pool=S.neighbors(p,old)
        assert observe(e,pool,1,'wan69')==observe(q,pool,1,'wan69')
        base=make_net(17,'base67');wan=make_net(17,'wan69');x=torch.tensor(observe(e,pool,1,'base67'));y=torch.tensor(observe(e,pool,1,'wan69'))
        with torch.no_grad():
            for u,v in zip(base(x),wan(y)):assert torch.allclose(u,v,atol=1e-6,rtol=1e-6)
        # All-cloud placement must have zero business WAN demand.
        allcloud=next(l for l in S.B.layouts(p) if all(z==1 for z in l[1]))
        assert wan_features(e,[allcloud])==[[0.,0.]]
        F.write(ROOT/'preflight.json',dict(status='passed',same_initial_outputs=True,no_future_features=True,
                                          all_cloud_zero_wan=True,extra_parameters=64))
        F.write(ROOT/'status.json',dict(phase='running',jobs=10))
        print('R153 PREFLIGHT PASSED; 10 JOBS; CHECKPOINT EVERY 512 EPISODES',flush=True)
        with cf.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as ex:
            list(ex.map(train,[(s,v) for s in F.SEEDS for v in VARIANTS]))
        summarize();F.write(ROOT/'status.json',dict(phase='completed_pending_user_analysis'))
    except Exception:
        F.write(ROOT/'status.json',dict(phase='failed',error=traceback.format_exc()))
        print('R153_FAILED',flush=True);raise


if __name__=='__main__':main()
