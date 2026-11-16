"""CPU-only continuous two-decision smoke; see R151_PROTOCOL.md for boundaries."""
import argparse
import hashlib
import heapq
import itertools as it
import json
import math
from pathlib import Path
import random
import statistics as st
import time

import torch
import workflow_core as B

ROOT = Path(__file__).resolve().parent / 'outputs/r151_stopping_ppo_smoke'


def canonical(parts):
    m = [None] * 4
    z = []
    for g, (ns, site) in enumerate(sorted(parts, key=lambda x: min(x[0]))):
        z.append(site)
        for n in ns:
            m[n] = g
    return tuple(m), tuple(z)


def neighbors(p, old):
    gs = B.groups(old[0]); found = {old}
    for g, ns in enumerate(gs):
        other = [(v, old[1][j]) for j, v in enumerate(gs) if j != g]
        found.add(canonical(other + [(ns, 1-old[1][g])]))
        for mask in range(1, 2**len(ns)-1):
            if not mask & 1:
                continue
            a = [n for j, n in enumerate(ns) if mask >> j & 1]
            b = [n for n in ns if n not in a]
            for x, y in it.product(range(2), repeat=2):
                found.add(canonical(other + [(a, x), (b, y)]))
    return [old] + sorted(l for l in found if l != old and B.valid(p, *l))


def graph(p, l):
    m, z = l; gs = B.groups(m)
    pred = [set() for _ in gs]; succ = [set() for _ in gs]; data = {}
    for u, v, b in p['edges']:
        a, c = m[u], m[v]
        if a != c:
            pred[c].add(a); succ[a].add(c)
            data[a, c] = data.get((a, c), 0.) + b
    return gs, z, pred, succ, data


class Engine:
    def __init__(self, p, layout, arrivals):
        self.p = p; self.layout = layout; self.arrivals = arrivals
        self.now = 0.; self.serial = it.count(); self.events = []
        self.cores = [[0.] * c for c in p['cores']]; self.links = [0., 0.]
        self.jobs = {}; self.active = 0; self.backlog = []; self.responses = []
        self.task_intervals = []; self.link_intervals = []; self.admitted = []
        self.actions = []; self.deployment_total = 0.; self.drain_total = 0.
        self.visited = {layout}; self.changes = 0; self.peak_ram_fraction = 0.
        self.check_ram()
        for r, t in enumerate(arrivals):
            self.push(t, 'arrival', r, -1)

    def check_ram(self):
        used = [0., 0.]
        for g, ns in enumerate(B.groups(self.layout[0])):
            used[self.layout[1][g]] += .3*sum(self.p['mem'][n] for n in ns)+.7*max(self.p['mem'][n] for n in ns)
        assert B.valid(self.p, *self.layout)
        self.peak_ram_fraction = max(self.peak_ram_fraction, *(v/c for v, c in zip(used, self.p['ram'])))
        assert self.peak_ram_fraction <= 1. + 1e-9

    def push(self, t, kind, r, g):
        heapq.heappush(self.events, (t, next(self.serial), kind, r, g))

    def admit(self, r, t):
        gs, z, pred, succ, data = graph(self.p, self.layout)
        self.jobs[r] = dict(gs=gs, z=z, succ=succ, data=data,
                            remaining=[len(v) for v in pred], release=[0.]*len(gs),
                            sinks=sum(not v for v in succ), completed_sinks=0, last=0.)
        self.active += 1; self.admitted.append((r, t))
        for g in range(len(gs)):
            if not pred[g]:
                self.push(t, 'ready', r, g)

    def event(self, e, hold=False):
        t, _, kind, r, g = e; self.now = max(self.now, t)
        if kind == 'arrival':
            if hold:
                self.backlog.append(r)
            else:
                self.admit(r, t)
            return
        job = self.jobs[r]; z = job['z']
        if kind == 'ready':
            site = z[g]; k = min(range(len(self.cores[site])), key=lambda j: self.cores[site][j])
            begin = max(t, self.cores[site][k]); end = begin + sum(self.p['work'][n] for n in job['gs'][g])/self.p['speed'][site]
            self.cores[site][k] = end; self.task_intervals.append((site, begin, end))
            self.push(end, 'done', r, g)
        else:
            job['last'] = max(job['last'], t)
            for h in sorted(job['succ'][g]):
                ready = t
                if z[g] != z[h]:
                    d = z[g]; begin = max(t, self.links[d])
                    self.links[d] = begin + job['data'][g, h]*8/self.p['mbps']
                    self.link_intervals.append((d, begin, self.links[d]))
                    ready = self.links[d]+self.p['rtt']
                else:
                    ready += .003
                job['release'][h] = max(job['release'][h], ready)
                job['remaining'][h] -= 1
                if job['remaining'][h] == 0:
                    self.push(job['release'][h], 'ready', r, h)
            if not job['succ'][g]:
                job['completed_sinks'] += 1
                if job['completed_sinks'] == job['sinks']:
                    self.responses.append((r, job['last']-self.arrivals[r]))
                    self.active -= 1
                    del self.jobs[r]

    def advance(self, t, hold=False):
        while self.events and self.events[0][0] <= t:
            self.event(heapq.heappop(self.events), hold)
        self.now = max(self.now, t)

    def change(self, new):
        if new == self.layout:
            return
        before = self.layout; start = self.now
        core_snapshot = [list(v) for v in self.cores]
        while self.active:
            assert self.events
            self.event(heapq.heappop(self.events), True)
        drained = self.now
        cost = B.cost(self.p, before, new)
        ready = drained + cost
        self.advance(ready, hold=True)
        self.layout = new; self.check_ram(); self.visited.add(new); self.changes += 1
        self.deployment_total += cost; self.drain_total += drained-start
        # Hardware timelines persist. Never reset core/link occupancy.
        assert all(b >= a for xs, ys in zip(core_snapshot, self.cores) for a, b in zip(xs, ys))
        queued = self.backlog; self.backlog = []
        for r in queued:
            self.admit(r, ready)
        self.actions.append(dict(start=start, drained=drained, ready=ready,
                                 deployment=cost, drain=drained-start, old=before, new=new,
                                 queued=len(queued)))

    def finish(self):
        while self.events:
            self.event(heapq.heappop(self.events))
        assert self.active == 0 and not self.backlog
        assert len(self.responses) == len(self.arrivals)
        assert len({r for r, v in self.responses}) == len(self.arrivals)
        vals = sorted(v for r, v in self.responses)
        return dict(mean=st.mean(vals), p95=vals[math.ceil(.95*len(vals))-1],
                    changes=self.changes, deployment=self.deployment_total, drain=self.drain_total,
                    completed=len(vals), peak_ram_fraction=self.peak_ram_fraction,
                    actions=self.actions)


def features(env, pool, step):
    p = env.p; now = env.now
    seen = [t for t in env.arrivals if t <= now]  # only past observations
    recent = sum(t > now-10 for t in seen)/min(10., max(now, 1.))
    rate = len(seen)/max(1., now)
    queue = [sum(max(0., end-max(now, begin)) for site, begin, end in env.task_intervals if site == s) for s in range(2)]
    links = [sum(max(0., end-max(now, begin)) for site, begin, end in env.link_intervals if site == s) for s in range(2)]
    common = p['work']+[v/256. for v in p['mem']]+[next((b for u,v,b in p['edges'] if (u,v)==pair),0.) for pair in B.PAIRS]
    common += p['speed']+[float(v) for v in p['cores']]+[p['mbps']/100.,p['rtt']]
    common += [now/60.,float(step),env.changes/2.,env.deployment_total/10.,env.drain_total/10.,
               env.active/100.,len(env.backlog)/100.,rate/10.,recent/10.]+queue+links
    out = []
    for l in pool:
        work = [sum(p['work'][i] for i in range(4) if l[1][l[0][i]] == s) for s in range(2)]
        used = [0., 0.]
        for g, ns in enumerate(B.groups(l[0])):
            used[l[1][g]] += .3*sum(p['mem'][i] for i in ns)+.7*max(p['mem'][i] for i in ns)
        raw = common+B.vector(p, env.layout)+B.vector(p,l)+[B.cost(p,env.layout,l),float(l==env.layout)]
        raw += [1-used[s]/p['ram'][s] for s in range(2)]+[rate*work[s]/p['speed'][s]/p['cores'][s] for s in range(2)]
        out.append([math.asinh(v) for v in raw])
    return out


class Policy(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(dim,32),torch.nn.Tanh(),torch.nn.Linear(32,16),torch.nn.Tanh())
        self.score = torch.nn.Linear(16,1); self.gate = torch.nn.Linear(48,1)
        self.critic = torch.nn.Sequential(torch.nn.Linear(48,32),torch.nn.Tanh(),torch.nn.Linear(32,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
        for head in (self.score,self.gate):
            torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)

    def forward(self, x, forced=False):
        h=self.encoder(x); other=h[1:] if len(h)>1 else h[:1]*0
        ctx=torch.cat((h[0],other.mean(0),other.max(0).values))
        logits=torch.cat((-self.gate(ctx).reshape(1),self.score(h[1:]).flatten()))
        if forced and len(x)>1:
            logits=torch.cat((logits[:1]*0-1e9,logits[1:]))
        lp=torch.log_softmax(logits,0)
        return lp,self.critic(ctx).squeeze(),torch.distributions.Categorical(logits=logits).entropy()


def rollout(p,old,arr,net,arm,stochastic=False):
    e=Engine(p,old,arr);ds=[];limit=1 if arm=='single' else 2
    for step in range(limit):
        pool=[e.layout]+[v for v in neighbors(p,e.layout)[1:] if v not in e.visited]
        x=features(e,pool,step);forced=arm=='forced'
        with torch.no_grad():
            lp,v,ent=net(torch.tensor(x,dtype=torch.float32),forced)
            idx=int(torch.distributions.Categorical(logits=lp).sample()) if stochastic else int(lp.argmax())
        ds.append(dict(x=x,index=idx,logp=float(lp[idx]),value=float(v),reward=0.,forced=forced,
                       time=e.now,options=len(pool)))
        if idx==0:
            break
        e.change(pool[idx])
        if step+1<limit:
            e.advance(max(15.,e.now))
            if e.now>=60.:
                break
    result=e.finish();ds[-1]['reward']=-result['mean']/100.
    result['decisions']=len(ds);result['stopped_early']=len(ds)==1 and ds[0]['index']==0
    result['indices']=[d['index'] for d in ds]
    return result,ds


def update(net,opt,batch):
    flat=[]
    for ep in batch:
        advantage=0.;future=0.
        for d in reversed(ep):
            delta=d['reward']+future-d['value'];advantage=delta+.95*advantage
            flat.append(dict(d,advantage=advantage,target=advantage+d['value']));future=d['value']
    aa=torch.tensor([d['advantage'] for d in flat]);aa=(aa-aa.mean())/(aa.std(unbiased=False)+1e-8)
    for _ in range(4):
        actors=[];critics=[];ents=[]
        for j,d in enumerate(flat):
            lp,v,ent=net(torch.tensor(d['x'],dtype=torch.float32),d['forced'])
            ratio=torch.exp(lp[d['index']]-d['logp'])
            actors.append(-torch.minimum(ratio*aa[j],ratio.clamp(.8,1.2)*aa[j]))
            critics.append((v-d['target']).square());ents.append(ent)
        loss=torch.stack(actors).mean()+.5*torch.stack(critics).mean()-.01*torch.stack(ents).mean()
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);opt.step()
    return float(loss.detach())


def profiles():
    out=[];ts=[[(0,1),(1,2),(2,3)],[(0,1),(0,2),(1,3),(2,3)],[(0,1),(0,2),(0,3)]]
    for i in range(18):
        rng=random.Random(1511000+i)
        p=dict(id=151000+i,family=('chain','diamond','tree')[i%3],work=[rng.uniform(.015,.16) for _ in range(4)],
               mem=[rng.choice([32,64,96]) for _ in range(4)],edges=[(u,v,rng.choice([.01,.1,1.])) for u,v in ts[i%3]],
               ram=[256,2048],cores=[1,3],speed=[1.,rng.uniform(1.5,3.)],mbps=rng.choice([10.,50.,100.]),rtt=rng.choice([.01,.05,.08]))
        ls=B.layouts(p);initials=random.Random(1512000+i).sample(ls,4)
        out.append(dict(profile=p,initials=initials,split='train' if i<12 else 'check'))
    return out


def preflight(ps):
    errors=[];count=0
    for block in ps[:6]:
        p=block['profile'];arr=B.flow(1513000+p['id'])
        for old in block['initials'][:2]:
            for new in (old,neighbors(p,old)[-1]):
                e=Engine(p,old,arr);e.change(new);r=e.finish()
                ref=B.simulate(p,new,arr,B.cost(p,old,new));errors.append(abs(r['mean']-ref));count+=1
            e=Engine(p,old,arr);e.advance(15.);before=e.active
            new=neighbors(p,old)[-1];e.change(new);r=e.finish()
            assert len(r['actions'])==1
            if before:
                assert r['actions'][0]['drained']>=15.
            count+=1
    assert max(errors)<1e-9,max(errors)
    # Two deployments, both charged, no resetting between them.
    p=ps[0]['profile'];old=ps[0]['initials'][0];e=Engine(p,old,B.flow(1517777))
    one=neighbors(p,old)[1];e.change(one);e.advance(max(15.,e.now))
    two=next(l for l in neighbors(p,one)[1:] if l!=old);e.change(two);r=e.finish()
    assert math.isclose(r['deployment'],B.cost(p,old,one)+B.cost(p,one,two))
    # Arrival exactly at switch timestamp is held and released only after deployment.
    e=Engine(p,old,[0.,15.,15.001]);e.advance(15.);e.change(one);r=e.finish()
    assert r['completed']==3
    return dict(cases=count+2,static_alignment_max_error=max(errors),two_costs_accumulated=True,
                all_requests_completed=True,persistent_cores_links=True,resource_constraints=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=512);args=parser.parse_args()
    torch.set_num_threads(1);ROOT.mkdir(parents=True,exist_ok=True);start=time.monotonic();ps=profiles()
    audit=preflight(ps);(ROOT/'preflight.json').write_text(json.dumps(audit,indent=2));print('PREFLIGHT',audit,flush=True)
    dim=len(features(Engine(ps[0]['profile'],ps[0]['initials'][0],[]),[ps[0]['initials'][0]],0)[0])
    manifest=dict(seeds=[17,29],arms=['single','stop','forced'],steps=args.steps,profiles=ps,feature_dim=dim,
                  protocol_sha256=hashlib.sha256(Path(__file__).with_name('R151_PROTOCOL.md').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    trainstates=[(b['profile'],l) for b in ps if b['split']=='train' for l in b['initials']]
    checks=[(b['profile'],l) for b in ps if b['split']=='check' for l in b['initials']]
    rows=[];training=[]
    for seed in (17,29):
        for arm in ('single','stop','forced'):
            torch.manual_seed(seed);net=Policy(dim);opt=torch.optim.Adam(net.parameters(),lr=.0003)
            count=0;ep=0;raw=[];tick=time.monotonic()
            while count<args.steps:
                batch=[]
                while len(batch)<8 and count<args.steps:
                    p,old=trainstates[random.Random(1514000+seed*10000+ep).randrange(len(trainstates))]
                    arr=B.flow(151500000+seed*10000+ep);torch.manual_seed(151600000+seed*10000+ep)
                    r,ds=rollout(p,old,arr,net,arm,True);batch.append(ds);count+=len(ds);ep+=1
                    raw.append(dict(episode=ep,profile=p['id'],**r))
                loss=update(net,opt,batch)
                if ep%64<8:print('TRAIN',seed,arm,'steps',count,'episodes',ep,flush=True)
            torch.save(dict(state_dict=net.state_dict(),input_dim=dim,seed=seed,arm=arm,steps=count),ROOT/f'{arm}_{seed}.pt')
            (ROOT/f'{arm}_{seed}_train.json').write_text(json.dumps(raw))
            training.append(dict(seed=seed,arm=arm,steps=count,episodes=ep,seconds=time.monotonic()-tick,loss=loss))
            for j,(p,old) in enumerate(checks):
                for stream in range(2):
                    arr=B.flow(151700000+p['id']*7+stream)
                    r,ds=rollout(p,old,arr,net,arm)
                    keep=Engine(p,old,arr).finish()
                    # Same first selected action, disable all subsequent actions: attribution only.
                    e=Engine(p,old,arr)
                    if r['actions']:e.change(tuple(map(tuple,r['actions'][0]['new'])))
                    firstonly=e.finish()
                    rows.append(dict(seed=seed,arm=arm,profile=p['id'],initial_index=j%4,stream=stream,
                                     keep_mean=keep['mean'],first_only_mean=firstonly['mean'],
                                     decision_times=[d['time'] for d in ds],**r))
            print('EVALUATED',seed,arm,'wall',round(time.monotonic()-tick,2),flush=True)
            (ROOT/'rows.json').write_text(json.dumps(rows))
    summary=dict(preflight=audit,training=training,feature_dim=dim,wall_s=time.monotonic()-start,methods={},seeds=[])
    for arm in ('single','stop','forced'):
        rr=[r for r in rows if r['arm']==arm]
        summary['methods'][arm]={k:st.mean(r[k] for r in rr) for k in ('mean','p95','changes','deployment','drain','keep_mean','first_only_mean')}
        summary['methods'][arm].update(early_stops=sum(r['stopped_early'] for r in rr),
            second_actions=sum(r['changes']==2 for r in rr),count=len(rr),
            serious_vs_keep=sum(r['mean']>1.1*r['keep_mean'] for r in rr),
            second_action_wins=sum(r['changes']==2 and r['mean']<r['first_only_mean'] for r in rr))
    for seed in (17,29):
        summary['seeds'].append(dict(seed=seed,**{a:st.mean(r['mean'] for r in rows if r['seed']==seed and r['arm']==a) for a in ('single','stop','forced')}))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
