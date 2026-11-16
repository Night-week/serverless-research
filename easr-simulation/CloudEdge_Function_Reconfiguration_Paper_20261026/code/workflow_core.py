"""Shared four-function workflow and layout utilities.

This is the reusable simulator portion of the former R116 module. The
similarity probe and its threshold-selection grid are intentionally omitted.
"""
import heapq
import itertools as it
import json
import math
from pathlib import Path
import random
import statistics as st

PAIRS=list(it.combinations(range(4),2))


def partitions(n=4):
    def rec(a):
        if len(a)==n:
            yield tuple(a); return
        for x in range(max(a)+2): yield from rec(a+[x])
    yield from rec([0])


def groups(m):
    return [[i for i in range(4) if m[i]==g] for g in sorted(set(m))]


def valid(p,m,z):
    gs=groups(m)
    used=[0.,0.]
    for g,ns in enumerate(gs):
        used[z[g]]+=.3*sum(p['mem'][i] for i in ns)+.7*max(p['mem'][i] for i in ns)
    if any(x>c for x,c in zip(used,p['ram'])): return False
    es={(m[u],m[v]) for u,v,b in p['edges'] if m[u]!=m[v]}
    done=set()
    while len(done)<len(gs):
        ready=[g for g in range(len(gs)) if g not in done and all(u in done for u,v in es if v==g)]
        if not ready:return False
        done.update(ready)
    return True


def layouts(p):
    return [(m,z) for m in partitions() for z in it.product(range(2),repeat=max(m)+1) if valid(p,m,z)]


def flow(seed):
    rng=random.Random(seed);t=0.;arr=[]
    # Independently sampled durations and rates, with an identical distribution
    # across history and evaluation. No knowledge of the evaluation stream.
    while t<60:
        end=min(60,t+rng.uniform(6,18));rate=rng.choice([2.,5.,10.])
        now=t
        while True:
            now+=rng.expovariate(rate)
            if now>=end:break
            arr.append(now)
        t=end
    return arr


def simulate(p,layout,arr,startup=0.):
    m,z=layout;gs=groups(m);K=len(gs)
    pred=[set() for _ in gs];succ=[set() for _ in gs];payload={}
    for u,v,b in p['edges']:
        a,c=m[u],m[v]
        if a!=c:
            pred[c].add(a);succ[a].add(c);payload[a,c]=payload.get((a,c),0)+b
    cores=[[0.]*p['cores'][j] for j in range(2)];links=[0.,0.]
    events=[];serial=it.count();remaining={};release={};last={};responses=[]
    def push(t,kind,r,g):heapq.heappush(events,(t,next(serial),kind,r,g))
    for r,t in enumerate(arr):
        remaining[r]=[len(x) for x in pred];release[r]=[0.]*K;last[r]=0
        for g in range(K):
            if not pred[g]:push(max(t,startup),0,r,g)
    while events:
        t,_,kind,r,g=heapq.heappop(events)
        if kind==0:
            site=z[g];idx=min(range(len(cores[site])),key=lambda j:cores[site][j])
            start=max(t,cores[site][idx]);end=start+sum(p['work'][n] for n in gs[g])/p['speed'][site]
            cores[site][idx]=end;push(end,1,r,g)
        else:
            last[r]=max(last[r],t)
            for h in sorted(succ[g]):
                ready=t
                if z[g]!=z[h]:
                    d=z[g];begin=max(t,links[d]);links[d]=begin+payload[g,h]*8/p['mbps']
                    ready=links[d]+p['rtt']
                else:ready+=.003
                release[r][h]=max(release[r][h],ready);remaining[r][h]-=1
                if remaining[r][h]==0:push(release[r][h],0,r,h)
            # All tasks have been scheduled at most once; sink completion count.
            if not succ[g]:
                remaining[r].append(-1)
                if len(remaining[r])-K==sum(not x for x in succ):responses.append(last[r]-arr[r])
    assert len(responses)==len(arr)
    return st.mean(responses)


def cost(p,old,new):
    if old==new:return 0.
    om,oz=old;nm,nz=new
    oldobjects={(tuple(ns),oz[g]) for g,ns in enumerate(groups(om))}
    fresh=[(ns,nz[g]) for g,ns in enumerate(groups(nm)) if (tuple(ns),nz[g]) not in oldobjects]
    # Synthetic image build/start plus WAN image transfer for new cloud packages.
    return .08+.04*len(fresh)+sum(sum(p['mem'][i] for i in ns)*.05*8/p['mbps'] for ns,s in fresh if s==1)


def local(old,new):
    om,oz=old;nm,nz=new
    changes=0
    for g,ns in enumerate(groups(om)):
        children={nm[i] for i in ns}
        if len(children)==1:
            if nz[next(iter(children))]!=oz[g]:return False
        elif len(children)==2:changes+=1
        else:return False
    # Refinement forbids merging nodes from different old packages.
    if any(om[i]!=om[j] and nm[i]==nm[j] for i,j in PAIRS):return False
    return changes==1 or old==new


def vector(p,l):
    m,z=l;gs=groups(m)
    return [float(m[i]==m[j]) for i,j in PAIRS]+[float(z[m[i]]==0) for i in range(4)]+[
        len(gs)/4]+[sum(p['work'][i] for i in range(4) if z[m[i]]==s)/sum(p['work']) for s in range(2)]


def cosine(x,y):return sum(a*b for a,b in zip(x,y))/math.sqrt(sum(a*a for a in x)*sum(b*b for b in y))

