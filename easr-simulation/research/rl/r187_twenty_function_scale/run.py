"""R187: twenty-function resource-proportional scale confirmation, from-scratch paired RF/PPO."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'1')
import sys,json,hashlib,random,math,statistics as st,time,copy,traceback,argparse,itertools as it
from pathlib import Path
import concurrent.futures as cf,multiprocessing as mp
import torch,joblib
from sklearn.ensemble import RandomForestClassifier
import core as B
from engine import Engine
HERE=Path(__file__).resolve().parent;ROOT=HERE/'outputs';WORKSPACE=HERE.parents[2]
FROZEN=WORKSPACE/'CloudEdge_Function_Reconfiguration_Paper_20261026/code'
sys.path.insert(0,str(FROZEN))
import r151_stopping_ppo_smoke as REF
SEEDS=(601,631,659,683,719,743,769,797,823,853);BUDGET=32768;MILESTONES=(0,4096,8192,16384,24576,32768)
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def save(p,v):
 q=p.with_suffix('.tmp');torch.save(v,q);q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def layout(x):return tuple(map(tuple,x))
def observe(e,pool,step):
 p=e.p;now=e.now;n=len(p['work']);seen=[t for t in e.arrivals if t<=now]
 rate=len(seen)/max(1.,now);recent=sum(t>now-10 for t in seen)/min(10.,max(now,1.))
 queue=[sum(max(0.,end-max(now,begin)) for site,begin,end in e.task_intervals if site==s) for s in (0,1)]
 links=[sum(max(0.,end-max(now,begin)) for site,begin,end in e.link_intervals if site==s) for s in (0,1)]
 ed={(u,v):b for u,v,b in p['edges']}
 common=p['work']+[v/256 for v in p['mem']]+[ed.get(pair,0.) for pair in it.combinations(range(n),2)]
 common+=p['speed']+list(map(float,p['cores']))+[p['mbps']/100,p['rtt']]
 common+=[now/60,float(step),e.changes/2,e.deployment_total/10,e.drain_total/10,e.active/100,len(e.backlog)/100,rate/10,recent/10]+queue+links
 past=[t-60 for t in B.flow(152300000+p['id'])];known=past+seen;hr=sum(now-60<t<=now for t in known)/60;rr=sum(now-10<t<=now for t in known)/10
 xs=[]
 for l in pool:
  used=[0.,0.];work=[sum(p['work'][v] for v in range(n) if l[1][l[0][v]]==s) for s in (0,1)]
  for g,ns in enumerate(B.groups(l[0])):used[l[1][g]]+=.3*sum(p['mem'][v] for v in ns)+.7*max(p['mem'][v] for v in ns)
  raw=common+B.vector(p,e.layout)+B.vector(p,l)+[B.cost(p,e.layout,l),float(l==e.layout)]+[1-used[s]/p['ram'][s] for s in (0,1)]+[rate*work[s]/p['speed'][s]/p['cores'][s] for s in (0,1)]+[hr,rr]
  wan=[0.,0.]
  for u,v,b in p['edges']:
   a,c=l[1][l[0][u]],l[1][l[0][v]]
   if a!=c:wan[a]+=b
  xs.append([math.asinh(v) for v in raw]+[math.asinh(hr*v*8/p['mbps']) for v in wan])
 return xs
def pressure(p,l,rate):
 cpu=[rate*sum(p['work'][v] for v in range(len(p['work'])) if l[1][l[0][v]]==s)/p['speed'][s]/p['cores'][s] for s in (0,1)];wan=[0.,0.]
 for u,v,b in p['edges']:
  a,c=l[1][l[0][u]],l[1][l[0][v]]
  if a!=c:wan[a]+=rate*b*8/p['mbps']
 return cpu,wan,max(cpu+wan)
class Policy(torch.nn.Module):
 def __init__(self,dim):
  super().__init__();self.policy=REF.Policy(dim);self.stop_value=torch.nn.Sequential(torch.nn.Linear(48,16),torch.nn.Tanh(),torch.nn.Linear(16,1));self.relative_value=torch.nn.Sequential(torch.nn.Linear(64,16),torch.nn.Tanh(),torch.nn.Linear(16,1));torch.nn.init.zeros_(self.relative_value[-1].weight);torch.nn.init.zeros_(self.relative_value[-1].bias)
 def context(self,x):
  h=self.policy.encoder(x);others=h[1:] if len(h)>1 else h[:1]*0;ctx=torch.cat((h[0],others.mean(0),others.max(0).values));return h,ctx
 def forward(self,x):
  lp,v,en=self.policy(x)
  # Step index location generalizes R151 index21: 2*N+N*(N-1)/2 + 7.
  n=20;stepindex=2*n+n*(n-1)//2+7
  if len(x)>1 and float(x[0,stepindex].detach())>0:
   h,ctx=self.context(x);delta=self.relative_value(torch.cat((ctx.expand(len(h)-1,-1),h[1:]-h[0]),1)).flatten();lp=torch.log_softmax(lp+torch.cat((delta.new_zeros(1),torch.tanh(delta/.1))).detach(),0)
   en=torch.distributions.Categorical(logits=lp).entropy()
  return lp,v,en
 def values(self,x,index):
  h,ctx=self.context(x);return self.stop_value(ctx).squeeze(),self.relative_value(torch.cat((ctx,h[index]-h[0]))).squeeze()
def rollout(p,old,arr,net,stochastic=False):
 e=Engine(p,old,arr);ds=[];pair=None
 for step in range(2):
  pool=[e.layout]+[l for l in B.neighbors(p,e.layout)[1:] if l not in e.visited];x=observe(e,pool,step)
  if step==1 and len(pool)>1:pair=(copy.deepcopy(e),pool,x)
  with torch.no_grad():
   lp,v,en=net(torch.tensor(x,dtype=torch.float32));idx=int(torch.distributions.Categorical(logits=lp).sample()) if stochastic else int(lp.argmax())
  ds.append(dict(x=x,index=idx,logp=float(lp[idx]),value=float(v),reward=0.))
  if idx==0:break
  e.change(pool[idx])
  if step==0:
   e.advance(max(15.,e.now))
   if e.now>=60:break
 r=e.finish();ds[-1]['reward']=-r['mean']/100;r['indices']=[d['index'] for d in ds];return r,ds,pair
def make_pair(state,rng):
 e,pool,x=state;j=rng.randrange(1,len(pool));stop=copy.deepcopy(e).finish()['mean'];a=copy.deepcopy(e);a.change(pool[j]);act=a.finish()['mean'];return dict(x=x,index=j,target_stop=-stop/100,target_delta=(stop-act)/100)
def update(net,opt,batch,pairs):
 flat=[]
 for ep in batch:
  adv=future=0.
  for d in reversed(ep):
   delta=d['reward']+future-d['value'];adv=delta+.95*adv;flat.append(dict(d,advantage=adv,target=adv+d['value']));future=d['value']
 aa=torch.tensor([d['advantage'] for d in flat]);aa=(aa-aa.mean())/(aa.std(unbiased=False)+1e-8)
 for _ in range(4):
  actor=[];critic=[];entropy=[]
  for j,d in enumerate(flat):
   lp,v,en=net(torch.tensor(d['x'],dtype=torch.float32));ratio=torch.exp(lp[d['index']]-d['logp']);actor.append(-torch.minimum(ratio*aa[j],ratio.clamp(.8,1.2)*aa[j]));critic.append((v-d['target']).square());entropy.append(en)
  loss=torch.stack(actor).mean()+.5*torch.stack(critic).mean()-.01*torch.stack(entropy).mean();aux=[]
  for d in pairs:
   q,delta=net.values(torch.tensor(d['x'],dtype=torch.float32),d['index']);aux.append(torch.nn.functional.smooth_l1_loss(q,q.new_tensor(d['target_stop']))+torch.nn.functional.smooth_l1_loss(delta,delta.new_tensor(d['target_delta'])))
  if aux:loss=loss+.1*torch.stack(aux).mean()
  assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);opt.step()
 return float(loss.detach())
def raw_profile(index):
 rng=random.Random(1871000+index);n=20;family=('chain','diamond','tree')[index%3]
 if family=='chain':edges=[(i,i+1) for i in range(n-1)]
 elif family=='tree':edges=[((i-1)//2,i) for i in range(1,n)]
 else:
  edges=[]
  # Repeated fork/join motifs connected in series, final tail preserves all 20 nodes.
  for a in range(0,18,3):edges.extend([(a,a+1),(a,a+2),(a+1,a+3),(a+2,a+3)])
  edges.append((18,19))
 return dict(id=187000+index,family=family,work=[rng.uniform(.015,.16) for _ in range(n)],mem=[rng.choice([32,64,96]) for _ in range(n)],edges=[(u,v,rng.choice([.01,.1,1.])) for u,v in edges],ram=[1280,10240],cores=[5,15],speed=[1.,rng.uniform(1.5,3.)],mbps=rng.choice([10.,50.,100.]),rtt=rng.choice([.01,.05,.08]))
def setup():
 if (ROOT/'manifest.json').exists():return read_manifest()
 blocks=[];cache=ROOT/'profiles';cache.mkdir(parents=True,exist_ok=True)
 for index in range(108):
  dest=cache/f'{index}.json'
  if dest.exists():b=json.loads(dest.read_text())
  else:
   p=raw_profile(index);pool=B.plan_pool(p,1872000+index,32);hist=B.flow(152300000+p['id']);scored=[]
   for l in pool:scored.append((Engine(p,l,hist).finish()['mean'],l))
   plan=min(scored,key=lambda v:(v[0],v[1]))[1];rng=random.Random(1873000+index);old=rng.sample(pool,min(4,len(pool)))
   for s in (0,1):
    l=(tuple([0]*20),(s,))
    if B.valid(p,*l) and l not in old:old.append(l)
   b=dict(profile=p,initials=old,plan=plan,plan_candidates=len(pool),split='train' if index<48 else 'validation' if index<60 else 'confirmation');write(dest,b)
  blocks.append(b);write(ROOT/'status.json',dict(phase='planning_historical_references',profiles=len(blocks),total=108));print('R187_PLAN',len(blocks),'/108',flush=True)
 manifest=dict(seeds=SEEDS,budget=BUDGET,profiles=blocks,n_functions=20,resource_scaling=5,plan_budget=32,source_bindings={p.name:sha(p) for p in [HERE/'core.py',HERE/'engine.py',Path(__file__),HERE/'PROTOCOL.md']},frozen_bindings={str(p):sha(p) for p in [FROZEN/'r151_stopping_ppo_smoke.py',FROZEN/'workflow_core.py']});write(ROOT/'manifest.json',manifest);return manifest
def read_manifest():
 m=json.loads((ROOT/'manifest.json').read_text());assert all(sha(HERE/k)==v for k,v in m['source_bindings'].items());assert all(sha(Path(k))==v for k,v in m['frozen_bindings'].items());return m
def states(m,split):return [(b,l,j) for b in m['profiles'] if b['split']==split for j,l in enumerate(b['initials'])]
def validate(net,m):
 vals=[]
 for b,l,j in states(m,'validation'):
  r,_,_=rollout(b['profile'],layout(l),B.flow(187700000+b['profile']['id']*7),net);vals.append(r['mean'])
 return st.mean(vals)
def gate_features(p,old,net):
 e=Engine(p,old,[]);pool=B.neighbors(p,old);x=observe(e,pool,0)
 with torch.no_grad():lp=net(torch.tensor(x,dtype=torch.float32))[0];idx=int(lp.argmax())
 hr=len(B.flow(152300000+p['id']))/60;press=[pressure(p,l,hr) for l in pool];best=min(press,key=lambda v:v[2]);cp=press[idx]
 return x[0]+[math.log1p(best[2]),sum(v[2]<1 for v in press)/len(press),*[math.asinh(v) for v in best[0]+best[1]],float(idx==0),*[math.asinh(v) for v in cp[0]+cp[1]],math.asinh(cp[2]),math.log1p(B.cost(p,old,pool[idx])),float(lp[idx].exp())]
def evaluate(seed,net,m,dest):
 rfpath=dest/'rf.joblib'
 if rfpath.exists():gate=joblib.load(rfpath)
 else:
  xs=[];ys=[];rows=[]
  for pos,(b,l,j) in enumerate(states(m,'train')):
   p=b['profile'];old=layout(l);x=gate_features(p,old,net)
   for stream in range(2):
    arr=B.flow(187900000+p['id']*7+stream);local,_,_=rollout(p,old,arr,net);e=Engine(p,old,arr);e.change(layout(b['plan']));whole=e.finish();y=whole['mean']<.97*local['mean'];xs.append(x);ys.append(y);rows.append(dict(profile=p['id'],initial=j,stream=stream,local=local['mean'],whole=whole['mean'],label=y))
   if pos%10==0:write(dest/'status.json',dict(phase='fitting_rf',states_done=pos+1))
  gate=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=6,class_weight='balanced',random_state=121,n_jobs=1).fit(xs,ys);joblib.dump(gate,rfpath);write(dest/'rf_training.json',rows)
 for num,b in enumerate(v for v in m['profiles'] if v['split']=='confirmation'):
  p=b['profile'];out=dest/f"test_{p['id']}.json"
  if out.exists():continue
  rows=[]
  for j,l in enumerate(b['initials']):
   old=layout(l);x=gate_features(p,old,net);whole_route=bool(gate.predict([x])[0])
   for stream in range(3):
    arr=B.flow(187800000+p['id']*7+stream);local,_,_=rollout(p,old,arr,net);e=Engine(p,old,arr);e.change(layout(b['plan']));whole=e.finish();keep=Engine(p,old,arr).finish()
    rows.append(dict(seed=seed,profile=p['id'],initial=j,stream=stream,rebuild=whole_route,local=local,whole=whole,keep=keep,system=whole if whole_route else local))
  write(out,rows);write(dest/'status.json',dict(phase='confirmation',profiles=num+1,total=48))
def train(seed):
 torch.set_num_threads(1);m=read_manifest();dest=ROOT/f'seed_{seed}';dest.mkdir(exist_ok=True)
 if (dest/'result.json').exists():return
 ss=states(m,'train');dim=len(observe(Engine(ss[0][0]['profile'],layout(ss[0][1]),[]),[layout(ss[0][1])],0)[0]);torch.manual_seed(seed);net=Policy(dim);opt=torch.optim.Adam(net.parameters(),lr=.0003);calls=episodes=pairs=0;vals=[];elapsed=0.;binding=sha(ROOT/'manifest.json')
 if (dest/'resume.pt').exists():
  ck=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False);assert ck['binding']==binding;net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);calls=ck['calls'];episodes=ck['episodes'];pairs=ck['pairs'];vals=ck['validation'];elapsed=ck['elapsed']
 else:
  vals=[dict(calls=0,mean=validate(net,m))];save(dest/'checkpoint_0.pt',dict(state=net.state_dict(),optimizer=opt.state_dict(),calls=0,validation=vals));print('R187_INITIAL',seed,vals[-1]['mean'],flush=True)
 tick=time.perf_counter()
 while calls<BUDGET:
  boundary=min((calls//128+1)*128,BUDGET)
  while calls<boundary:
   batch=[];pp=[]
   while len(batch)<8 and calls<boundary:
    ep=episodes;rng=random.Random(187400000+seed*100000+ep);b,l,j=ss[rng.randrange(len(ss))];torch.manual_seed(187500000+seed*100000+ep);r,ds,state=rollout(b['profile'],layout(l),B.flow(187600000+seed*100000+ep),net,True);calls+=1;episodes+=1;batch.append(ds)
    if rng.random()<.25 and state is not None and calls+2<=boundary:pp.append(make_pair(state,rng));calls+=2;pairs+=1
   loss=update(net,opt,batch,pp)
  if calls in MILESTONES:vals.append(dict(calls=calls,mean=validate(net,m)));print('R187',seed,'CALLS',calls,'/',BUDGET,'VALID',vals[-1]['mean'],flush=True)
  ck=dict(state=net.state_dict(),optimizer=opt.state_dict(),calls=calls,episodes=episodes,pairs=pairs,validation=vals,binding=binding,elapsed=elapsed+time.perf_counter()-tick,dim=dim);save(dest/'resume.pt',ck)
  if calls in MILESTONES:save(dest/f'checkpoint_{calls}.pt',ck)
  write(dest/'status.json',dict(phase='training',calls=calls,budget=BUDGET,episodes=episodes,pairs=pairs,elapsed=ck['elapsed'],input_dim=dim));print('R187_PROGRESS',seed,calls,'/',BUDGET,flush=True)
 best=min(vals,key=lambda v:(v['mean'],v['calls']));ck=torch.load(dest/f"checkpoint_{best['calls']}.pt",map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);net.eval();save(dest/'selected.pt',ck);write(dest/'selection.json',best);evaluate(seed,net,m,dest);assert calls==episodes+2*pairs;write(dest/'result.json',dict(seed=seed,calls=calls,episodes=episodes,pairs=pairs,selected=best,dim=dim));write(dest/'status.json',dict(phase='completed'));print('R187_SEED_DONE',seed,flush=True)
def summarize(m):
 rows=[r for s in SEEDS for b in m['profiles'] if b['split']=='confirmation' for r in json.loads((ROOT/f'seed_{s}'/f"test_{b['profile']['id']}.json").read_text())];metrics={}
 for arm in ('system','local','whole','keep'):
  extras=[r[arm]['mean']-r['whole']['mean'] for r in rows];metrics[arm]={k:st.mean(r[arm][k] for r in rows) for k in ('mean','p95','deployment','drain','changes')};metrics[arm].update(over5=sum(x>5 for x in extras),worst_extra=max(extras),severe_loss=sum(x for x in extras if x>5),gross_savings=sum(max(0,-x) for x in extras))
 seeds=[dict(seed=s,system=st.mean(r['system']['mean'] for r in rows if r['seed']==s),whole=st.mean(r['whole']['mean'] for r in rows if r['seed']==s)) for s in SEEDS]
 pairs=[(sum(r['whole']['mean'] for r in rows if r['profile']==pid),sum(r['system']['mean'] for r in rows if r['profile']==pid)) for pid in sorted({r['profile'] for r in rows})];rng=random.Random(187990);boot=[]
 for _ in range(3000):
  zz=rng.choices(pairs,k=len(pairs));boot.append(100*(1-sum(v for u,v in zz)/sum(u for u,v in zz)))
 boot.sort();gain=100*(1-metrics['system']['mean']/metrics['whole']['mean']);wins=sum(r['system']<r['whole'] for r in seeds);v=metrics['system']
 assert len(rows)==len({(r['seed'],r['profile'],r['initial'],r['stream']) for r in rows})
 assert all(r[a]['peak_ram_fraction']<=1+1e-9 for r in rows for a in ('local','whole','keep','system'))
 performance=bool(gain>=3 and boot[75]>0 and wins>=7);risk=bool(v['p95']<=1.02*metrics['whole']['p95'] and v['over5']/len(rows)<=.01 and v['severe_loss']<=.1*v['gross_savings'])
 write(ROOT/'summary.json',dict(records=len(rows),configs=48,metrics=metrics,seeds=seeds,mean_improvement_pct=gain,config_ci95=[boot[75],boot[2924]],winning_seeds=wins,performance_gate=performance,risk_gate=risk,support=performance and risk,interpretation='Twenty-function proportional-resource scale experiment. Bounded global-reference pool; not exhaustive optimum, not original fixed 69/82-dimensional models.'))
def smoke():
 torch.set_num_threads(1);tick=time.perf_counter();p=dict(id=1,work=[.03,.06,.02,.05],mem=[32,64,32,64],edges=[(0,1,.1),(0,2,.1),(1,3,.1),(2,3,.1)],ram=[256,2048],cores=[1,3],speed=[1.,2.],mbps=50.,rtt=.05)
 old=((0,0,1,1),(0,1));new=((0,1,2,3),(0,1,1,1));arr=B.flow(789)
 for switch in (False,True):
  a=Engine(p,old,arr);b=REF.Engine(p,old,arr)
  if switch:a.advance(15);b.advance(15);a.change(new);b.change(new)
  assert json.loads(json.dumps(a.finish()))==json.loads(json.dumps(b.finish()))
 assert B.neighbors(p,old)==REF.neighbors(p,old)
 import r153_candidate_wan as prior
 assert observe(Engine(p,old,arr),[old,new],0)==prior.observe(REF.Engine(p,old,arr),[old,new],0,'wan69')
 p=raw_profile(0);pool=B.plan_pool(p,77);old=pool[0];e=Engine(p,old,[]);xx=observe(e,[old],0);dim=len(xx[0]);torch.manual_seed(601);net=Policy(dim);opt=torch.optim.Adam(net.parameters(),lr=.0003)
 # Verify generalized step location matches frozen 4-function slot position algebra.
 idx=2*20+20*19//2+7;assert xx[0][idx]==0 and observe(e,[old],1)[0][idx]==math.asinh(1.)
 batch=[];completed=[]
 for fam in range(3):
  p=raw_profile(fam);ll=B.plan_pool(p,78+fam);old=ll[0];nn=B.neighbors(p,old);assert all(B.valid(p,*l) for l in nn)
  r,ds,state=rollout(p,old,B.flow(234+fam),net,True);batch.append(ds);completed.append(r['completed']);assert r['peak_ram_fraction']<=1+1e-9
 before=[v.detach().clone() for v in net.parameters()];loss=update(net,opt,batch,[]);assert any(not torch.equal(a,b) for a,b in zip(before,net.parameters()))
 # Manual serial compute test: zero communication after full fusion, one request.
 p=raw_profile(0);l=(tuple([0]*20),(1,));r=Engine(p,l,[0.]).finish();assert abs(r['mean']-sum(p['work'])/p['speed'][1])<1e-9
 info=dict(status='passed',four_function_engine_and_neighbor_reference_exact=True,functions=20,ppo_input_dim=dim,rf_input_dim=len(gate_features(p,l,net)),finite_update=True,parameters_updated=True,manual_serial_case_exact=True,completed_requests=completed,seconds=time.perf_counter()-tick)
 write(ROOT/'preflight.json',info);print(json.dumps(info),flush=True)
def main():
 args=argparse.ArgumentParser();args.add_argument('--smoke',action='store_true');args=args.parse_args();ROOT.mkdir(exist_ok=True)
 try:
  if args.smoke:smoke();return
  assert json.loads((ROOT/'preflight.json').read_text())['status']=='passed';m=setup();write(ROOT/'status.json',dict(phase='training',workers=2,seeds=SEEDS))
  with cf.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:list(pool.map(train,SEEDS))
  summarize(m);read_manifest();write(ROOT/'status.json',dict(phase='completed_pending_user_analysis'));print('R187_ALL_FINISHED',flush=True)
 except Exception:write(ROOT/'status.json',dict(phase='failed',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
