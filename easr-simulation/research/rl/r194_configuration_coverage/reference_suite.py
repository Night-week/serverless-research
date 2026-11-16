"""Resumable sequential 20/50/100-function suite; old 20-function sources remain immutable."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,math,random,time,copy,hashlib,traceback,contextlib,argparse,statistics as st,itertools as it
from pathlib import Path
import concurrent.futures as cf,multiprocessing as mp
HERE=Path(__file__).resolve().parent;WORKSPACE=HERE.parents[2];RL=HERE.parent
sys.path.insert(0,str(RL/'r188_twenty_function_grl'));import run_grl as C
R=C.R;B=R.B;Engine=R.Engine;torch=R.torch;ORIGINAL_POLICY=R.Policy;ORIGINAL_NEIGHBORS=B.neighbors
SEEDS=R.SEEDS;WORKERS=8;BUDGET=32768;N=20
write=R.write;read=C.read;sha=R.sha
class ScalePolicy(ORIGINAL_POLICY):
 def __init__(self,dim):super().__init__(dim);self.n=N
 def forward(self,x):
  lp,v,en=self.policy(x);idx=2*self.n+self.n*(self.n-1)//2+7
  if len(x)>1 and float(x[0,idx].detach())>0:
   h,ctx=self.context(x);delta=self.relative_value(torch.cat((ctx.expand(len(h)-1,-1),h[1:]-h[0]),1)).flatten();lp=torch.log_softmax(lp+torch.cat((delta.new_zeros(1),torch.tanh(delta/.1))).detach(),0);en=torch.distributions.Categorical(logits=lp).entropy()
  return lp,v,en

def bounded_neighbors(p,old):
 n=len(p['work'])
 if n<=20:return ORIGINAL_NEIGHBORS(p,old)
 gs=B.groups(old[0]);moves=set();proposals=set()
 for g,ns in enumerate(gs):
  other=[(v,old[1][j]) for j,v in enumerate(gs) if j!=g];moves.add(B.canonical(other+[(ns,1-old[1][g])],n))
  if len(ns)<2:continue
  if len(ns)<=8:cuts=[([v for j,v in enumerate(ns) if mask>>j&1],[v for j,v in enumerate(ns) if not mask>>j&1]) for mask in range(1,2**len(ns)-1) if mask&1]
  else:cuts=[(ns[:k],ns[k:]) for k in sorted({1,len(ns)//4,len(ns)//3,len(ns)//2,2*len(ns)//3,3*len(ns)//4,len(ns)-1}) if 0<k<len(ns)]
  for left,right in cuts:
   for a,b in it.product(range(2),repeat=2):proposals.add(B.canonical(other+[(left,a),(right,b)],n))
 result=[old]+[l for l in sorted(moves) if l!=old and B.valid(p,*l)]
 proposals=sorted(proposals-set(result));token=int(hashlib.sha256(repr((p['id'],old)).encode()).hexdigest()[:16],16);random.Random(token).shuffle(proposals)
 for l in proposals:
  if len(result)>=128:break
  if B.valid(p,*l):result.append(l)
 return result

def configure(n):
 global N;N=n;torch.set_num_threads(1)
 if n==20:
  R.ROOT=RL/'r187_twenty_function_scale/outputs';R.Policy=ORIGINAL_POLICY;B.neighbors=ORIGINAL_NEIGHBORS;C.ROOT=RL/'r188_twenty_function_grl/outputs';C.SCALE=RL/'r187_twenty_function_scale'
 else:
  slot=HERE/'runs'/f'n{n}';R.ROOT=slot/'outputs';R.Policy=ScalePolicy;B.neighbors=bounded_neighbors;C.ROOT=slot/'grl_outputs';C.SCALE=slot
 R.ROOT.mkdir(parents=True,exist_ok=True);C.ROOT.mkdir(parents=True,exist_ok=True)

def scale_profile(n,index):
 rng=random.Random(1871000+index);family=('chain','diamond','tree')[index%3]
 if family=='chain':edges=[(v,v+1) for v in range(n-1)]
 elif family=='tree':edges=[((v-1)//2,v) for v in range(1,n)]
 else:
  edges=[];last=0
  for a in range(0,n-3,3):edges.extend([(a,a+1),(a,a+2),(a+1,a+3),(a+2,a+3)]);last=a+3
  edges.extend((v,v+1) for v in range(last,n-1))
 cores=math.ceil(n/4)
 return dict(id=187000+(n-20)*1000+index,family=family,work=[rng.uniform(.015,.16) for _ in range(n)],mem=[rng.choice([32,64,96]) for _ in range(n)],edges=[(u,v,rng.choice([.01,.1,1.])) for u,v in edges],ram=[64*n,512*n],cores=[cores,3*cores],speed=[1.,rng.uniform(1.5,3.)],mbps=rng.choice([10.,50.,100.]),rtt=rng.choice([.01,.05,.08]))

def build_profile(job):
 n,index=job;configure(n);dest=R.ROOT/'profiles'/f'{index}.json'
 if dest.exists():return read(dest)
 p=scale_profile(n,index);pool=B.plan_pool(p,1872000+index,32);hist=B.flow(152300000+p['id']);plan=min(pool,key=lambda l:(Engine(p,l,hist).finish()['mean'],l));old=random.Random(1873000+index).sample(pool,min(4,len(pool)))
 for s in (0,1):
  l=(tuple([0]*n),(s,))
  if B.valid(p,*l) and l not in old:old.append(l)
 b=dict(profile=p,initials=old,plan=plan,plan_candidates=len(pool),split='train' if index<48 else 'validation' if index<60 else 'confirmation');write(dest,b);return b

def prepare_scale(n):
 configure(n)
 if n==20:m=R.read_manifest()
 elif (R.ROOT/'manifest.json').exists():m=R.read_manifest()
 else:
  (R.ROOT/'profiles').mkdir(exist_ok=True);blocks=[]
  with cf.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
   for index,b in enumerate(pool.map(build_profile,[(n,k) for k in range(108)])):
    blocks.append(b);write(HERE/'status.json',dict(phase='planning',scale=n,profiles=index+1,total=108));print('SUITE_PLAN',n,index+1,'/108',flush=True)
  m=dict(seeds=SEEDS,budget=BUDGET,profiles=blocks,n_functions=n,resource_scaling=n/4,plan_budget=32,source_bindings={p.name:sha(p) for p in [R.HERE/'core.py',R.HERE/'engine.py',R.HERE/'run.py',R.HERE/'PROTOCOL.md']},frozen_bindings={str(p):sha(p) for p in [R.FROZEN/'r151_stopping_ppo_smoke.py',R.FROZEN/'workflow_core.py']},suite_bindings={str(p):sha(p) for p in [Path(__file__),HERE/'PROTOCOL.md']},note='Suite runtime supplies variable-N policy step index and bounded candidates; all original source files unchanged.');write(R.ROOT/'manifest.json',m)
 if not (C.ROOT/'manifest.json').exists():
  paths=[Path(__file__),HERE/'PROTOCOL.md',C.HERE/'run_grl.py',C.HERE/'vendor/r8_grl_orch.py',C.HERE/'vendor/r8_policy_adapters.py',R.ROOT/'manifest.json']
  cm=dict(seeds=SEEDS,budget=BUDGET,checkpoints=C.POINTS,prices=C.PRICE,config=C.dataclasses.asdict(C.G.GRLTrainConfig()),profiles=m['profiles'],bindings={str(p):sha(p) for p in paths},r187_manifest_sha256=sha(R.ROOT/'manifest.json'),scope='Scale suite, unmodified GRL vendor, native reward.');write(C.ROOT/'manifest.json',cm)
 else:
  cm=read(C.ROOT/'manifest.json');assert cm['profiles']==m['profiles'];assert all(sha(Path(p))==v for p,v in cm['bindings'].items())
 return m

def rules_plan(p,method):
 n=len(p['work']);parent=list(range(n))
 def find(v):
  while parent[v]!=v:v=parent[v]
  return v
 edges=list(p['edges'])
 if method=='critical_path':
  best={};tails={}
  for u in reversed(range(n)):
   val,v=max([(b*8/p['mbps']+p['rtt']+best[v],v) for a,v,b in edges if a==u],default=(0.,None));best[u]=p['work'][u]/min(p['speed'])+val;tails[u]=[u]+([] if v is None else tails[v])
  roots=[v for v in range(n) if not any(b==v for a,b,_ in edges)];path=tails[max(roots,key=lambda v:(best[v],v))];keep=set(zip(path,path[1:]));edges=sorted([e for e in edges if tuple(e[:2]) in keep],key=lambda e:(-e[2],e[0],e[1]))
 else:edges.sort(key=lambda e:(-(e[2]*8/p['mbps']+p['rtt'])/(.3*(p['mem'][e[0]]+p['mem'][e[1]])+.7*max(p['mem'][e[0]],p['mem'][e[1]])),e[0],e[1]))
 for u,v,_ in edges:
  a,b=find(u),find(v)
  if a==b:continue
  saved=parent[:];parent[b]=a;gs=[sorted(i for i in range(n) if find(i)==r) for r in sorted({find(i) for i in range(n)})];l=B.canonical([(g,1) for g in gs],n)
  if not B.valid(p,*l):parent[:]=saved
 gs=[sorted(i for i in range(n) if find(i)==r) for r in sorted({find(i) for i in range(n)})];mapping={v:g for g,ns in enumerate(gs) for v in ns};deps={(mapping[u],mapping[v]) for u,v,_ in p['edges'] if mapping[u]!=mapping[v]};todo=set(range(len(gs)));order=[]
 while todo:
  ready=sorted(g for g in todo if all(u in order for u,v in deps if v==g));assert ready;order+=ready;todo-=set(ready)
 remaining=p['ram'][:];assigned={};finish={}
 for g in order:
  ns=gs[g];mem=.3*sum(p['mem'][v] for v in ns)+.7*max(p['mem'][v] for v in ns);feasible=[s for s in (0,1) if remaining[s]+1e-9>=mem];assert feasible
  def estimate(s):
   arrivals=[0.]
   for h,site in assigned.items():
    es=[b for u,v,b in p['edges'] if u in gs[h] and v in ns]
    if es:arrivals.append(finish[h]+(.003 if site==s else sum(es)*8/p['mbps']+p['rtt']))
   return max(arrivals)+sum(p['work'][v] for v in ns)/p['speed'][s]
  s=min(feasible,key=lambda s:(estimate(s),s));finish[g]=estimate(s);assigned[g]=s;remaining[s]-=mem
 return B.canonical([(ns,assigned[g]) for g,ns in enumerate(gs)],n)

def rule_job(job):
 n,pid=job;configure(n);m=R.read_manifest();b=next(b for b in m['profiles'] if b['profile']['id']==pid);p=b['profile'];dest=R.ROOT.parent/'rules'/f'{pid}.json'
 if dest.exists():return
 plans={a:rules_plan(p,a) for a in ('greedy_fusion','critical_path')};rows=[]
 for j,l in enumerate(b['initials']):
  old=R.layout(l)
  for stream in range(3):
   arr=B.flow(187800000+p['id']*7+stream);values={}
   for a,plan in plans.items():
    e=Engine(p,old,arr);e.change(plan);values[a]=e.finish();values[a]['compute_cost']=C.bill(e)
   rows.append(dict(profile=pid,initial=j,stream=stream,methods=values,layouts=plans))
 write(dest,rows)

def variant_rollout(p,old,arr,net,mode):
 e=Engine(p,old,arr);indices=[];cap=1 if mode=='single' else 2
 for step in range(cap):
  pool=[e.layout]+[l for l in B.neighbors(p,e.layout)[1:] if l not in e.visited];x=R.observe(e,pool,step)
  with torch.no_grad():lp,_,_=net(torch.tensor(x,dtype=torch.float32));idx=int(lp[1:].argmax())+1 if mode=='forced_second' and step==1 and len(pool)>1 else int(lp.argmax())
  indices.append(idx)
  if idx==0:break
  e.change(pool[idx])
  if step+1<cap:
   e.advance(max(15.,e.now))
   if e.now>=60:break
 r=e.finish();r['indices']=indices;return r

def ablation_job(job):
 n,seed=job;configure(n);m=R.read_manifest();dest=R.ROOT.parent/'ablations'/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return
 source=R.ROOT/f'seed_{seed}';ck=torch.load(source/'selected.pt',map_location='cpu',weights_only=False);sample=R.states(m,'train')[0];dim=len(R.observe(Engine(sample[0]['profile'],R.layout(sample[1]),[]),[R.layout(sample[1])],0)[0]);net=R.Policy(dim);net.load_state_dict(ck['state']);net.eval();oldnet=R.Policy(dim);oldnet.load_state_dict(torch.load(source/'checkpoint_0.pt',map_location='cpu',weights_only=False)['state']);oldnet.eval()
 modes=('no_action','stale_labels','single','forced_second');gates={}
 trainrows={(r['profile'],r['initial'],r['stream']):r for r in read(source/'rf_training.json')}
 if all((dest/f'{a}.joblib').exists() for a in modes):gates={a:R.joblib.load(dest/f'{a}.joblib') for a in modes}
 else:
  xs={a:[] for a in modes};ys={a:[] for a in modes};audit=[]
  for num,(b,l,j) in enumerate(R.states(m,'train')):
   p=b['profile'];old=R.layout(l);x=R.gate_features(p,old,net)
   for stream in range(2):
    key=(p['id'],j,stream);orig=trainrows[key];arr=B.flow(187900000+p['id']*7+stream);rv={}
    rv['stale_labels']=R.rollout(p,old,arr,oldnet)[0]['mean']
    for a in ('single','forced_second'):rv[a]=variant_rollout(p,old,arr,net,a)['mean']
    for a in modes:xs[a].append(x[:-8] if a=='no_action' else x);ys[a].append(orig['label'] if a=='no_action' else orig['whole']<.97*rv[a])
    audit.append(dict(profile=p['id'],initial=j,stream=stream,whole=orig['whole'],variant_means=rv))
   if num%20==0:write(dest/'status.json',dict(phase='refitting_rf',states=num+1))
  for a in modes:
   gates[a]=R.RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=6,class_weight='balanced',random_state=121,n_jobs=1).fit(xs[a],ys[a]);R.joblib.dump(gates[a],dest/f'{a}.joblib')
  write(dest/'training_audit.json',audit)
 for k,b in enumerate(z for z in m['profiles'] if z['split']=='confirmation'):
  p=b['profile'];out=dest/f"test_{p['id']}.json"
  if out.exists():continue
  full=read(source/f"test_{p['id']}.json");xcache={};rows=[]
  for record in full:
   j=record['initial'];old=R.layout(b['initials'][j])
   if j not in xcache:xcache[j]=R.gate_features(p,old,net)
   x=xcache[j];arr=B.flow(187800000+p['id']*7+record['stream']);methods=dict(full=record['system'],whole=record['whole']);routes={}
   for a in modes:
    route=bool(gates[a].predict([x[:-8] if a=='no_action' else x])[0]);routes[a]=route
    if route:methods[a]=record['whole']
    elif a in ('single','forced_second'):methods[a]=variant_rollout(p,old,arr,net,a)
    else:methods[a]=record['local']
   rows.append(dict(seed=seed,profile=p['id'],initial=j,stream=record['stream'],routes=routes,methods=methods))
  write(out,rows);write(dest/'status.json',dict(phase='confirmation',profiles=k+1,total=48))
 write(dest/'result.json',dict(seed=seed,scale=n,phase='completed',frozen_ppo_interventions=True));write(dest/'status.json',dict(phase='completed'))

def worker(job):
 kind,n,key=job;configure(n);logs=HERE/'logs';logs.mkdir(exist_ok=True)
 with (logs/f'n{n}_{kind}_{key}.log').open('a',buffering=1) as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
  try:
   if kind=='ours':R.train(key)
   elif kind=='grl':C.work(key)
   elif kind=='rules':rule_job((n,key))
   elif kind=='ablation':ablation_job((n,key))
   else:raise ValueError(kind)
  except Exception:write(HERE/'errors'/f'n{n}_{kind}_{key}.json',dict(job=job,traceback=traceback.format_exc()));raise
 return dict(kind=kind,scale=n,key=key,status='completed')

def metrics(records,arm,ref='whole'):
 vv=[r[arm] for r in records];extras=[r[arm]['mean']-r[ref]['mean'] for r in records]
 return dict(**{k:st.mean(v[k] for v in vv) for k in ('mean','p95','deployment','drain','changes')},over5=sum(x>5 for x in extras),worst_extra=max(extras),severe_loss=sum(x for x in extras if x>5),gross_savings=sum(max(0,-x) for x in extras))

def summarize_scale(n):
 configure(n);R.summarize(R.read_manifest());C.summarize();m=R.read_manifest();rows=[];ab=[]
 for seed in SEEDS:
  for b in m['profiles']:
   if b['split']!='confirmation':continue
   pid=b['profile']['id'];ours=read(R.ROOT/f'seed_{seed}'/f'test_{pid}.json');grl={(r['initial'],r['stream']):r for r in read(C.ROOT/f'seed_{seed}'/f'test_{pid}.json')};rules={(r['initial'],r['stream']):r for r in read(R.ROOT.parent/'rules'/f'{pid}.json')}
   for r in ours:
    key=(r['initial'],r['stream']);rows.append(dict(seed=seed,profile=pid,initial=key[0],stream=key[1],ours=r['system'],whole=r['whole'],keep=r['keep'],local=r['local'],grl=grl[key]['grl'],**rules[key]['methods']))
   for r in read(R.ROOT.parent/'ablations'/f'seed_{seed}'/f'test_{pid}.json'):ab.append(dict(seed=seed,profile=pid,**r['methods']))
 baseline={a:metrics(rows,a) for a in ('ours','whole','keep','local','grl','greedy_fusion','critical_path')};ablations={a:metrics(ab,a) for a in ('full','no_action','stale_labels','single','forced_second')};result=dict(scale=n,records=len(rows),configs=48,seeds=SEEDS,baselines=baseline,ablations=ablations,source_manifest_sha256=sha(R.ROOT/'manifest.json'),grl_native_objective_retained=True,ablation_scope='frozen-PPO interventions plus matched RF refitting; not retrained architectures')
 write(R.ROOT.parent/'COMPLETE_SUMMARY.json',result)

def verify_bindings(n):
 configure(n);m=R.read_manifest()
 for p,v in m.get('suite_bindings',{}).items():assert sha(Path(p))==v
 cm=read(C.ROOT/'manifest.json');assert all(sha(Path(p))==v for p,v in cm['bindings'].items())

def smoke(n):
 configure(n);p=R.raw_profile(0) if n==20 else scale_profile(n,0);pool=B.plan_pool(p,44,32);old=pool[0];x=R.observe(Engine(p,old,[]),[old],0);dim=len(x[0]);net=R.Policy(dim);idx=2*n+n*(n-1)//2+7;assert x[0][idx]==0 and R.observe(Engine(p,old,[]),[old],1)[0][idx]==math.asinh(1.)
 with torch.no_grad():lp,_,_=net(torch.tensor(x,dtype=torch.float32))
 assert lp.isfinite().all();nb=B.neighbors(p,old);assert all(B.valid(p,*l) for l in nb);assert n==20 or len(nb)<=128
 l=(tuple([0]*n),(1,));res=Engine(p,l,[0.]).finish();assert abs(res['mean']-sum(p['work'])/p['speed'][1])<1e-9
 for a in ('greedy_fusion','critical_path'):assert B.valid(p,*rules_plan(p,a))
 w=C.workflow(p);g=C.G.GRLActorCritic(C.dataclasses.replace(C.G.GRLTrainConfig(),seed=601))
 with torch.no_grad():plan=C.G.sample_grl_plan(g,w,C.CFG,stochastic=True)
 glp,_,_=C.G._recompute_joint_logp(g,w,plan,C.CFG);assert abs(float(glp.detach()-plan.log_probability))<1e-4;C.execute(p,old,B.flow(188001),plan)
 r,ds,_=R.rollout(p,old,B.flow(188002),net,True);R.update(net,torch.optim.Adam(net.parameters(),lr=.0003),[ds],[])
 return dict(scale=n,ppo_dim=dim,rf_dim=len(R.gate_features(p,old,net)),candidate_count=len(nb),engine_manual_exact=True,finite_ppo_update=True,grl_logp_exact=True,rules_legal=True)

def main():
 args=argparse.ArgumentParser();args.add_argument('--smoke',action='store_true');args.add_argument('--summarize',type=int);args=args.parse_args()
 if args.smoke:result=[smoke(n) for n in (20,50,100)];write(HERE/'SMOKE.json',result);print(json.dumps(result,indent=2));return
 if args.summarize:summarize_scale(args.summarize);return
 assert (HERE/'SMOKE.json').exists();suite_manifest=HERE/'SUITE_MANIFEST.json'
 binding={str(p):sha(p) for p in [Path(__file__),HERE/'PROTOCOL.md']}
 if suite_manifest.exists():assert read(suite_manifest)['bindings']==binding
 else:write(suite_manifest,dict(seeds=SEEDS,scales=[20,50,100],budget=BUDGET,workers=WORKERS,bindings=binding))
 for n in (20,50,100):
  print('SCALE_START',n,flush=True);m=prepare_scale(n);verify_bindings(n)
  phases=[('train_and_baselines',[(kind,n,s) for s in SEEDS for kind in ('ours','grl')]),('rules',[('rules',n,b['profile']['id']) for b in m['profiles'] if b['split']=='confirmation']),('ablations',[('ablation',n,s) for s in SEEDS])]
  for phase,jobs in phases:
   write(HERE/'status.json',dict(scale=n,phase=phase,completed_jobs=0,total_jobs=len(jobs)));done=0
   with cf.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
    futures=[pool.submit(worker,job) for job in jobs]
    for future in cf.as_completed(futures):
     result=future.result();done+=1;write(HERE/'status.json',dict(scale=n,phase=phase,completed_jobs=done,total_jobs=len(jobs)));print('JOB_DONE',json.dumps(result),done,'/',len(jobs),flush=True)
  summarize_scale(n);verify_bindings(n);write(HERE/f'n{n}_COMPLETED.json',dict(scale=n,status='completed'));print('SCALE_FINISHED',n,flush=True)
 write(HERE/'status.json',dict(phase='completed_pending_user_analysis',scales=[20,50,100]));print('OVERNIGHT_ALL_FINISHED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:write(HERE/'status.json',dict(phase='failed',traceback=traceback.format_exc()));raise
