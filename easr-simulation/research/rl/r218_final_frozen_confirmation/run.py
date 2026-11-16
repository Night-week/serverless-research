import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,copy,math,random,time,dataclasses,importlib.util,concurrent.futures as cf,multiprocessing as mp
from pathlib import Path
H=Path(__file__).resolve().parent;RL=H.parent;sys.path.insert(0,str(RL/'r199_live_snapshot_review'));import run as X
R=X.R;B=X.B;C=X.S.C;torch=R.torch;SRC=X.SOURCE
spec=importlib.util.spec_from_file_location('rolling_final',RL/'r211_remaining_review_queue/rolling_transition.py');T=importlib.util.module_from_spec(spec);sys.modules[spec.name]=T;spec.loader.exec_module(T)
def load(seed):
 p=SRC/'outputs'/f'seed_{seed}';ck=torch.load(p/'selected.pt',map_location='cpu',weights_only=False);net=R.Policy(ck.get('dim',685));net.load_state_dict(ck['state']);net.eval();return net,R.joblib.load(p/'rf.joblib')
def calibrate():
 dest=H/'calibration.json'
 if dest.exists():return X.read(dest)
 torch.set_num_threads(1);blocks=[b for b in X.read(RL/'r187_twenty_function_scale/outputs/manifest.json')['profiles'] if b['split']=='validation'];times=[]
 for seed in [601,631,659]:
  net,gate=load(seed)
  for b in blocks[:6]:
   p=b['profile'];old=R.layout(b['initials'][0])
   for rep in range(12):
    tick=time.perf_counter();z=R.gate_features(p,old,net);route=bool(gate.predict([z])[0]);final_route=route or math.sinh(z[-3])>1;elapsed=time.perf_counter()-tick
    if rep>=2:times.append(elapsed)
 out=dict(samples=len(times),mean_seconds=sum(times)/len(times),p95_seconds=sorted(times)[math.ceil(.95*len(times))-1],scope='warmed complete first-route + fixed pressure test,3 frozen models,6 original validation profiles,CPU1thread; second-step uses same mean as conservative proxy')
 R.write(dest,out);print('R218_CALIBRATED',out,flush=True);return out

def profile(i):
 dest=H/'profiles'/f'{i}.json'
 if dest.exists():return X.read(dest)
 p=R.raw_profile(800000+i);p['id']=218000+i;pool=B.plan_pool(p,218100000+i,32);hist=B.flow(218200000+i);ref=min(pool,key=lambda l:(R.Engine(p,l,hist).finish()['mean'],l));initial=random.Random(218300000+i).sample(pool,min(4,len(pool)))
 for s in [0,1]:
  l=(tuple([0]*20),(s,))
  if B.valid(p,*l) and l not in initial:initial.append(l)
 b=dict(profile=p,plan=ref,initials=initial);R.write(dest,b);return b

def job(task):
 seed,b,delay=task;torch.set_num_threads(1);pid=b['profile']['id'];dest=H/'outputs'/f'seed_{seed}'/f'{pid}.json'
 if dest.exists():return
 net,gate=load(seed);extra={}
 for name,path in [('grl_native',SRC/'grl_outputs'/f'seed_{seed}'/'selected.pt'),('grl_response',RL/'r209_grl_response_objective/outputs'/f'seed_{seed}'/'selected.pt')]:
  ck=torch.load(path,map_location='cpu',weights_only=False);g=C.G.GRLActorCritic(dataclasses.replace(C.G.GRLTrainConfig(),seed=seed));g.load_state_dict(ck['state']);g.eval();w=C.workflow(b['profile'])
  with torch.no_grad():plan=C.G.sample_grl_plan(g,w,C.CFG,stochastic=False,threshold=ck['threshold'])
  extra[name]=C.layout(plan,20)
 for name in ['greedy_fusion','critical_path']:extra[name]=X.S.rules_plan(b['profile'],name)
 rows=[]
 for j,l in enumerate(b['initials']):
  old=R.layout(l)
  for stream in range(3):
   original=B.flow(218400000+pid*7+stream)
   for mode in ['startup','live','rise']:
    t0=0. if mode=='startup' else 15.;arr=list(original)
    if mode=='rise':
     rng=random.Random(218500000+pid*7+stream);t=15.
     while True:
      t+=rng.expovariate(5.)
      if t>=60:break
      arr.append(t)
     arr.sort()
    for semantics,Engine in [('global',R.Engine),('rolling',T.Rolling)]:
     e=Engine(b['profile'],old,arr);e.advance(t0);done={i for i,v in e.responses};pending=[i for i,t in enumerate(arr) if t<=t0 and i not in done];z=X.features(e,net);route=bool(gate.predict([z])[0]);pressure=math.sinh(z[-3]);pool=B.neighbors(b['profile'],old)
     with torch.no_grad():first=int(net(torch.tensor(R.observe(e,pool,0),dtype=torch.float32))[0].argmax())
     w=copy.deepcopy(e);w.change(R.layout(b['plan']));whole=X.finish(w,t0,pending);base=whole if route else X.local(copy.deepcopy(e),net,t0,pending)
     f=copy.deepcopy(e);f.advance(t0+delay)
     if route or pressure>1:f.change(R.layout(b['plan']));final=X.finish(f,t0,pending)
     else:final=T.local(f,net,t0,pending,first,delay)
     methods=dict(base=base,final=final,whole=whole)
     if mode=='startup' and semantics=='global':
      methods['keep']=X.finish(copy.deepcopy(e),t0,pending)
      for name,plan in extra.items():v=copy.deepcopy(e);v.change(plan);methods[name]=X.finish(v,t0,pending)
     assert all(v['completed']==len(arr) for v in methods.values());rows.append(dict(seed=seed,profile=pid,initial=j,stream=stream,mode=mode,semantics=semantics,pressure=pressure,methods=methods))
 R.write(dest,rows);print('R218_JOB_DONE',seed,pid,flush=True)
def main():
 R.write(H/'protocol_binding.json',dict(threshold=1,cap=2,seeds=list(R.SEEDS),configs=100,conditions='startup/live/rise x global/rolling',gate='final vs base mean deterioration<=2%, severe loss decrease>=20%; each condition report CI and seeds',source_hashes={str(p):R.sha(p) for p in [Path(__file__),RL/'r199_live_snapshot_review/run.py',RL/'r211_remaining_review_queue/rolling_transition.py']},frozen_weights={str(p):R.sha(p) for s in R.SEEDS for p in [SRC/'outputs'/f'seed_{s}'/'selected.pt',SRC/'outputs'/f'seed_{s}'/'rf.joblib',SRC/'grl_outputs'/f'seed_{s}'/'selected.pt',RL/'r209_grl_response_objective/outputs'/f'seed_{s}'/'selected.pt']}))
 cal=calibrate()
 with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:blocks=list(pool.map(profile,range(100)))
 def sig(p):return __import__('hashlib').sha256(json.dumps({k:v for k,v in p.items() if k!='id'},sort_keys=True).encode()).hexdigest()
 prior=X.read(SRC/'full_config_manifest.json')['profiles']+X.read(RL/'r187_twenty_function_scale/outputs/manifest.json')['profiles']+[X.read(p) for p in (RL/'r215_grouped_risk_confirmation/profiles').glob('*.json')];assert len({sig(b['profile']) for b in blocks})==100;assert not {sig(b['profile']) for b in blocks}&{sig(b['profile']) for b in prior}
 # Three-scale smoke is not relevant here: first condition exercises all six scenarios and baselines.
 job((R.SEEDS[0],blocks[0],cal['mean_seconds']));print('R218_SMOKE_PASSED',flush=True)
 with cf.ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context('spawn')) as pool:list(pool.map(job,[(s,b,cal['mean_seconds']) for b in blocks for s in R.SEEDS]))
 binding=X.read(H/'protocol_binding.json');assert all(R.sha(Path(p))==v for p,v in binding['frozen_weights'].items());R.write(H/'status.json',dict(phase='completed'));print('R218_ALL_FINISHED',flush=True)
if __name__=='__main__':main()
