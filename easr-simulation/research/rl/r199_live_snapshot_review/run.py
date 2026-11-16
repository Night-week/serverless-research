import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,random,math,statistics,time,argparse,traceback
from pathlib import Path
import concurrent.futures as cf,multiprocessing as mp
HERE=Path(__file__).resolve().parent; RL=HERE.parent
sys.path.insert(0,str(RL/'overnight_scale_20261028'));import suite as S
R=S.R;B=S.B;torch=R.torch;ROOT=HERE/'outputs';SOURCE=RL/'r194_configuration_coverage'
def read(p):return json.loads(p.read_text())
def features(e,net):
 p=e.p;pool=B.neighbors(p,e.layout);x=R.observe(e,pool,0)
 with torch.no_grad():lp=net(torch.tensor(x,dtype=torch.float32))[0];idx=int(lp.argmax())
 past=[t-60 for t in B.flow(152300000+p['id'])];known=past+[t for t in e.arrivals if t<=e.now];rate=sum(e.now-60<t<=e.now for t in known)/60
 press=[R.pressure(p,l,rate) for l in pool];best=min(press,key=lambda v:v[2]);cp=press[idx]
 return x[0]+[math.log1p(best[2]),sum(v[2]<1 for v in press)/len(press),*[math.asinh(v) for v in best[0]+best[1]],float(idx==0),*[math.asinh(v) for v in cp[0]+cp[1]],math.asinh(cp[2]),math.log1p(B.cost(p,e.layout,pool[idx])),float(lp[idx].exp())]
def finish(e,t0,pending):
 r=e.finish(); by=dict(e.responses)
 for name,ids in [('pending',pending),('future',[i for i,t in enumerate(e.arrivals) if t>t0]),('affected',sorted(set(pending)|{i for i,t in enumerate(e.arrivals) if t>t0}))]:
  vals=sorted(by[i] for i in ids);r[name]=dict(n=len(vals),mean=statistics.mean(vals) if vals else None,p95=vals[math.ceil(.95*len(vals))-1] if vals else None)
 return r
def local(e,net,t0,pending):
 indices=[]
 for step in range(2):
  pool=[e.layout]+[l for l in B.neighbors(e.p,e.layout)[1:] if l not in e.visited]
  with torch.no_grad():idx=int(net(torch.tensor(R.observe(e,pool,step),dtype=torch.float32))[0].argmax())
  indices.append(idx)
  if idx==0:break
  e.change(pool[idx])
  if step==0:
   e.advance(max(t0+15.,e.now))
   if e.now>=60:break
 r=finish(e,t0,pending);r['indices']=indices;return r
def job(task):
 seed,b=task;torch.set_num_threads(1);pid=b['profile']['id'];dest=ROOT/f'seed_{seed}'/f'{pid}.json'
 if dest.exists():return
 src=SOURCE/'outputs'/f'seed_{seed}';ck=torch.load(src/'selected.pt',map_location='cpu',weights_only=False);net=R.Policy(ck.get('dim',685));net.load_state_dict(ck['state']);net.eval();gate=R.joblib.load(src/'rf.joblib');rows=[]
 for j,l in enumerate(b['initials']):
  old=R.layout(l)
  for stream in range(3):
   original=B.flow(187800000+pid*7+stream)
   for mode in ('startup','live','rise'):
    t0=0. if mode=='startup' else 15.;arr=list(original)
    if mode=='rise':
     rng=random.Random(199000000+pid*7+stream);t=15.
     while True:
      t+=rng.expovariate(5.)
      if t>=60:break
      arr.append(t)
     arr.sort()
    e=R.Engine(b['profile'],old,arr);e.advance(t0);done={i for i,v in e.responses};pending=[i for i,t in enumerate(arr) if t<=t0 and i not in done]
    before=dict(active=e.active,backlog=len(e.backlog),pending=len(pending),now=e.now)
    z=features(e,net);route=bool(gate.predict([z])[0]);lo=local(copy.deepcopy(e),net,t0,pending);wh=copy.deepcopy(e);wh.change(R.layout(b['plan']));wh=finish(wh,t0,pending);keep=finish(copy.deepcopy(e),t0,pending)
    assert lo['completed']==wh['completed']==keep['completed']==len(arr)
    rows.append(dict(seed=seed,profile=pid,initial=j,stream=stream,mode=mode,snapshot=before,rebuild=route,local=lo,whole=wh,keep=keep,system=wh if route else lo))
 R.write(dest,rows);print('R199_CONFIG_DONE',seed,pid,len(rows),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--configs',type=int,default=100);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--seeds',type=int,default=10);a=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
 blocks=read(SOURCE/'full_config_manifest.json')['profiles'][:a.configs];seeds=list(R.SEEDS)[:a.seeds]
 R.write(HERE/'protocol.json',dict(configs=a.configs,seeds=seeds,modes=['startup','live_at_15s','live_at_15s_plus_independent_5rps'],frozen=True,no_retraining=True,second_opportunity='max(snapshot_time+15, deployment_ready)',metrics='all requests and pending/future/affected subsets',source_hashes={str(p):R.sha(p) for seed in seeds for p in [SOURCE/'outputs'/f'seed_{seed}'/'selected.pt',SOURCE/'outputs'/f'seed_{seed}'/'rf.joblib']}))
 tasks=[(s,b) for b in blocks for s in seeds]
 R.write(HERE/'status.json',dict(phase='running',jobs=len(tasks)))
 with cf.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:list(pool.map(job,tasks))
 R.write(HERE/'status.json',dict(phase='completed',jobs=len(tasks)));print('R199_ALL_FINISHED',flush=True)
if __name__=='__main__':main()
