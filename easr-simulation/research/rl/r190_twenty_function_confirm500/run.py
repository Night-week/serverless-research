import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,random,hashlib,shutil,traceback,concurrent.futures as cf,multiprocessing as mp
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'r187_twenty_function_scale'))
import importlib.util
spec=importlib.util.spec_from_file_location('r187run',HERE.parent/'r187_twenty_function_scale/run.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
SOURCE=R.ROOT;ROOT=HERE/'outputs';WORKERS=6

def profile(index):
 R.ROOT=ROOT
 dest=ROOT/'profiles'/f'{index}.json'
 if dest.exists():return json.loads(dest.read_text())
 p=R.raw_profile(100000+index);p['id']=190000+index
 pool=R.B.plan_pool(p,190200000+index,32)
 hist=R.B.flow(190300000+index)
 plan=min(pool,key=lambda l:(R.Engine(p,l,hist).finish()['mean'],l))
 old=random.Random(190400000+index).sample(pool,min(4,len(pool)))
 for s in (0,1):
  l=(tuple([0]*20),(s,))
  if R.B.valid(p,*l) and l not in old:old.append(l)
 b=dict(profile=p,initials=old,plan=plan,plan_candidates=len(pool),split='confirmation')
 R.write(dest,b);print('PROFILE_DONE',index+1,flush=True);return b

def evaluate(seed):
 R.ROOT=ROOT;R.torch.set_num_threads(1)
 dest=ROOT/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
 shutil.copy2(SOURCE/f'seed_{seed}/rf.joblib',dest/'rf.joblib')
 ck=R.torch.load(SOURCE/f'seed_{seed}/selected.pt',map_location='cpu',weights_only=False)
 net=R.Policy(ck['dim']);net.load_state_dict(ck['state']);net.eval()
 m=json.loads((ROOT/'manifest.json').read_text());R.evaluate(seed,net,m,dest)
 R.write(dest/'completed.json',dict(seed=seed,status='completed'));print('SEED_DONE',seed,flush=True)

def main():
 ROOT.mkdir(exist_ok=True)
 frozen={str(p):R.sha(p) for s in R.SEEDS for p in (SOURCE/f'seed_{s}/selected.pt',SOURCE/f'seed_{s}/rf.joblib')}
 old=json.loads((SOURCE/'manifest.json').read_text())
 with cf.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:blocks=list(pool.map(profile,range(500)))
 def signature(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='id'},sort_keys=True).encode()).hexdigest()
 new={signature(b['profile']) for b in blocks};prior={signature(b['profile']) for b in old['profiles']}
 assert len(new)==500 and not new&prior
 m=dict(profiles=blocks,n_functions=20,frozen_models=frozen,source_manifest_sha256=R.sha(SOURCE/'manifest.json'),independent_confirmation=True)
 R.write(ROOT/'manifest.json',m);R.write(ROOT/'status.json',dict(phase='evaluation',configs=500,workers=WORKERS))
 with cf.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:list(pool.map(evaluate,R.SEEDS))
 R.ROOT=ROOT;R.summarize(m)
 out=json.loads((ROOT/'summary.json').read_text());out['configs']=500;out['interpretation']='500 new held-out 20-function synthetic configurations; frozen PPO and matched RF; same three topology families.';R.write(ROOT/'summary.json',out)
 assert all(R.sha(Path(p))==h for p,h in frozen.items())
 R.write(ROOT/'status.json',dict(phase='completed_pending_user_analysis'));print('R190_ALL_FINISHED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:R.write(ROOT/'status.json',dict(phase='failed',traceback=traceback.format_exc()));raise
