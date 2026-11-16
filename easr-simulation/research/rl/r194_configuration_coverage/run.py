import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,random,shutil,hashlib,collections,concurrent.futures as cf,multiprocessing as mp
from pathlib import Path
HERE=Path(__file__).resolve().parent;RL=HERE.parent
sys.path.insert(0,str(RL/'overnight_scale_20261028'));import suite as S
R=S.R;C=S.C;B=S.B;ROOT=HERE/'outputs';SOURCE=RL/'r187_twenty_function_scale';OLD=RL/'r190_twenty_function_confirm500';SEEDS=R.SEEDS;COUNTS=[20,50,100,500,1000]
def configure(n=20):
 assert n==20
 R.ROOT=ROOT;C.ROOT=HERE/'grl_outputs';C.SCALE=HERE;S.N=20
S.configure=configure

def sig(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='id'},sort_keys=True).encode()).hexdigest()
def plan(i):
 dest=ROOT/'new_profiles'/f'{i}.json'
 if dest.exists():return S.read(dest)
 p=R.raw_profile(100000+i);p['id']=190000+i;pool=B.plan_pool(p,190200000+i,32);hist=B.flow(190300000+i);best=min(pool,key=lambda l:(R.Engine(p,l,hist).finish()['mean'],l));initial=random.Random(190400000+i).sample(pool,min(4,len(pool)))
 for site in (0,1):
  l=(tuple([0]*20),(site,))
  if B.valid(p,*l) and l not in initial:initial.append(l)
 b=dict(profile=p,initials=initial,plan=best,plan_candidates=len(pool),split='confirmation');S.write(dest,b);print('NEW_CONFIG',i+1,flush=True);return b

def copy(src,dst):
 if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True);configure()
 old=S.read(OLD/'outputs/manifest.json');original=S.read(SOURCE/'outputs/manifest.json')
 assert len(old['profiles'])==500
 oldgrl=S.read(OLD/'grl_outputs/manifest.json');assert oldgrl['profiles']==old['profiles'],'GRL cache manifest mismatch'
 for seed in SEEDS:
  for name in ['selected.pt','checkpoint_0.pt','rf.joblib','rf_training.json']:copy(SOURCE/f'outputs/seed_{seed}'/name,ROOT/f'seed_{seed}'/name)
  for name in ['no_action','stale_labels','single','forced_second']:copy(SOURCE/f'ablations/seed_{seed}/{name}.joblib',HERE/f'ablations/seed_{seed}/{name}.joblib')
  copy(RL/f'r188_twenty_function_grl/outputs/seed_{seed}/selected.pt',HERE/f'grl_outputs/seed_{seed}/selected.pt')
  for b in old['profiles']:
   pid=b['profile']['id'];copy(OLD/f'outputs/seed_{seed}/test_{pid}.json',ROOT/f'seed_{seed}/test_{pid}.json');copy(OLD/f'grl_outputs/seed_{seed}/test_{pid}.json',HERE/f'grl_outputs/seed_{seed}/test_{pid}.json')
 # Reuse exactly completed results; never train or tune on this pool.
 blocks=old['profiles'];assert len({sig(b['profile']) for b in blocks})==500;assert not {sig(b['profile']) for b in blocks}&{sig(b['profile']) for b in original['profiles']}
 order=list(range(500));random.Random(19420261030).shuffle(order);blocks=[blocks[i] for i in order]
 frozen={str(p):S.sha(p) for seed in SEEDS for p in [SOURCE/f'outputs/seed_{seed}/selected.pt',SOURCE/f'outputs/seed_{seed}/rf.joblib',RL/f'r188_twenty_function_grl/outputs/seed_{seed}/selected.pt']}
 manifest=dict(profiles=blocks,frozen=frozen,original_manifest=original,nested_counts=COUNTS,prior500_reused=True,new_configs=500)
 target=HERE/'full_config_manifest.json'
 if target.exists():return S.read(target)
 else:S.write(target,manifest)
 return manifest

def job(task):
 kind,key=task;configure();R.torch.set_num_threads(1)
 if kind=='rules':S.rule_job((20,key))
 elif kind=='ours':
  d=ROOT/f'seed_{key}';ck=R.torch.load(d/'selected.pt',map_location='cpu',weights_only=False);net=R.Policy(ck.get('dim',685));net.load_state_dict(ck['state']);net.eval();R.evaluate(key,net,S.read(ROOT/'manifest.json'),d)
 elif kind=='grl':C.evaluate(key)
 elif kind=='ablation':
  done=HERE/f'ablations/seed_{key}/result.json'
  if done.exists():done.unlink()
  S.ablation_job((20,key))
 print('COVERAGE_JOB_DONE',kind,key,flush=True)

def summary(n,blocks):
 sums=collections.defaultdict(lambda:collections.Counter());counts=collections.Counter()
 def add(name,v,whole):
  counts[name]+=1
  for k in ['mean','p95','deployment','drain','changes']:sums[name][k]+=v[k]
  delta=v['mean']-whole['mean'];sums[name]['over5']+=int(delta>5);sums[name]['severe_loss']+=delta if delta>5 else 0;sums[name]['gross_savings']+=max(0,-delta)
 for seed in SEEDS:
  for b in blocks:
   pid=b['profile']['id'];ours=S.read(ROOT/f'seed_{seed}/test_{pid}.json');grl={(r['initial'],r['stream']):r for r in S.read(HERE/f'grl_outputs/seed_{seed}/test_{pid}.json')};rules={(r['initial'],r['stream']):r for r in S.read(HERE/f'rules/{pid}.json')};ab={(r['initial'],r['stream']):r for r in S.read(HERE/f'ablations/seed_{seed}/test_{pid}.json')}
   for r in ours:
    key=r['initial'],r['stream'];assert key in grl and key in rules and key in ab
    for a in ['system','local','whole','keep']:add(a,r[a],r['whole'])
    add('grl',grl[key]['grl'],r['whole'])
    for a,v in rules[key]['methods'].items():add(a,v,r['whole'])
    for a,v in ab[key]['methods'].items():
     if a not in ['whole','full']:add('ablation_'+a,v,r['whole'])
 out={a:dict(records=counts[a],**{k:v/counts[a] if k in ['mean','p95','deployment','drain','changes'] else v for k,v in val.items()}) for a,val in sums.items()}
 S.write(HERE/f'summary_{n}.json',dict(configs=n,n_functions=20,seeds=list(SEEDS),metrics=out,scope='Nested synthetic configuration coverage; frozen two-step models; old500 reused; four frozen-policy interventions with original matched RFs, no retraining. GRL local same-environment adapter, not author reproduction.'))

def main():
 m=prepare()
 for n in COUNTS:
  if (HERE/f'summary_{n}.json').exists():continue
  if n==1000 and len(m['profiles'])<1000:
   with cf.ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:new=list(pool.map(plan,range(500,1000)))
   m['profiles']+=new;assert len({sig(b['profile']) for b in m['profiles']})==1000;assert not {sig(b['profile']) for b in new}&{sig(b['profile']) for b in m['original_manifest']['profiles']};S.write(HERE/'full_config_manifest.json',m)
  blocks=m['profiles'][:n];active=dict(m['original_manifest']);active['profiles']=[b for b in active['profiles'] if b['split']!='confirmation']+blocks;S.write(ROOT/'manifest.json',active);S.write(HERE/'grl_outputs/manifest.json',dict(profiles=blocks))
  S.write(HERE/'status.json',dict(phase='evaluating',configs=n));print('COVERAGE_START',n,flush=True)
  phases=[('evaluation',[('rules',b['profile']['id']) for b in blocks]+[(kind,s) for s in SEEDS for kind in ['ours','grl']]),('ablations',[('ablation',s) for s in SEEDS])]
  for phase,tasks in phases:
   S.write(HERE/'status.json',dict(phase=phase,configs=n))
   with cf.ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:list(pool.map(job,tasks))
  summary(n,blocks);assert all(S.sha(Path(p))==h for p,h in m['frozen'].items());print('COVERAGE_FINISHED',n,flush=True)
 S.write(HERE/'status.json',dict(phase='completed'));print('R194_ALL_FINISHED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  S.write(HERE/'error.json',dict(traceback=traceback.format_exc()));raise
