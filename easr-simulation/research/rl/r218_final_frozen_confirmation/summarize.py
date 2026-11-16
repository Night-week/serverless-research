import json,collections
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;groups=collections.defaultdict(list);main=collections.defaultdict(list);seedgroups=collections.defaultdict(list);counts=collections.Counter()
for f in (H/'outputs').glob('seed_*/*.json'):
 for r in json.loads(f.read_text()):
  vals=[]
  for m in ['base','final','whole']:
   v=r['methods'][m];delta=v['affected']['mean']-r['methods']['whole']['affected']['mean'];vals.extend([v['affected']['mean'],v['affected']['p95'],delta if delta>5 else 0,float(delta>5)]);counts[r['semantics'],r['mode'],m]+=delta>5
  groups[r['semantics'],r['mode'],r['profile']].append(vals);seedgroups[r['semantics'],r['mode'],r['seed'],r['profile']].append(vals)
  if r['semantics']=='global' and r['mode']=='startup':
   for m,v in r['methods'].items():main[m,r['profile']].append([v['mean'],v['p95']])
rng=np.random.default_rng(21820261101);out={'conditions':{},'main_baselines':{}}
for sem in ['global','rolling']:
 for mode in ['startup','live','rise']:
  a=np.array([np.mean(v,0) for (s,m,p),v in groups.items() if s==sem and m==mode]);mean=a.mean(0).reshape(3,4);boot=[]
  for _ in range(10000):
   z=a[rng.integers(len(a),size=len(a))].mean(0).reshape(3,4);boot.append([100*(z[1,0]/z[0,0]-1),100*(1-z[1,2]/z[0,2]) if z[0,2] else float('nan')])
  change=100*(mean[1,0]/mean[0,0]-1);reduction=100*(1-mean[1,2]/mean[0,2]) if mean[0,2] else None
  result=dict(configs=len(a),methods={m:dict(mean=mean[i,0],p95_mean=mean[i,1],severe_loss=mean[i,2],severe_rate_pct=100*mean[i,3],severe_records=counts[sem,mode,m]) for i,m in enumerate(['base','final','whole'])},mean_change_pct=change,severe_loss_reduction_pct=reduction,ci95=np.nanquantile(boot,[.025,.975],axis=0).tolist(),point_gate=bool(change<=2 and reduction is not None and reduction>=20),by_seed={})
  for seed in [601,631,659,683,719,743,769,797,823,853]:
   z=np.mean([np.mean(v,0) for (s,m,k,p),v in seedgroups.items() if s==sem and m==mode and k==seed],0).reshape(3,4);result['by_seed'][str(seed)]=dict(mean_change_pct=100*(z[1,0]/z[0,0]-1),severe_loss_reduction_pct=100*(1-z[1,2]/z[0,2]) if z[0,2] else None)
  out['conditions'][sem+'_'+mode]=result
for name in sorted({m for m,p in main}):out['main_baselines'][name]=np.mean([np.mean(v,0) for (m,p),v in main.items() if m==name],0).tolist()
(H/'summary.json').write_text(json.dumps(out,indent=2));print('R218_ANALYSIS_FINISHED',flush=True)
