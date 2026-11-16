import json,collections
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;groups=collections.defaultdict(list);names=['final','base','grl_native','grl_response','greedy_fusion','critical_path','whole','keep']
for f in (H/'outputs').glob('seed_*/*.json'):
 for r in json.loads(f.read_text()):
  if r['mode']=='startup' and r['semantics']=='global':groups[r['profile']].append([r['methods'][m]['mean'] for m in names])
a=np.array([np.mean(v,0) for v in groups.values()]);means=a.mean(0);rng=np.random.default_rng(21820261101);boot=[]
for _ in range(20000):
 v=a[rng.integers(len(a),size=len(a))].mean(0);boot.append(100*(v[1:]-v[0])/v[1:])
ci=np.quantile(boot,[.025,.975],axis=0);out=dict(estimand='configuration-equal means,global startup;100 paired config clusters,10 frozen models;20000 bootstrap;unadjusted CIs',records=sum(len(v) for v in groups.values()),results={name:dict(mean=means[i],final_improvement_pct=100*(means[i]-means[0])/means[i],ci95=ci[:,i-1].tolist()) for i,name in enumerate(names) if i})
(H/'baseline_comparison_ci.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
