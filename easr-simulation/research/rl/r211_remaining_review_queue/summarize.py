import json,collections
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;out={};rng=np.random.default_rng(21120261031)
for root in sorted((H/'outputs').iterdir()):
 if not root.is_dir():continue
 by=collections.defaultdict(lambda:np.zeros(6))
 for f in root.glob('seed_*/*.json'):
  for r in json.loads(f.read_text()):
   a=by[r['mode'],r['profile']];a+=[1,r['system']['affected']['mean'],r['whole']['affected']['mean'],int(r['system']['affected']['mean']>r['whole']['affected']['mean']+5),r['system'].get('rolling_changes',0),r['system'].get('rolling_fallbacks',0)]
 result={}
 for mode in ['startup','live','rise']:
  a=np.array([v for (m,p),v in by.items() if m==mode]);z=a.sum(0);boot=[]
  for _ in range(10000):
   v=a[rng.integers(len(a),size=len(a))].sum(0);boot.append(100*(v[2]-v[1])/v[2])
  result[mode]=dict(configs=len(a),records=int(z[0]),system=z[1]/z[0],whole=z[2]/z[0],improvement_pct=100*(z[2]-z[1])/z[2],ci95=np.quantile(boot,[.025,.975]).tolist(),severe_pct=100*z[3]/z[0],mean_rolling_changes=z[4]/z[0],mean_fallbacks=z[5]/z[0])
 out[root.name]=result
(H/'summary.json').write_text(json.dumps(out,indent=2));print('R211_ANALYSIS_FINISHED',flush=True)
