import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil,concurrent.futures as cf,multiprocessing as mp
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'r188_twenty_function_grl'))
import run_grl as C
SOURCE=C.ROOT;ROOT=HERE/'grl_outputs'
def evaluate(seed):
 C.ROOT=ROOT;C.SCALE=HERE
 C.evaluate(seed)
def main():
 ROOT.mkdir(exist_ok=True)
 m=json.loads((HERE/'outputs/manifest.json').read_text())
 bindings={}
 for s in C.SEEDS:
  p=SOURCE/f'seed_{s}/selected.pt';bindings[str(p)]=C.R.sha(p)
  d=ROOT/f'seed_{s}';d.mkdir(exist_ok=True);shutil.copy2(p,d/'selected.pt')
 C.write(ROOT/'manifest.json',dict(profiles=m['profiles'],frozen_models=bindings,scope='Frozen R188 20-function GRL-Orch local adapter; exact R190 configurations and streams; no retraining or threshold selection.'))
 with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:list(pool.map(evaluate,C.SEEDS))
 C.ROOT=ROOT;C.SCALE=HERE;C.summarize()
 assert all(C.R.sha(Path(p))==h for p,h in bindings.items())
 C.write(ROOT/'status.json',dict(phase='completed_pending_user_analysis'));print('R190_GRL_ALL_FINISHED',flush=True)
if __name__=='__main__':main()
