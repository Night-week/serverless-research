"""Explicit response-objective adapter variant; original GRL baseline is untouched."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,dataclasses,concurrent.futures as cf,multiprocessing as mp,argparse
from pathlib import Path
H=Path(__file__).resolve().parent;RL=H.parent
sys.path.insert(0,str(RL/'r188_twenty_function_grl'));import run_grl as C
R=C.R;SOURCE=RL/'r194_configuration_coverage';C.ROOT=H/'outputs';C.SCALE=SOURCE
C.CFG=dataclasses.replace(C.CFG,latency_weight=1.,cost_weight=0.)
def response_reward(metrics,workflow):return -metrics['mean']/100.
C.reward=response_reward

def work(seed):
 C.torch.set_num_threads(1);C.train(seed);C.evaluate(seed);print('R209_SEED_DONE',seed,flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();C.ROOT.mkdir(parents=True,exist_ok=True)
 original=C.read(RL/'r187_twenty_function_scale/outputs/manifest.json');confirm=C.read(SOURCE/'full_config_manifest.json')['profiles'][:100];m=dict(profiles=[b for b in original['profiles'] if b['split'] in ['train','validation']]+confirm,seeds=list(R.SEEDS),objective='negative mean response /100; original architecture/action/optimizer retained',budget_episodes=C.BUDGET)
 path=C.ROOT/'manifest.json'
 if path.exists():assert C.read(path)==json.loads(json.dumps(m))
 else:C.write(path,m)
 C.write(H/'protocol.json',dict(objective='-mean_response/100',native_baseline='preserved in r188/r194',scope='objective-aligned adapter variant, not authors reproduction',seeds=list(R.SEEDS),train=48,validation=12,confirmation=100,budget_episodes=C.BUDGET,selection='same validation reward checkpoint and threshold selection, no confirmation tuning',source_hashes={str(p):R.sha(p) for p in [Path(__file__),RL/'r188_twenty_function_grl/run_grl.py',RL/'r188_twenty_function_grl/vendor/r8_grl_orch.py']}))
 if a.smoke:
  C.preflight();print('R209_SMOKE_PASSED',flush=True);return
 C.write(H/'status.json',dict(phase='training',seeds=list(R.SEEDS)))
 # First seed finishes before remaining seeds begin; preserve resumable per-seed checkpoints.
 work(R.SEEDS[0])
 with cf.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:list(pool.map(work,R.SEEDS[1:]))
 C.write(H/'status.json',dict(phase='completed'));print('R209_ALL_FINISHED',flush=True)
if __name__=='__main__':main()
