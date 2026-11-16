"""20-function GRL-Orch adapter, unchanged R181 vendor, same R187 conditions/seeds."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys,json,math,random,time,traceback,dataclasses,hashlib,importlib.util,statistics as st,argparse
from pathlib import Path
import concurrent.futures as cf,multiprocessing as mp
HERE=Path(__file__).resolve().parent;SCALE=HERE.parent/'r187_twenty_function_scale'
sys.path.insert(0,str(SCALE));spec=importlib.util.spec_from_file_location('r187_shared',SCALE/'run.py');R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)
sys.path.insert(0,str(HERE/'vendor'));import r8_grl_orch as G
from r8_policy_adapters import Edge
torch=R.torch;B=R.B;Engine=R.Engine;ROOT=HERE/'outputs';SEEDS=R.SEEDS;BUDGET=32768;POINTS=(0,4000,4096,8192,16384,24576,32768);CFG=G.GRLRewardConfig();PRICE=(.35,.9)
def read(p):return json.loads(p.read_text())
write=R.write;save=R.save;sha=R.sha

def workflow(p):
 n=len(p['work']);single=(tuple(range(n)),tuple([1]*n));baseline=Engine(p,single,B.flow(152300000+p['id'])).finish()['mean']
 return G.GRLWorkflow(workflow_id=str(p['id']),nodes=tuple(str(i) for i in range(n)),edges=tuple(Edge(str(u),str(v),'dependency',b) for u,v,b in p['edges']),workload_cycles={str(i):v*1e9 for i,v in enumerate(p['work'])},memory_mib={str(i):v for i,v in enumerate(p['mem'])},deadline_s={str(i):60. for i in range(n)},site_cpu_cycles_s={str(s):p['speed'][s]*1e9 for s in (0,1)},site_memory_mib={str(s):p['ram'][s] for s in (0,1)},site_price_per_s={str(s):PRICE[s] for s in (0,1)},link_bandwidth_mbps={('0','1'):p['mbps'],('1','0'):p['mbps']},link_latency_s={('0','1'):p['rtt'],('1','0'):p['rtt']},baseline_latency_s=max(1e-9,baseline),baseline_cost=sum(p['work'])/p['speed'][1]*PRICE[1])
def layout(plan,n):return B.canonical([(list(map(int,g)),int(plan.placement[g])) for g in plan.groups],n)
def bill(e):return sum((end-begin)*PRICE[s] for s,begin,end in e.task_intervals)/len(e.arrivals)
def execute(p,old,arr,plan):
 l=layout(plan,len(p['work']));assert B.valid(p,*l),'Infeasible GRL plan; do not silently replace';e=Engine(p,old,arr);e.change(l);r=e.finish();r['compute_cost']=bill(e);return r
def reward(r,w):return G.grl_reward(G.GRLMetrics(r['mean'],r['compute_cost'],0.),w,CFG)
def prepare():
 m=R.read_manifest();paths=[Path(__file__),HERE/'PROTOCOL.md',HERE/'vendor/r8_grl_orch.py',HERE/'vendor/r8_policy_adapters.py',SCALE/'outputs/manifest.json']
 paths+=[SCALE/p for p in ('run.py','core.py','engine.py')]
 frozen=dict(seeds=SEEDS,budget=BUDGET,checkpoints=POINTS,prices=PRICE,config=dataclasses.asdict(G.GRLTrainConfig()),profiles=m['profiles'],bindings={str(p):sha(p) for p in paths},r187_manifest_sha256=sha(SCALE/'outputs/manifest.json'),scope='Same-environment 20-function adapter. Unmodified R181 GAT/PPO vendor; native reward retained; original author system not reproduced.')
 if (ROOT/'manifest.json').exists():assert read(ROOT/'manifest.json')==json.loads(json.dumps(frozen))
 else:write(ROOT/'manifest.json',frozen)
 return frozen
def validation(net,blocks,ws,threshold=.5):
 net.eval();rr=[]
 for b in blocks:
  if b['split']!='validation':continue
  p=b['profile'];w=ws[p['id']]
  with torch.no_grad():plan=G.sample_grl_plan(net,w,CFG,stochastic=False,threshold=threshold)
  for j,l in enumerate(b['initials']):
   arr=B.flow(187700000+p['id']*7);r=execute(p,R.layout(l),arr,plan);rr.append(dict(reward=reward(r,w),mean=r['mean'],p95=r['p95']))
 return dict(reward=st.mean(r['reward'] for r in rr),mean=st.mean(r['mean'] for r in rr),p95=st.mean(r['p95'] for r in rr),rows=len(rr))

def update(net,opt,batch):
    net.train();oldlp=torch.stack([x[1].log_probability.detach() for x in batch]);ret=torch.tensor([x[2] for x in batch]);oldv=torch.stack([x[1].value.detach() for x in batch]);adv=ret-oldv;adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    for epoch in range(4):
        lps=[];ents=[];vs=[]
        for w,plan,r in batch:
            lp,en,v=G._recompute_joint_logp(net,w,plan,CFG);lps.append(lp);ents.append(en);vs.append(v)
        ratio=(torch.stack(lps)-oldlp).exp();loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*(torch.stack(vs)-ret).square().mean()-.01*torch.stack(ents).mean()
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
    return float(loss.detach())
def train(seed):
 torch.set_num_threads(1);dest=ROOT/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'trained.json').exists():return
 m=read(ROOT/'manifest.json');blocks=m['profiles'];states=R.states(m,'train');ws={b['profile']['id']:workflow(b['profile']) for b in blocks if b['split']!='confirmation'}
 cfg=dataclasses.replace(G.GRLTrainConfig(),seed=seed,episodes=BUDGET);torch.manual_seed(seed);net=G.GRLActorCritic(cfg);opt=torch.optim.Adam(net.parameters(),lr=3e-4);calls=0;vals=[];elapsed=0.;binding=sha(ROOT/'manifest.json')
 if (dest/'resume.pt').exists():
  ck=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False);assert ck['binding']==binding;net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);calls=ck['calls'];vals=ck['validation'];elapsed=ck['elapsed']
 elif not (dest/'checkpoint_0.pt').exists():
  vals=[dict(calls=0,**validation(net,blocks,ws))];save(dest/'checkpoint_0.pt',dict(state=net.state_dict(),optimizer=opt.state_dict(),calls=0,validation=vals));print('R188_INITIAL',seed,vals[-1]['mean'],flush=True)
 else:vals=torch.load(dest/'checkpoint_0.pt',map_location='cpu',weights_only=False)['validation']
 tick=time.perf_counter();log=dest/'training.jsonl'
 if log.exists():log.write_text(''.join(line+'\n' for line in log.read_text().splitlines() if json.loads(line)['calls']<=calls))
 while calls<BUDGET:
  boundary=min(p for p in POINTS if p>calls);batch=[]
  for _ in range(min(32,boundary-calls)):
   ep=calls;b,l,j=states[random.Random(187400000+seed*100000+ep).randrange(len(states))];p=b['profile'];w=ws[p['id']];torch.manual_seed(188500000+seed*100000+ep)
   with torch.no_grad():plan=G.sample_grl_plan(net,w,CFG,stochastic=True)
   r=execute(p,R.layout(l),B.flow(187600000+seed*100000+ep),plan);batch.append((w,plan,reward(r,w)));calls+=1
  loss=update(net,opt,batch)
  with log.open('a') as f:f.write(json.dumps(dict(calls=calls,loss=loss,mean_reward=st.mean(x[2] for x in batch)))+'\n')
  if calls==boundary:
   v=validation(net,blocks,ws);vals.append(dict(calls=calls,**v));print('R188',seed,'CALLS',calls,'/',BUDGET,'VALID',v['mean'],flush=True)
  if calls%256==0 or calls==boundary:
   ck=dict(state=net.state_dict(),optimizer=opt.state_dict(),calls=calls,validation=vals,binding=binding,elapsed=elapsed+time.perf_counter()-tick);save(dest/'resume.pt',ck)
   if calls==boundary:save(dest/f'checkpoint_{calls}.pt',ck)
   write(dest/'status.json',dict(phase='training',calls=calls,budget=BUDGET,elapsed=ck['elapsed'],validation=vals))
   if calls%1024==0:print('R188_PROGRESS',seed,calls,'/',BUDGET,flush=True)
 best=max(vals,key=lambda v:(v['reward'],-v['calls']));ck=torch.load(dest/f"checkpoint_{best['calls']}.pt",map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 thresholds=[dict(threshold=t,**validation(net,blocks,ws,t)) for t in cfg.threshold_grid];selected=max(thresholds,key=lambda v:(v['reward'],-v['threshold']))
 save(dest/'selected.pt',dict(state=net.state_dict(),calls=best['calls'],threshold=selected['threshold'],validation=selected));write(dest/'trained.json',dict(seed=seed,calls=BUDGET,selected_checkpoint=best,thresholds=thresholds,selection=selected));print('R188_TRAINED',seed,flush=True)
def evaluate(seed):
 torch.set_num_threads(1);m=read(ROOT/'manifest.json');dest=ROOT/f'seed_{seed}';ck=torch.load(dest/'selected.pt',map_location='cpu',weights_only=False);net=G.GRLActorCritic(dataclasses.replace(G.GRLTrainConfig(),seed=seed));net.load_state_dict(ck['state']);net.eval()
 for num,b in enumerate(x for x in m['profiles'] if x['split']=='confirmation'):
  p=b['profile'];out=dest/f"test_{p['id']}.json"
  if out.exists():continue
  w=workflow(p);tick=time.perf_counter()
  with torch.no_grad():plan=G.sample_grl_plan(net,w,CFG,stochastic=False,threshold=ck['threshold'])
  seconds=time.perf_counter()-tick;rows=[]
  for j,l in enumerate(b['initials']):
   old=R.layout(l)
   for stream in range(3):
    arr=B.flow(187800000+p['id']*7+stream);r=execute(p,old,arr,plan)
    rows.append(dict(seed=seed,profile=p['id'],initial=j,stream=stream,grl=r,grl_objective=-reward(r,w),decision_seconds=seconds,layout=layout(plan,len(p['work'])),fusion=list(plan.fusion_actions),repair=list(plan.repair_edges),fallback=list(plan.placement_fallbacks)))
  write(out,rows);write(dest/'status.json',dict(phase='confirmation',profiles=num+1,total=48))
 write(dest/'result.json',dict(seed=seed,phase='completed',calls=BUDGET));write(dest/'status.json',dict(phase='completed'));print('R188_SEED_DONE',seed,flush=True)
def work(seed):train(seed);evaluate(seed)
def summarize():
 m=read(ROOT/'manifest.json');rows=[];paired=[];done=[]
 for s in SEEDS:
  d=ROOT/f'seed_{s}'
  if not (d/'result.json').exists():continue
  done.append(s)
  for b in m['profiles']:
   if b['split']!='confirmation':continue
   pid=b['profile']['id'];rr=read(d/f'test_{pid}.json');rows.extend(rr);ours=SCALE/'outputs'/f'seed_{s}'/f'test_{pid}.json'
   if not ours.exists():continue
   oo={(r['initial'],r['stream']):r for r in read(ours)}
   for r in rr:
    a=oo[(r['initial'],r['stream'])];paired.append(dict(seed=s,profile=pid,ours=a['system'],grl=r['grl'],whole=a['whole']))
 assert rows
 metrics={k:st.mean(r['grl'][k] for r in rows) for k in ('mean','p95','deployment','drain','changes','compute_cost')};comparison={}
 if paired:
  for a in ('ours','grl','whole'):comparison[a]={k:st.mean(r[a][k] for r in paired) for k in ('mean','p95','deployment','drain','changes')}
  comparison['ours_improvement_pct']=100*(1-comparison['ours']['mean']/comparison['grl']['mean']);comparison['paired_records']=len(paired);comparison['paired_seeds']=sorted({r['seed'] for r in paired})
 write(ROOT/'summary.json',dict(completed_grl_seeds=done,grl_records=len(rows),grl_metrics=metrics,paired_comparison=comparison,scope='Same R187 inputs; unchanged GRL adapter, original composite reward. Full comparison pending if R187 matching outputs unavailable.'))
def preflight():
 torch.set_num_threads(1);p=R.raw_profile(0);w=workflow(p);old=(tuple([0]*20),(0,));torch.manual_seed(601);net=G.GRLActorCritic(dataclasses.replace(G.GRLTrainConfig(),seed=601));before={k:v.detach().clone() for k,v in net.state_dict().items()};batch=[];start=time.perf_counter()
 for j in range(6):
  p=R.raw_profile(j%3);w=workflow(p)
  with torch.no_grad():plan=G.sample_grl_plan(net,w,CFG,stochastic=True)
  lp,_,_=G._recompute_joint_logp(net,w,plan,CFG);assert abs(float(lp.detach()-plan.log_probability))<1e-5
  rr=execute(p,old,B.flow(1883300+j),plan);assert rr['completed']>0 and rr['peak_ram_fraction']<=1+1e-9;batch.append((w,plan,reward(rr,w)))
 update(net,torch.optim.Adam(net.parameters(),lr=3e-4),batch);assert any(not torch.equal(v,net.state_dict()[k]) for k,v in before.items())
 for name in ('r8_grl_orch.py','r8_policy_adapters.py'):assert sha(HERE/'vendor'/name)==sha(HERE.parent/'r181_grl_matched_comparison/vendor'/name)
 return dict(status='passed',n_functions=20,vendor_unchanged_from_R181=True,joint_logp_replay_exact=True,parameters_updated=True,legal_plans=True,seconds=time.perf_counter()-start)
def main():
 args=argparse.ArgumentParser();args.add_argument('--smoke',action='store_true');args.add_argument('--summarize',action='store_true');args=args.parse_args();ROOT.mkdir(exist_ok=True)
 try:
  if args.smoke:info=preflight();write(ROOT/'preflight.json',info);print(json.dumps(info));return
  if args.summarize:summarize();return
  m=prepare();assert read(ROOT/'preflight.json')['status']=='passed';assert all(sha(Path(p))==v for p,v in m['bindings'].items());write(ROOT/'status.json',dict(phase='training',workers=2,seeds=SEEDS,budget=BUDGET))
  with cf.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:list(pool.map(work,SEEDS))
  summarize();assert all(sha(Path(p))==v for p,v in m['bindings'].items());write(ROOT/'status.json',dict(phase='completed_pending_analysis'));print('R188_ALL_FINISHED',flush=True)
 except Exception:write(ROOT/'status.json',dict(phase='failed',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
