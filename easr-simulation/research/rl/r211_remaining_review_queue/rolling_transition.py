"""Shared rolling-transition semantics used by the R218 evaluator.

The standalone response-delay sensitivity runner and its delay variants are
intentionally excluded. R218 supplies its single frozen decision-latency
estimate to ``local`` as an argument.
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parent
sys.path.insert(0, str(RL / 'r199_live_snapshot_review'))
import run as X  # noqa: E402

R = X.R
B = X.B
torch = R.torch
Base = R.Engine

class Rolling(Base):
 def change(self,new):
  if new==self.layout:return
  objects={}
  def add(gs,z):
   for g,ns in enumerate(gs):objects[tuple(ns),z[g]]=.3*sum(self.p['mem'][i] for i in ns)+.7*max(self.p['mem'][i] for i in ns)
  add(B.groups(self.layout[0]),self.layout[1]);add(B.groups(new[0]),new[1])
  for job in self.jobs.values():add(job['gs'],job['z'])
  used=[sum(v for (ns,s),v in objects.items() if s==tier) for tier in [0,1]]
  if any(v>c+1e-9 for v,c in zip(used,self.p['ram'])):
   self.rolling_fallbacks=getattr(self,'rolling_fallbacks',0)+1;return super().change(new)
  old=self.layout;start=self.now;cost=B.cost(self.p,old,new);self.advance(start+cost,hold=True)
  self.layout=new;self.visited.add(new);self.changes+=1;self.deployment_total+=cost;self.check_ram();self.peak_ram_fraction=max(self.peak_ram_fraction,*[v/c for v,c in zip(used,self.p['ram'])]);queued=self.backlog;self.backlog=[]
  for r in queued:self.admit(r,self.now)
  self.rolling_changes=getattr(self,'rolling_changes',0)+1
  self.actions.append(dict(start=start,drained=start,ready=self.now,deployment=cost,drain=0.,old=old,new=new,queued=len(queued),version_overlap=True))
 def finish(self):
  r=super().finish();r.update(rolling_changes=getattr(self,'rolling_changes',0),rolling_fallbacks=getattr(self,'rolling_fallbacks',0));return r

def local(e,net,t0,pending,first,delay):
 indices=[]
 for step in range(2):
  pool=[e.layout]+[l for l in B.neighbors(e.p,e.layout)[1:] if l not in e.visited]
  if step==0:idx=first;assert idx<len(pool)
  else:
   with torch.no_grad():idx=int(net(torch.tensor(R.observe(e,pool,step),dtype=torch.float32))[0].argmax())
   e.advance(e.now+delay)
  indices.append(idx)
  if idx==0:break
  e.change(pool[idx])
  if step==0:
   e.advance(max(t0+15,e.now))
   if e.now>=60:break
 result=X.finish(e,t0,pending);result['indices']=indices;return result
