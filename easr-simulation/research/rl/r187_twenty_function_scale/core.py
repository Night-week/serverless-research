"""Variable-N model utilities; same execution/transfer/deployment equations as frozen four-function core."""
import itertools as it,random,math

def groups(m):return [[i for i,g in enumerate(m) if g==k] for k in sorted(set(m))]
def canonical(parts,n):
 m=[None]*n;z=[]
 for g,(ns,s) in enumerate(sorted(parts,key=lambda a:min(a[0]))):
  z.append(s)
  for i in ns:assert m[i] is None;m[i]=g
 assert all(x is not None for x in m)
 return tuple(m),tuple(z)
def valid(p,m,z):
 if len(m)!=len(p['work']) or sorted(set(m))!=list(range(len(z))) or any(s not in (0,1) for s in z):return False
 gs=groups(m);used=[0.,0.]
 for g,ns in enumerate(gs):used[z[g]]+=.3*sum(p['mem'][i] for i in ns)+.7*max(p['mem'][i] for i in ns)
 if any(a>b+1e-9 for a,b in zip(used,p['ram'])):return False
 edges={(m[u],m[v]) for u,v,_ in p['edges'] if m[u]!=m[v]};done=set()
 while len(done)<len(gs):
  ready=[g for g in range(len(gs)) if g not in done and all(u in done for u,v in edges if v==g)]
  if not ready:return False
  done.update(ready)
 return True
def flow(seed):
 rng=random.Random(seed);t=0.;arr=[]
 while t<60:
  end=min(60,t+rng.uniform(6,18));rate=rng.choice([2.,5.,10.]);now=t
  while True:
   now+=rng.expovariate(rate)
   if now>=end:break
   arr.append(now)
  t=end
 return arr
def cost(p,old,new):
 if old==new:return 0.
 oldobjects={(tuple(ns),old[1][g]) for g,ns in enumerate(groups(old[0]))}
 fresh=[(ns,new[1][g]) for g,ns in enumerate(groups(new[0])) if (tuple(ns),new[1][g]) not in oldobjects]
 return .08+.04*len(fresh)+sum(sum(p['mem'][i] for i in ns)*.05*8/p['mbps'] for ns,s in fresh if s==1)
def vector(p,l):
 m,z=l;n=len(m);total=sum(p['work'])
 return [float(m[u]==m[v]) for u,v in it.combinations(range(n),2)]+[float(z[m[i]]==0) for i in range(n)]+[len(z)/n]+[sum(p['work'][i] for i in range(n) if z[m[i]]==s)/total for s in range(2)]
def neighbors(p,old):
 """All host moves; all cuts for groups<=8, bounded topology-order cuts for larger groups."""
 n=len(p['work']);gs=groups(old[0]);found={old}
 for g,ns in enumerate(gs):
  other=[(v,old[1][j]) for j,v in enumerate(gs) if j!=g]
  found.add(canonical(other+[(ns,1-old[1][g])],n))
  if len(ns)<2:continue
  if len(ns)<=8:
   cuts=[([v for j,v in enumerate(ns) if mask>>j&1],[v for j,v in enumerate(ns) if not mask>>j&1]) for mask in range(1,2**len(ns)-1) if mask&1]
  else:cuts=[(ns[:k],ns[k:]) for k in sorted({1,len(ns)//4,len(ns)//3,len(ns)//2,2*len(ns)//3,3*len(ns)//4,len(ns)-1}) if 0<k<len(ns)]
  for left,right in cuts:
   for x,y in it.product(range(2),repeat=2):found.add(canonical(other+[(left,x),(right,y)],n))
 return [old]+sorted(l for l in found if l!=old and valid(p,*l))
def layout_from_cuts(p,cuts,sites):
 n=len(p['work']);bounds=[0]+sorted(set(cuts))+[n]
 return canonical([(list(range(a,b)),sites[j]) for j,(a,b) in enumerate(zip(bounds,bounds[1:]))],n)
def plan_pool(p,seed,limit=32):
 n=len(p['work']);rng=random.Random(seed);pool=set()
 for size in (1,2,4,5,10,n):
  cuts=list(range(size,n,size))
  for s in (0,1):
   l=layout_from_cuts(p,cuts,[s]*(len(cuts)+1))
   if valid(p,*l):pool.add(l)
 attempts=0
 while len(pool)<limit and attempts<limit*20:
  attempts+=1;k=rng.randint(1,min(n,10));cuts=sorted(rng.sample(range(1,n),k-1));l=layout_from_cuts(p,cuts,[rng.randrange(2) for _ in range(k)])
  if valid(p,*l):pool.add(l)
 assert pool
 return sorted(pool)
