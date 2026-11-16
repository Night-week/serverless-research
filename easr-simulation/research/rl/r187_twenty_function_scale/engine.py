"""Frozen graph/Engine extracted verbatim from R151; dynamic-N core B supplied separately."""
import heapq,itertools as it,math,statistics as st
import core as B

def graph(p, l):
    m, z = l; gs = B.groups(m)
    pred = [set() for _ in gs]; succ = [set() for _ in gs]; data = {}
    for u, v, b in p['edges']:
        a, c = m[u], m[v]
        if a != c:
            pred[c].add(a); succ[a].add(c)
            data[a, c] = data.get((a, c), 0.) + b
    return gs, z, pred, succ, data

class Engine:
    def __init__(self, p, layout, arrivals):
        self.p = p; self.layout = layout; self.arrivals = arrivals
        self.now = 0.; self.serial = it.count(); self.events = []
        self.cores = [[0.] * c for c in p['cores']]; self.links = [0., 0.]
        self.jobs = {}; self.active = 0; self.backlog = []; self.responses = []
        self.task_intervals = []; self.link_intervals = []; self.admitted = []
        self.actions = []; self.deployment_total = 0.; self.drain_total = 0.
        self.visited = {layout}; self.changes = 0; self.peak_ram_fraction = 0.
        self.check_ram()
        for r, t in enumerate(arrivals):
            self.push(t, 'arrival', r, -1)

    def check_ram(self):
        used = [0., 0.]
        for g, ns in enumerate(B.groups(self.layout[0])):
            used[self.layout[1][g]] += .3*sum(self.p['mem'][n] for n in ns)+.7*max(self.p['mem'][n] for n in ns)
        assert B.valid(self.p, *self.layout)
        self.peak_ram_fraction = max(self.peak_ram_fraction, *(v/c for v, c in zip(used, self.p['ram'])))
        assert self.peak_ram_fraction <= 1. + 1e-9

    def push(self, t, kind, r, g):
        heapq.heappush(self.events, (t, next(self.serial), kind, r, g))

    def admit(self, r, t):
        gs, z, pred, succ, data = graph(self.p, self.layout)
        self.jobs[r] = dict(gs=gs, z=z, succ=succ, data=data,
                            remaining=[len(v) for v in pred], release=[0.]*len(gs),
                            sinks=sum(not v for v in succ), completed_sinks=0, last=0.)
        self.active += 1; self.admitted.append((r, t))
        for g in range(len(gs)):
            if not pred[g]:
                self.push(t, 'ready', r, g)

    def event(self, e, hold=False):
        t, _, kind, r, g = e; self.now = max(self.now, t)
        if kind == 'arrival':
            if hold:
                self.backlog.append(r)
            else:
                self.admit(r, t)
            return
        job = self.jobs[r]; z = job['z']
        if kind == 'ready':
            site = z[g]; k = min(range(len(self.cores[site])), key=lambda j: self.cores[site][j])
            begin = max(t, self.cores[site][k]); end = begin + sum(self.p['work'][n] for n in job['gs'][g])/self.p['speed'][site]
            self.cores[site][k] = end; self.task_intervals.append((site, begin, end))
            self.push(end, 'done', r, g)
        else:
            job['last'] = max(job['last'], t)
            for h in sorted(job['succ'][g]):
                ready = t
                if z[g] != z[h]:
                    d = z[g]; begin = max(t, self.links[d])
                    self.links[d] = begin + job['data'][g, h]*8/self.p['mbps']
                    self.link_intervals.append((d, begin, self.links[d]))
                    ready = self.links[d]+self.p['rtt']
                else:
                    ready += .003
                job['release'][h] = max(job['release'][h], ready)
                job['remaining'][h] -= 1
                if job['remaining'][h] == 0:
                    self.push(job['release'][h], 'ready', r, h)
            if not job['succ'][g]:
                job['completed_sinks'] += 1
                if job['completed_sinks'] == job['sinks']:
                    self.responses.append((r, job['last']-self.arrivals[r]))
                    self.active -= 1
                    del self.jobs[r]

    def advance(self, t, hold=False):
        while self.events and self.events[0][0] <= t:
            self.event(heapq.heappop(self.events), hold)
        self.now = max(self.now, t)

    def change(self, new):
        if new == self.layout:
            return
        before = self.layout; start = self.now
        core_snapshot = [list(v) for v in self.cores]
        while self.active:
            assert self.events
            self.event(heapq.heappop(self.events), True)
        drained = self.now
        cost = B.cost(self.p, before, new)
        ready = drained + cost
        self.advance(ready, hold=True)
        self.layout = new; self.check_ram(); self.visited.add(new); self.changes += 1
        self.deployment_total += cost; self.drain_total += drained-start
        # Hardware timelines persist. Never reset core/link occupancy.
        assert all(b >= a for xs, ys in zip(core_snapshot, self.cores) for a, b in zip(xs, ys))
        queued = self.backlog; self.backlog = []
        for r in queued:
            self.admit(r, ready)
        self.actions.append(dict(start=start, drained=drained, ready=ready,
                                 deployment=cost, drain=drained-start, old=before, new=new,
                                 queued=len(queued)))

    def finish(self):
        while self.events:
            self.event(heapq.heappop(self.events))
        assert self.active == 0 and not self.backlog
        assert len(self.responses) == len(self.arrivals)
        assert len({r for r, v in self.responses}) == len(self.arrivals)
        vals = sorted(v for r, v in self.responses)
        return dict(mean=st.mean(vals), p95=vals[math.ceil(.95*len(vals))-1],
                    changes=self.changes, deployment=self.deployment_total, drain=self.drain_total,
                    completed=len(vals), peak_ram_fraction=self.peak_ram_fraction,
                    actions=self.actions)
