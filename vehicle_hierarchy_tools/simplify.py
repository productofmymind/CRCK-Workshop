"""CRCK Workshop' mesh simplifier: greedy edge collapse ordered by quadric error (Garland and Heckbert 1997). Pure Python and numpy.

The mesh is reduced on welded positions, so pieces that a game mesh splits at UV seams, hard edges or materials stay joined (no cracks). Each corner
has a "wedge": a row of attributes (UVs, colours, bone weights, normal). Collapsing u into v removes u.

Where both ends are plain surface (no feature line, no seam, not locked) the merged vertex may slide along the edge to the point of least quadric
error, and its wedge becomes the matching blend of the two wedges. Elsewhere v stays where it is with its wedges unchanged (half-edge collapse), so
seams and lines keep their exact original vertices and attributes.

Rules that keep the result clean:
  * Feature lines (open borders and material borders) are kept: a vertex on one can only slide along it, and its ends are fixed.
    Plane quadrics perpendicular to each feature edge keep the line's shape. A material never loses its last triangle.
  * UV, colour and normal seams are soft: a vertex may leave a seam, and the corners that then take another wedge cost their area times the
    attribute jump (attributes are scaled to metres), so seams that matter are kept and cheap ones give way at low levels.
  * A collapse is refused when it would fold or flip a triangle, make a needle triangle, or change the topology (link condition).
  * Cost = quadric error (area weighted) + attribute cost + a small preference for short edges + a penalty for thin triangles.
  * When nothing is left to collapse above a target, the rules relax in two steps (line ends and given locks may move; then material borders
    are soft), so only the levels that need it are affected.
One run produces every level: the mesh is saved each time the triangle count reaches the next target.
"""
import heapq
import math

import numpy as np

FEATURE_WEIGHT = 8.0     # feature-edge plane quadrics, relative to face quadrics (weight = length squared)
SEAM_WEIGHT = 2.0        # the same for soft seams
LENGTH_WEIGHT = 0.02     # prefer short edges: cost += LENGTH_WEIGHT * length^2 * (area around both ends)
GOOD_QUALITY = 0.35      # triangles below this shape quality (1 = equilateral, 0 = needle) add a penalty
QUALITY_WEIGHT = 1.0
MIN_QUALITY = 0.02       # refuse to make triangles thinner than this (unless the triangle was already that thin)
FLIP_DOT = 0.25          # refuse to turn a triangle's normal by more than about 75 degrees
ATTR_WEIGHT = 1.0        # attribute jump cost: ATTR_WEIGHT * corner area * (attribute distance in metres)^2
MAX_RELAX = 2            # how far the rules may relax when a target is out of reach (0: never)
SQ3x4 = 4.0 * math.sqrt(3.0)


def _qerr(q, x, y, z):
    return (q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y
            + q[7] * z * z + 2 * q[8] * z + q[9])


def _solve(q):
    """Point of least quadric error, or None when the quadric is close to singular (flat or straight regions)."""
    a, b, c, d, e, f = q[0], q[1], q[2], q[4], q[5], q[7]
    det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c)
    tr = a + d + f
    if tr <= 0 or abs(det) < 1e-9 * tr * tr * tr:
        return None
    r0, r1, r2 = -q[3], -q[6], -q[8]
    x = (r0 * (d * f - e * e) - b * (r1 * f - e * r2) + c * (r1 * e - d * r2)) / det
    y = (a * (r1 * f - e * r2) - r0 * (b * f - e * c) + c * (b * r2 - r1 * c)) / det
    z = (a * (d * r2 - r1 * e) - b * (b * r2 - r1 * c) + r0 * (b * e - d * c)) / det
    return x, y, z


def _tri_check(pa, pb, pc, f):
    """(double area, quality) of a triangle, or None when it is degenerate or turned too far from the unit normal f."""
    e1x, e1y, e1z = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
    e2x, e2y, e2z = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
    nx, ny, nz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
    dbl = math.sqrt(nx * nx + ny * ny + nz * nz)
    if dbl <= 1e-30 or (nx * f[0] + ny * f[1] + nz * f[2]) / dbl < FLIP_DOT:
        return None
    e3x, e3y, e3z = pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]
    s = (e1x * e1x + e1y * e1y + e1z * e1z) + (e2x * e2x + e2y * e2y + e2z * e2z) + (e3x * e3x + e3y * e3y + e3z * e3z)
    return dbl, (SQ3x4 * 0.5 * dbl / s if s > 0 else 0.0)


class Simplifier:
    """pos: (P, 3) welded positions. tris: (T, 3) position ids. wedges: (T, 3) wedge ids per corner. mats: (T,) material per triangle.
    locked: (P,) bool, vertices that must stay. attrs: (W, k) wedge attributes; scale: (k,) metres per unit of each column (both None: seams are
    hard and vertices never move)."""

    def __init__(self, pos, tris, wedges, mats, locked=None, attrs=None, scale=None):
        pos = np.asarray(pos, np.float64)
        tris = np.asarray(tris, np.int64).reshape(-1, 3)
        wedges = np.asarray(wedges, np.int64).reshape(-1, 3)
        mats = np.asarray(mats, np.int64)
        keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])
        tris, wedges, mats = tris[keep], wedges[keep], mats[keep]
        P, T = len(pos), len(tris)
        lock = np.zeros(P, bool) if locked is None else np.asarray(locked, bool).copy()

        a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
        cr = np.cross(b - a, c - a)
        dbl = np.linalg.norm(cr, axis=1)
        n = cr / np.maximum(dbl, 1e-30)[:, None]
        Q = np.zeros((P, 11))
        fq = _quadrics(n, -(n * a).sum(1), dbl * 0.5)
        for k in range(3):
            np.add.at(Q, tris[:, k], fq)

        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        et = np.tile(np.arange(T), 3)
        lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
        order = np.lexsort((hi, lo))
        lo, hi, et = lo[order], hi[order], et[order]
        start = np.flatnonzero(np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])]) if len(lo) else np.zeros(0, np.int64)
        count = np.diff(np.r_[start, len(lo)])
        feature = count == 1                              # open border
        many = count > 2                                  # non-manifold: both ends stay
        hard = np.zeros(P, bool)
        hard[lo[start[many]]] = True
        hard[hi[start[many]]] = True
        two = np.flatnonzero(count == 2)
        t1, t2 = et[start[two]], et[start[two] + 1]
        el, eh = lo[start[two]], hi[start[two]]
        seam = np.zeros(len(start), bool)
        seam[two] = ((_wedge_at(tris, wedges, t1, el) != _wedge_at(tris, wedges, t2, el))
                     | (_wedge_at(tris, wedges, t1, eh) != _wedge_at(tris, wedges, t2, eh)))
        feature[two] = mats[t1] != mats[t2]
        if attrs is None:
            feature |= seam
        fs = np.flatnonzero(feature)
        fl, fh = lo[start[fs]], hi[start[fs]]
        for edges, weight in ((fs, FEATURE_WEIGHT), (np.flatnonzero(seam & ~feature), SEAM_WEIGHT)):
            for off in (0, 1):
                sel = edges[count[edges] > off]
                if not len(sel):
                    continue
                t = et[start[sel] + off]
                pl, ph = pos[lo[start[sel]]], pos[hi[start[sel]]]
                ed = ph - pl
                m = np.cross(ed, n[t])
                ml = np.linalg.norm(m, axis=1)
                ok = ml > 1e-30
                m = m[ok] / ml[ok][:, None]
                q = _quadrics(m, -(m * pl[ok]).sum(1), weight * (ed[ok] ** 2).sum(1))
                np.add.at(Q, lo[start[sel]][ok], q)
                np.add.at(Q, hi[start[sel]][ok], q)
        deg = np.bincount(np.r_[fl, fh], minlength=P)
        lock |= (deg == 1) | (deg > 2) | hard              # ends and junctions of feature lines stay (until relaxed)

        self.P = [tuple(p) for p in pos.tolist()]
        self.Q = Q.tolist()
        self.T = tris.tolist()
        self.W = wedges.tolist()
        self.M = mats.tolist()
        self.mcount = {}
        for m in self.M:
            self.mcount[m] = self.mcount.get(m, 0) + 1
        if attrs is None:
            self.R = self.A = self.S = None
        else:
            raw = np.asarray(attrs, np.float64)
            sc = np.asarray(scale, np.float64)
            self.S = sc.tolist()
            self.R = raw.tolist()                                   # wedge attributes; blended wedges are appended
            self.A = [tuple(r) for r in (raw * sc).tolist()]        # the same in metres
        self.fn = [tuple(v) for v in n.tolist()]
        self.alive = [True] * T
        self.count = T
        self.vt = [set() for _ in range(P)]
        for i, (x, y, z) in enumerate(self.T):
            self.vt[x].add(i)
            self.vt[y].add(i)
            self.vt[z].add(i)
        self.feat = set(zip(fl.tolist(), fh.tolist()))
        self.locked = lock.tolist()
        self.hard = hard.tolist()
        self.relax = 0  # 1: line ends and given locks may move; 2: material borders are soft too (only open borders stay lines)
        self.dead = [False] * P
        self.heap = []
        self.stamp = [0] * P  # changes whenever the triangles around a vertex change: queue entries with older stamps are stale
        self.pushed = {}      # (u, v) -> stamps of its newest queue entry, so a stale entry is re-queued once, not every time it comes up

    # --- costs ---------------------------------------------------------------------------

    def _base_cost(self, u, v):
        """Cheap first estimate (v stays, no checks); evaluate() gives the real cost when the entry comes up."""
        qu, qv = self.Q[u], self.Q[v]
        x, y, z = self.P[v]
        q = [qu[i] + qv[i] for i in range(11)]
        pu = self.P[u]
        l2 = (pu[0] - x) ** 2 + (pu[1] - y) ** 2 + (pu[2] - z) ** 2
        return max(_qerr(q, x, y, z), 0.0) + LENGTH_WEIGHT * l2 * (qu[10] + qv[10])

    def _single_wedge(self, x):
        T, W = self.T, self.W
        w = None
        for t in self.vt[x]:
            wx = W[t][T[t].index(x)]
            if w is None:
                w = wx
            elif wx != w:
                return None
        return w

    def evaluate(self, u, v):
        """(cost, wedge map, move) for collapsing u into v, or None when it is not allowed. move is None (v stays) or (new position, blend
        factor from v towards u)."""
        if self.dead[u] or self.dead[v] or self.locked[u] and (self.relax == 0 or self.hard[u]):
            return None
        T, W = self.T, self.W
        shared, others = [], []
        nu = set()
        for t in self.vt[u]:
            tri = T[t]
            nu.update(tri)
            (shared if v in tri else others).append(t)
        if not shared:
            return None
        nu.discard(u)
        feat = self.feat
        fdeg, along = 0, False
        for x in nu:
            if ((u, x) if u < x else (x, u)) in feat:
                fdeg += 1
                if x == v:
                    along = True
        if fdeg and not along or fdeg and fdeg != 2 and self.relax == 0:
            return None
        nv = set()
        for t in self.vt[v]:
            nv.update(T[t])
        nv.discard(v)
        opp = set()
        for t in shared:
            for x in T[t]:
                if x != u and x != v:
                    opp.add(x)
        if (nu & nv) != opp:
            return None
        M, mcount = self.M, self.mcount
        if len(shared) == 1:
            if mcount[M[shared[0]]] <= 1:
                return None
        elif M[shared[0]] == M[shared[1]] and mcount[M[shared[0]]] <= 2 or any(mcount[M[t]] <= 1 for t in shared):
            return None
        wmap = {}
        for t in shared:
            tri, w = T[t], W[t]
            wmap[w[tri.index(u)]] = w[tri.index(v)]

        P, A = self.P, self.A
        pu, pv = P[u], P[v]
        qu, qv = self.Q[u], self.Q[v]
        q = [qu[i] + qv[i] for i in range(11)]
        move, target = None, pv
        if (A is not None and not fdeg and not self.locked[u] and not self.locked[v]
                and not any(((v, x) if v < x else (x, v)) in feat for x in nv)
                and self._single_wedge(u) is not None and self._single_wedge(v) is not None):
            # both ends are plain surface: slide the merged vertex along the edge to the least quadric error
            dx, dy, dz = pu[0] - pv[0], pu[1] - pv[1], pu[2] - pv[2]
            ax = q[0] * dx + q[1] * dy + q[2] * dz
            ay = q[1] * dx + q[4] * dy + q[5] * dz
            az = q[2] * dx + q[5] * dy + q[7] * dz
            dad = ax * dx + ay * dy + az * dz
            if dad > 1e-30:
                g = ((q[0] * pv[0] + q[1] * pv[1] + q[2] * pv[2] + q[3]) * dx + (q[1] * pv[0] + q[4] * pv[1] + q[5] * pv[2] + q[6]) * dy
                     + (q[2] * pv[0] + q[5] * pv[1] + q[7] * pv[2] + q[8]) * dz)
                k = min(1.0, max(0.0, -g / dad))
                if k > 1e-3:
                    target = (pv[0] + k * dx, pv[1] + k * dy, pv[2] + k * dz)
                    move = (target, k)
                best = _qerr(q, *target)
                opt = _solve(q)  # the point of least error in space, if it is well defined and stays near the edge
                if opt is not None:
                    l2e = dx * dx + dy * dy + dz * dz
                    mx, my, mz = opt[0] - (pv[0] + pu[0]) * 0.5, opt[1] - (pv[1] + pu[1]) * 0.5, opt[2] - (pv[2] + pu[2]) * 0.5
                    if mx * mx + my * my + mz * mz <= l2e and _qerr(q, *opt) < best:
                        k = min(1.0, max(0.0, ((opt[0] - pv[0]) * dx + (opt[1] - pv[1]) * dy + (opt[2] - pv[2]) * dz) / l2e))
                        target = opt
                        move = (target, k)

        jump = {}
        fn = self.fn
        minq = 1.0
        attr_cost = 0.0
        for t in others:
            tri = T[t]
            i = tri.index(u)
            wu = W[t][i]
            if wu not in wmap:
                if A is None:
                    return None
                au, best, bd = A[wu], None, 0.0
                for tv in self.vt[v]:
                    wv = W[tv][T[tv].index(v)]
                    av = A[wv]
                    dd = 0.0
                    for j in range(len(au)):
                        dd += (au[j] - av[j]) ** 2
                    if best is None or dd < bd:
                        best, bd = wv, dd
                if best is None:
                    return None
                wmap[wu], jump[wu] = best, bd
            r = _tri_check(target if i == 0 else P[tri[0]], target if i == 1 else P[tri[1]], target if i == 2 else P[tri[2]], fn[t])
            if r is None or r[1] < MIN_QUALITY and r[1] < self._quality(t):
                return None
            minq = min(minq, r[1])
            if wu in jump:
                attr_cost += 0.5 * r[0] * jump[wu]
        if move is not None:  # v moves too: its own triangles must stay valid
            for t in self.vt[v]:
                if t in shared:
                    continue
                tri = T[t]
                i = tri.index(v)
                r = _tri_check(target if i == 0 else P[tri[0]], target if i == 1 else P[tri[1]], target if i == 2 else P[tri[2]], fn[t])
                if r is None or r[1] < MIN_QUALITY and r[1] < self._quality(t):
                    return None
                minq = min(minq, r[1])
        l2 = (pu[0] - pv[0]) ** 2 + (pu[1] - pv[1]) ** 2 + (pu[2] - pv[2]) ** 2
        scale = l2 * (qu[10] + qv[10])
        cost = max(_qerr(q, *target), 0.0) + LENGTH_WEIGHT * scale + ATTR_WEIGHT * attr_cost
        if minq < GOOD_QUALITY:
            cost += QUALITY_WEIGHT * (GOOD_QUALITY - minq) / GOOD_QUALITY * scale
        return cost, wmap, move

    def _quality(self, t):
        P = self.P
        a, b, c = (P[i] for i in self.T[t])
        e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        e3 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
        nx, ny, nz = e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]
        s = sum(x * x for x in e1) + sum(x * x for x in e2) + sum(x * x for x in e3)
        return SQ3x4 * 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz) / s if s > 0 else 0.0

    # --- collapsing ----------------------------------------------------------------------

    def _movable(self, x):
        return not self.locked[x] or self.relax > 0 and not self.hard[x]

    def _push_around(self, v):
        nv = set()
        for t in self.vt[v]:
            nv.update(self.T[t])
        nv.discard(v)
        heap, st, pushed = self.heap, self.stamp, self.pushed
        for x in nv:
            if self._movable(v):
                heapq.heappush(heap, (self._base_cost(v, x), v, x, st[v], st[x]))
                pushed[(v, x)] = (st[v], st[x])
            if self._movable(x):
                heapq.heappush(heap, (self._base_cost(x, v), x, v, st[x], st[v]))
                pushed[(x, v)] = (st[x], st[v])

    def _normal(self, t):
        P, tri = self.P, self.T[t]
        pa, pb, pc = P[tri[0]], P[tri[1]], P[tri[2]]
        e1 = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        e2 = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        nx, ny, nz = e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]
        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        return nx / ln, ny / ln, nz / ln

    def _collapse(self, u, v, wmap, move):
        T, W = self.T, self.W
        wu_single = self._single_wedge(u) if move else None
        wv_single = self._single_wedge(v) if move else None
        nu = set()
        for t in self.vt[u]:
            nu.update(T[t])
        nu.discard(u)
        for t in list(self.vt[u]):
            tri = T[t]
            if v in tri:
                self.alive[t] = False
                self.count -= 1
                self.mcount[self.M[t]] -= 1
                for x in tri:
                    if x != u:
                        self.vt[x].discard(t)
            else:
                i = tri.index(u)
                tri[i] = v
                W[t][i] = wmap[W[t][i]]
                self.vt[v].add(t)
                self.fn[t] = self._normal(t)
        self.vt[u] = set()
        self.dead[u] = True
        qu, qv = self.Q[u], self.Q[v]
        for i in range(11):
            qv[i] += qu[i]
        if move:
            pos, k = move
            self.P[v] = pos
            ru, rv = self.R[wu_single], self.R[wv_single]
            blend = [rv[j] + k * (ru[j] - rv[j]) for j in range(len(rv))]
            nw = len(self.R)
            self.R.append(blend)
            self.A.append(tuple(blend[j] * self.S[j] for j in range(len(blend))))
            for t in self.vt[v]:
                W[t][T[t].index(v)] = nw
                self.fn[t] = self._normal(t)
        st = self.stamp
        st[v] += 1
        for t in self.vt[v]:
            for x in T[t]:
                if x != v:
                    st[x] += 1
        feat = self.feat
        for x in nu:
            key = (u, x) if u < x else (x, u)
            if key in feat:
                feat.discard(key)
                if x != v:
                    feat.add((v, x) if v < x else (x, v))
        self._push_around(v)

    def _fill_heap(self):
        self.heap = []
        self.pushed = {}
        st = self.stamp
        seen = set()
        for t, tri in enumerate(self.T):
            if not self.alive[t]:
                continue
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                if self._movable(a):
                    self.heap.append((self._base_cost(a, b), a, b, st[a], st[b]))
                if self._movable(b):
                    self.heap.append((self._base_cost(b, a), b, a, st[b], st[a]))
        heapq.heapify(self.heap)

    def _soften_materials(self):
        """Relax level 2: only open borders stay feature lines."""
        T = self.T
        self.feat = {(a, b) for a, b in self.feat if sum(1 for t in self.vt[a] if b in T[t]) == 1}

    def snapshot(self):
        """(wedge triangles, position-id triangles, materials, positions now) of the current mesh."""
        idx = [t for t in range(len(self.T)) if self.alive[t]]
        if not idx:
            z = np.zeros((0, 3), np.int64)
            return z, z, np.zeros(0, np.int64), np.zeros((0, 3))
        return (np.array([self.W[t] for t in idx], np.int64), np.array([self.T[t] for t in idx], np.int64),
                np.array([self.M[t] for t in idx], np.int64), np.array(self.P, np.float64))

    def attributes(self):
        """(W', k) attribute rows of every wedge, the blended ones included."""
        return None if self.R is None else np.array(self.R, np.float64)

    def run(self, targets):
        """Reduce to each triangle count in `targets` (any order); returns {target: snapshot()}. A target that cannot be reached gets the
        smallest mesh possible."""
        out = {}
        pending = sorted(set(int(t) for t in targets), reverse=True)
        while pending and self.count <= pending[0]:
            out[pending.pop(0)] = self.snapshot()
        while pending:
            self._fill_heap()
            collapsed = 0
            heap, st, pushed = self.heap, self.stamp, self.pushed
            while heap and pending:
                cost, u, v, su, sv = heapq.heappop(heap)
                if self.dead[u] or self.dead[v]:
                    continue
                if st[u] != su or st[v] != sv:  # the surface around it changed since it was queued
                    now = (st[u], st[v])
                    if pushed.get((u, v)) != now:
                        pushed[(u, v)] = now
                        heapq.heappush(heap, (self._base_cost(u, v), u, v, now[0], now[1]))
                    continue
                r = self.evaluate(u, v)
                if r is None:
                    continue
                if r[0] > cost * (1 + 1e-9) + 1e-24:
                    heapq.heappush(heap, (r[0], u, v, su, sv))
                    continue
                self._collapse(u, v, r[1], r[2])
                collapsed += 1
                while pending and self.count <= pending[0]:
                    out[pending.pop(0)] = self.snapshot()
            if not collapsed:
                if self.relax >= MAX_RELAX:
                    break
                self.relax += 1
                if self.relax == 2:
                    self._soften_materials()
        for tgt in pending:
            out[tgt] = self.snapshot()
        return out


def _wedge_at(tris, wedges, t, p):
    """Wedge of triangle t at position id p (vectorised)."""
    col = np.argmax(tris[t] == p[:, None], axis=1)
    return wedges[t, col]


def _quadrics(n, d, w):
    a, b, c = n[:, 0], n[:, 1], n[:, 2]
    return np.stack([w * a * a, w * a * b, w * a * c, w * a * d, w * b * b, w * b * c, w * b * d, w * c * c, w * c * d, w * d * d, w], axis=1)
