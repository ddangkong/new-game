"""Procedural vegetation: branching trees with real leaf geometry, grass clumps,
wildflowers, ferns, reeds, dandelion clocks and the player's seed.

Every generator returns an MB (mesh builder) with material slots documented
per function; build() it with the matching materials."""
import math

import numpy as np

from blx import MB, ico, perp_frame, tube
from world import fbm

Z = np.array([0.0, 0.0, 1.0])


def _n(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def rot_dir(t, angle, azim):
    u, w = perp_frame(t)
    return _n(math.cos(angle) * t + math.sin(angle) * (math.cos(azim) * u + math.sin(azim) * w))


# =====================================================================
# leaf / petal templates  (local: x along the blade, y across, z = normal)
# =====================================================================
def blade_template(width_fn, n=9, fold=0.25, curl=0.15, tip_pinch=True):
    t = np.linspace(0, 1, n)
    w = np.array([width_fn(v) for v in t])
    if tip_pinch:
        w[-1] = 0.0
    x = t
    zc = -curl * x ** 2
    mid = np.stack([x, np.zeros(n), zc + fold * w], -1)
    left = np.stack([x, w, zc], -1)
    right = np.stack([x, -w, zc], -1)
    V = np.concatenate([mid, left, right])
    F = []
    for i in range(n - 1):
        F.append([i, n + i, n + i + 1, i + 1])
        F.append([i, i + 1, 2 * n + i + 1, 2 * n + i])
    shade = np.concatenate([np.full(n, 1.12), np.full(n, 0.92), np.full(n, 0.92)])
    along = np.concatenate([t, t, t])
    return V, np.array(F), shade, along


LEAF_SHAPES = {
    "oval": lambda t: 0.32 * math.sin(math.pi * t ** 0.85) ** 0.8,
    "oak": lambda t: 0.3 * math.sin(math.pi * t ** 0.9) ** 0.7 * (0.72 + 0.28 * math.cos(2 * math.pi * t * 4.5)),
    "birch": lambda t: 0.36 * math.sin(math.pi * t ** 0.55) ** 1.1 * (1 - 0.1 * (math.sin(t * 60) > 0)),
    "lance": lambda t: 0.09 * math.sin(math.pi * t) ** 0.7,
    "needle": lambda t: 0.035,
    "needlebunch": lambda t: 0.11 * (1 - t ** 2) ** 0.7,
    "round": lambda t: 0.45 * math.sin(math.pi * t ** 0.8) ** 0.6,
    "ray": lambda t: 0.12 * math.sin(math.pi * t ** 0.6) ** 0.5,
    "petal_round": lambda t: 0.5 * math.sin(math.pi * t ** 0.75) ** 0.55,
    "petal_notch": lambda t: 0.42 * math.sin(math.pi * t ** 0.7) ** 0.6 * (1 - 0.35 * max(0, (t - 0.85) / 0.15)),
    "pinna": lambda t: 0.17 * math.sin(math.pi * t ** 0.8) ** 0.7,
    "grassleaf": lambda t: 0.05 * (1 - t ** 1.5),
}
_TPL = {}


def leaf_tpl(kind, n=None, fold=0.25, curl=0.15):
    key = (kind, n, fold, curl)
    if key not in _TPL:
        n = n or (15 if kind in ("oak", "birch") else 7)
        _TPL[key] = blade_template(LEAF_SHAPES[kind], n, fold, curl, tip_pinch=kind != "needle")
    return _TPL[key]


def place_leaves(mb, tpl, base, xdir, normal, size, colors, mi, width_scale=1.0, tip_col=None, tip_pow=2.0):
    """Instantiate template tpl at many frames (vectorised)."""
    V0, F0, shade, along = tpl
    k = len(base)
    if k == 0:
        return
    X = _n(xdir)
    Zn = _n(normal - X * np.sum(normal * X, -1, keepdims=True))
    Y = np.cross(Zn, X)
    s = np.asarray(size, float).reshape(-1, 1, 1)
    loc = V0[None] * s
    loc[..., 1] *= width_scale
    V = base[:, None] + loc[..., 0:1] * X[:, None] + loc[..., 1:2] * Y[:, None] + loc[..., 2:3] * Zn[:, None]
    C = np.asarray(colors, float)[:, None, :3] * shade[None, :, None]
    if tip_col is not None:
        tc = np.asarray(tip_col, float)
        tc = tc[:, None, :3] if tc.ndim == 2 else tc[None, None, :3]
        g = along[None, :, None] ** tip_pow
        C = C * (1 - g) + tc * g
    F = (F0[None] + (np.arange(k) * len(V0))[:, None, None]).reshape(-1, F0.shape[1])
    mb.add(V.reshape(-1, 3), F, C=np.clip(C.reshape(-1, 3), 0, 1), mi=mi)


def palette(rng, n, cols, weights=None, jitter=0.08):
    cols = np.asarray(cols, float)
    idx = rng.choice(len(cols), n, p=weights)
    a = cols[idx]
    b = cols[rng.integers(0, len(cols), n)]
    t = rng.random((n, 1)) * 0.5
    c = a * (1 - t) + b * t
    return np.clip(c * (1 + rng.normal(0, jitter, (n, 1))), 0, 1)


# =====================================================================
# trees
# =====================================================================
SPECIES = {
    # level 1 lengths are fractions of the tree height H, deeper levels of the parent length
    "oak": dict(
        H=13.0, trunk=0.3, R0=0.5, lean=0.08, wobble=0.5, flare=0.9, tsegs=14, tsides=22, bark="oak",
        levels=[
            dict(n=6, t0=0.55, t1=1.0, ang=(30, 68), len=0.58, prof="flat", rr=0.62, segs=12, sides=12, tort=0.3, grav=0.04, up=0.1),
            dict(n=11, t0=0.15, t1=1.0, ang=(35, 75), len=0.45, prof="taper", rr=0.5, segs=7, sides=7, tort=0.32, grav=0.12, up=0.06),
            dict(n=9, t0=0.15, t1=1.0, ang=(40, 80), len=0.4, prof="taper", rr=0.5, segs=4, sides=4, tort=0.4, grav=0.05, up=0.15),
        ],
        leaf=dict(kind="oak", per=22, size=0.17, t0=0.0, spread=(40, 85), on_parent=12, cluster=3, cr=0.14, tpl_n=7,
                  cols=[(0.05, 0.14, 0.025), (0.09, 0.2, 0.035), (0.14, 0.25, 0.045), (0.24, 0.31, 0.06)],
                  tip=(0.28, 0.34, 0.07)),
    ),
    "birch": dict(
        H=16.0, trunk=0.97, R0=0.24, lean=0.05, wobble=0.25, flare=0.5, tsegs=18, tsides=14, bark="birch",
        levels=[
            dict(n=30, t0=0.3, t1=0.98, ang=(38, 58), len=0.3, prof="cone", rr=0.45, segs=8, sides=6, tort=0.2, grav=0.2, up=0.08),
            dict(n=12, t0=0.2, t1=1.0, ang=(30, 60), len=0.55, prof="taper", rr=0.5, segs=6, sides=3, tort=0.2, grav=0.6, up=0.0),
        ],
        leaf=dict(kind="birch", per=30, size=0.13, t0=0.05, spread=(45, 85), cluster=3, cr=0.1, tpl_n=7,
                  cols=[(0.2, 0.3, 0.05), (0.33, 0.4, 0.07), (0.48, 0.5, 0.1), (0.6, 0.53, 0.12)],
                  tip=(0.58, 0.56, 0.14)),
    ),
    "willow": dict(
        H=13.0, trunk=0.34, R0=0.6, lean=0.12, wobble=0.6, flare=0.8, tsegs=12, tsides=18, bark="willow",
        levels=[
            dict(n=8, t0=0.65, t1=1.0, ang=(40, 72), len=0.55, prof="flat", rr=0.6, segs=12, sides=10, tort=0.2, grav=0.3, up=0.1),
            dict(n=34, t0=0.15, t1=1.0, ang=(50, 95), len=1.15, prof="flat", rr=0.3, segs=16, sides=3, tort=0.04, grav=2.2, up=0.0),
        ],
        leaf=dict(kind="lance", per=80, size=0.17, t0=0.02, spread=(15, 45), cluster=2, cr=0.05, tpl_n=5,
                  cols=[(0.18, 0.28, 0.1), (0.28, 0.38, 0.15), (0.4, 0.48, 0.2)], tip=(0.5, 0.55, 0.25)),
    ),
    "spruce": dict(
        H=22.0, trunk=1.0, R0=0.42, lean=0.02, wobble=0.12, flare=0.6, tsegs=18, tsides=12, bark="pine",
        levels=[
            dict(n=80, t0=0.06, t1=0.99, ang=(80, 102), len=0.26, prof="cone", rr=0.3, segs=6, sides=5, tort=0.1, grav=0.3, up=0.0, tipup=0.5),
            dict(n=16, t0=0.05, t1=1.0, ang=(45, 65), len=0.34, prof="taper", rr=0.5, segs=3, sides=3, tort=0.15, grav=0.25, up=0.0, planar=True),
        ],
        leaf=dict(kind="needlebunch", per=40, size=0.22, t0=0.0, spread=(35, 75), on_parent=30, cluster=2, cr=0.05, tpl_n=4,
                  cols=[(0.02, 0.065, 0.04), (0.03, 0.09, 0.055), (0.05, 0.12, 0.07)], tip=(0.12, 0.22, 0.1)),
    ),
    "sacred": dict(
        H=30.0, trunk=0.36, R0=1.6, lean=0.03, wobble=0.8, flare=1.6, tsegs=24, tsides=40, bark="sacred", flutes=7,
        levels=[
            dict(n=9, t0=0.72, t1=1.0, ang=(52, 80), len=0.58, prof="flat", rr=0.52, segs=14, sides=16, tort=0.22, grav=0.06, up=0.02),
            dict(n=10, t0=0.18, t1=1.0, ang=(30, 70), len=0.42, prof="taper", rr=0.5, segs=8, sides=8, tort=0.28, grav=0.1, up=0.06),
            dict(n=8, t0=0.2, t1=1.0, ang=(35, 75), len=0.4, prof="taper", rr=0.5, segs=4, sides=4, tort=0.35, grav=0.08, up=0.1),
        ],
        leaf=dict(kind="blossom", per=5, size=0.2, t0=0.2, spread=(40, 80),
                  cols=[(1.0, 0.78, 0.86), (0.98, 0.66, 0.8), (1.0, 0.9, 0.93), (0.92, 0.6, 0.78)], tip=None),
    ),
    "shrub": dict(
        H=2.4, trunk=0.0, R0=0.05, lean=0.0, wobble=0.2, flare=0.0, tsegs=2, tsides=4, bark="oak", stems=9,
        levels=[
            dict(n=7, t0=0.3, t1=1.0, ang=(30, 60), len=0.45, prof="taper", rr=0.6, segs=4, sides=4, tort=0.3, grav=0.1, up=0.1),
        ],
        leaf=dict(kind="round", per=16, size=0.12, t0=0.1, spread=(45, 80), cluster=3, cr=0.08, tpl_n=6,
                  cols=[(0.07, 0.18, 0.04), (0.14, 0.26, 0.05), (0.22, 0.32, 0.07)], tip=(0.3, 0.38, 0.1)),
    ),
}


class _Tree:
    def __init__(self, sp, rng):
        self.sp, self.rng = sp, rng
        self.branches = []  # (pts, radii, level)

    def grow_branch(self, start, d, length, r0, level, lp):
        rng = self.rng
        n = lp["segs"] if level > 0 else self.sp["tsegs"]
        pts = [np.array(start, float)]
        d = _n(np.asarray(d, float))
        step = length / n
        for i in range(n):
            f = (i + 1) / n
            tort = lp.get("tort", 0.2)
            grav = lp.get("grav", 0.0) * (0.4 + f)
            up = lp.get("up", 0.0)
            tipup = lp.get("tipup", 0.0) * f ** 3
            d = _n(d + rng.normal(0, tort * 0.35, 3) + (up + tipup - grav) * Z)
            pts.append(pts[-1] + d * step)
        pts = np.array(pts)
        t = np.linspace(0, 1, n + 1)
        taper = 0.75 if level < len(self.sp["levels"]) else 0.6
        radii = np.maximum(r0 * (1 - taper * t), 0.004)
        return pts, radii

    def build(self):
        sp, rng = self.sp, self.rng
        H = sp["H"]
        stems = []
        if sp.get("stems"):
            for k in range(sp["stems"]):
                d = rot_dir(Z, math.radians(rng.uniform(8, 38)), rng.uniform(0, 2 * np.pi))
                L = H * rng.uniform(0.55, 0.9)
                pts, radii = self.grow_branch(rng.normal(0, 0.08, 3) * [1, 1, 0], d, L, sp["R0"] * rng.uniform(0.7, 1.2), 0,
                                              dict(segs=6, tort=0.2, up=0.05))
                stems.append((pts, radii))
        else:
            n = sp["tsegs"]
            L = H * sp["trunk"]
            t = np.linspace(0, 1, n + 1)
            lean = rng.normal(0, sp["lean"], 2)
            wob = sp["wobble"]
            ph = rng.uniform(0, 10, 2)
            x = lean[0] * t * L + wob * 0.3 * np.sin(t * 5 + ph[0]) * t
            y = lean[1] * t * L + wob * 0.3 * np.sin(t * 4 + ph[1]) * t
            pts = np.stack([x, y, t * L], -1)
            radii = sp["R0"] * (1 - 0.55 * t) * (1 + sp["flare"] * np.exp(-t * L / 0.9))
            stems.append((pts, radii))
        self.trunks = stems
        for pts, radii in stems:
            self.branches.append((pts, radii, 0))
            self.spawn(pts, radii, 1)
        # buttress roots
        if not sp.get("stems") and sp["flare"] > 0.4:
            for k in range(rng.integers(5, 9)):
                a = 2 * np.pi * k / 7 + rng.uniform(-0.3, 0.3)
                d = np.array([math.cos(a), math.sin(a), 0.0])
                L = sp["R0"] * rng.uniform(3, 6)
                tt = np.linspace(0, 1, 6)[:, None]
                p = d * L * tt + np.array([0, 0, 1.0]) * (sp["R0"] * 1.2 * (1 - tt) ** 2 - 0.25 * tt)
                self.branches.append((p, np.linspace(sp["R0"] * 0.55, sp["R0"] * 0.08, 6), -1))
        return self

    def spawn(self, ppts, prad, level):
        sp, rng = self.sp, self.rng
        if level > len(sp["levels"]):
            return
        lp = sp["levels"][level - 1]
        seglen = np.linalg.norm(np.diff(ppts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seglen)])
        plen = cum[-1]
        count = max(1, int(round(lp["n"] * rng.uniform(0.8, 1.2))))
        az0 = rng.uniform(0, 2 * np.pi)
        for k in range(count):
            t = lp["t0"] + (lp["t1"] - lp["t0"]) * (k + rng.uniform(0.2, 0.8)) / count
            s = t * plen
            i = min(int(np.searchsorted(cum, s)) - 1, len(ppts) - 2)
            i = max(i, 0)
            f = (s - cum[i]) / max(seglen[i], 1e-6)
            p = ppts[i] * (1 - f) + ppts[i + 1] * f
            r_at = prad[i] * (1 - f) + prad[i + 1] * f
            tdir = _n(ppts[i + 1] - ppts[i])
            if lp.get("planar"):
                az = (np.pi / 2 if k % 2 == 0 else -np.pi / 2) + rng.normal(0, 0.25)
                u, w = perp_frame(tdir)
                # keep twigs roughly horizontal-planar
                side = _n(np.cross(tdir, Z)) if abs(tdir[2]) < 0.95 else u
                d = _n(math.cos(math.radians(rng.uniform(*lp["ang"]))) * tdir
                       + math.sin(math.radians(rng.uniform(*lp["ang"]))) * side * (1 if k % 2 == 0 else -1))
            else:
                az = az0 + k * math.radians(137.5) + rng.normal(0, 0.3)
                d = rot_dir(tdir, math.radians(rng.uniform(*lp["ang"])), az)
            prof = {"flat": 1.0, "taper": 1.0 - 0.55 * t, "cone": (1.0 - t) ** 0.9 * 1.1 + 0.08}[lp["prof"]]
            ref = self.sp["H"] if level == 1 and not self.sp.get("stems") else plen
            L = ref * lp["len"] * prof * rng.uniform(0.8, 1.2)
            r0 = max(r_at * lp["rr"], 0.005)
            pts, radii = self.grow_branch(p, d, L, r0, level, lp)
            self.branches.append((pts, radii, level))
            self.spawn(pts, radii, level + 1)

    def mesh(self, leaf_mi=1, blossom_mi=2):
        sp, rng = self.sp, self.rng
        mb = MB()
        nlev = len(sp["levels"])
        bark_col = np.ones(3)
        for pts, radii, level in self.branches:
            if level < 0:
                sides = 10
            elif level == 0:
                sides = sp["tsides"]
            else:
                sides = sp["levels"][level - 1]["sides"]
            rmod = None
            if level == 0 and sp.get("flutes"):
                k_ = sp["flutes"]
                rmod = lambda t, a, k_=k_: 1 + 0.16 * np.cos(k_ * a + 2.2 * t) * (1 - 0.5 * t) + 0.05 * np.cos(3 * k_ * a - 4 * t)
            V, F, UV, C = tube(pts, radii, sides, np.repeat(bark_col[None], len(pts), 0), rmod=rmod)
            mb.add(V, F, UV, C, mi=0)
        # leaves
        lf = sp["leaf"]
        twigs = [(p, r) for p, r, lv in self.branches if lv == nlev]
        parents = [(p, r) for p, r, lv in self.branches if lv == nlev - 1] if lf.get("on_parent") else []
        base, xd, nm, sz = [], [], [], []
        for group, per in ((twigs, lf["per"]), (parents, lf.get("on_parent", 0))):
            for pts, radii in group:
                seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                cum = np.concatenate([[0], np.cumsum(seg)])
                L = cum[-1]
                m = max(1, int(per * rng.uniform(0.7, 1.2) * (min(L, 1.5) if lf["kind"].startswith("needle") else 1.0)))
                ts = np.sort(rng.uniform(lf["t0"], 1.0, m))
                s = ts * L
                idx = np.clip(np.searchsorted(cum, s) - 1, 0, len(pts) - 2)
                f = ((s - cum[idx]) / np.maximum(seg[idx], 1e-6))[:, None]
                p = pts[idx] * (1 - f) + pts[idx + 1] * f
                tdir = _n(pts[idx + 1] - pts[idx])
                for j in range(m):
                    ang = math.radians(rng.uniform(*lf["spread"]))
                    az = j * 2.4 + rng.normal(0, 0.5)
                    d = rot_dir(tdir[j], ang, az)
                    if not lf["kind"].startswith("needle") and lf["kind"] != "lance":
                        d = _n(d + 0.35 * Z)
                    if lf["kind"] == "lance":
                        d = _n(d - 0.6 * Z)
                    n_ = _n(Z + rng.normal(0, 0.35, 3) + 0.4 * _n(np.array([p[j][0], p[j][1], 0]) + 1e-6))
                    base.append(p[j])
                    xd.append(d)
                    nm.append(n_)
                    sz.append(lf["size"] * rng.uniform(0.75, 1.25))
        base, xd, nm, sz = map(np.array, (base, xd, nm, sz))
        k = lf.get("cluster", 1)
        if k > 1 and len(base):
            m = len(base)
            base = np.repeat(base, k, 0) + rng.normal(0, lf.get("cr", 0.1), (m * k, 3))
            xd = _n(np.repeat(xd, k, 0) + rng.normal(0, 0.45, (m * k, 3)))
            nm = _n(np.repeat(nm, k, 0) + rng.normal(0, 0.3, (m * k, 3)))
            sz = np.repeat(sz, k) * rng.uniform(0.7, 1.1, m * k)
        if lf["kind"] == "blossom":
            self._blossoms(mb, base, xd, sz, blossom_mi, leaf_mi)
        else:
            cols = palette(rng, len(base), lf["cols"])
            tip = None if lf.get("tip") is None else np.repeat(np.array(lf["tip"])[None], len(base), 0) * (0.85 + 0.3 * rng.random((len(base), 1)))
            needle = lf["kind"].startswith("needle")
            fold = 0.08 if needle else 0.28
            curl = 0.05 if needle else 0.18
            place_leaves(mb, leaf_tpl(lf["kind"], n=lf.get("tpl_n"), fold=fold, curl=curl), base, xd, nm, sz, cols,
                         leaf_mi, tip_col=tip)
        return mb

    def _twisted_trunk(self, mb):
        sp, rng = self.sp, self.rng
        H, R0 = sp["H"], sp["R0"]
        L = H * sp["trunk"]
        for k in range(4):
            t = np.linspace(0, 1, 30)
            a = 2 * np.pi * k / 4 + t * 2.6
            rr = R0 * 1.6 * (1 - t) ** 1.5 + R0 * 0.35
            pts = np.stack([rr * np.cos(a), rr * np.sin(a), t * L * 1.02], -1)
            radii = R0 * 0.62 * (1 - 0.35 * t) * (1 + 0.8 * np.exp(-t * L / 2.0))
            V, F, UV, C = tube(pts, radii, 18, np.ones((30, 3)))
            mb.add(V, F, UV, C, mi=0)

    def _blossoms(self, mb, base, xd, sz, mi_blossom, mi_leaf):
        rng = self.rng
        # clusters of 5-petal flowers around every twig anchor
        tpl = leaf_tpl("petal_notch", n=6, fold=0.05, curl=-0.35)
        cols_all = self.sp["leaf"]["cols"]
        k = len(base)
        per = 7
        cb = np.repeat(base, per, 0) + rng.normal(0, 0.25, (k * per, 3))
        fdir = _n(rng.normal(0, 1, (k * per, 3)) + np.repeat(xd, per, 0) * 0.8 - Z * 0.3)
        fsz = np.repeat(sz, per) * rng.uniform(0.8, 1.2, k * per)
        fc = palette(rng, k * per, cols_all, jitter=0.05)
        B, X, N, S_, C = [], [], [], [], []
        for p in range(5):
            a = 2 * np.pi * p / 5
            u = _n(np.cross(fdir, Z) + 1e-6)
            w = np.cross(fdir, u)
            pd = _n(math.cos(a) * u + math.sin(a) * w + fdir * 0.35)
            B.append(cb)
            X.append(pd)
            N.append(fdir)
            S_.append(fsz)
            C.append(fc)
        place_leaves(mb, tpl, np.concatenate(B), np.concatenate(X), np.concatenate(N), np.concatenate(S_),
                     np.concatenate(C), mi_blossom, width_scale=1.0,
                     tip_col=None)
        # hanging wisteria-like racemes from the lower twigs
        sel = rng.random(k) < 0.22
        rb = base[sel]
        RB, RX, RN, RS, RC = [], [], [], [], []
        for j, p0 in enumerate(rb):
            L = rng.uniform(0.8, 2.6)
            m = int(L * 22)
            tt = np.linspace(0, 1, m)
            sway = rng.normal(0, 0.08, 2)
            pts = p0 + np.stack([sway[0] * tt, sway[1] * tt, -L * tt], -1)
            for q in range(3):
                a = 2 * np.pi * q / 3 + tt * 7
                d = _n(np.stack([np.cos(a), np.sin(a), -0.5 * np.ones(m)], -1))
                RB.append(pts)
                RX.append(d)
                RN.append(np.repeat(Z[None], m, 0))
                RS.append(0.07 * (1 - 0.6 * tt) * rng.uniform(0.8, 1.2))
                c = np.array([0.78, 0.62, 0.98]) * (1 - tt[:, None]) + np.array([1.0, 0.72, 0.88]) * tt[:, None]
                RC.append(c)
        if RB:
            place_leaves(mb, leaf_tpl("petal_round", n=5, fold=0.2, curl=-0.4), np.concatenate(RB), np.concatenate(RX),
                         np.concatenate(RN), np.concatenate(RS), np.concatenate(RC), mi_blossom)
        # a sprinkle of fresh leaves
        sel = rng.random(k) < 0.5
        lc = palette(rng, sel.sum(), [(0.2, 0.35, 0.08), (0.3, 0.45, 0.1)])
        place_leaves(mb, leaf_tpl("oval"), base[sel], xd[sel], np.repeat(Z[None], sel.sum(), 0), sz[sel] * 1.6, lc, mi_leaf)


def make_tree(species, seed):
    """Returns MB with material slots: 0 bark, 1 leaves, 2 blossoms."""
    rng = np.random.default_rng(seed)
    t = _Tree(SPECIES[species], rng).build()
    return t.mesh()


# =====================================================================
# grass
# =====================================================================
GRASS_STYLES = {
    "meadow": dict(n=46, h=(0.35, 0.85), w=0.006, lean=(4, 32), dry=0.12, seed_frac=0.1,
                   base=(0.03, 0.08, 0.015), mid=(0.12, 0.26, 0.04), tip=(0.36, 0.45, 0.1)),
    "lush": dict(n=40, h=(0.3, 0.7), w=0.007, lean=(4, 28), dry=0.03, seed_frac=0.03,
                 base=(0.02, 0.07, 0.012), mid=(0.08, 0.22, 0.03), tip=(0.22, 0.38, 0.06)),
    "dry": dict(n=34, h=(0.5, 1.05), w=0.005, lean=(5, 30), dry=0.55, seed_frac=0.35,
                base=(0.05, 0.1, 0.02), mid=(0.2, 0.3, 0.07), tip=(0.55, 0.5, 0.2)),
    "short": dict(n=60, h=(0.08, 0.24), w=0.004, lean=(5, 45), dry=0.08, seed_frac=0.0,
                  base=(0.03, 0.08, 0.015), mid=(0.1, 0.24, 0.04), tip=(0.28, 0.4, 0.08)),
}


def make_grass(style, seed):
    """Material slots: 0 grass (blades + seed heads)."""
    st = GRASS_STYLES[style]
    rng = np.random.default_rng(seed)
    mb = MB()
    nseg = 7
    for b in range(st["n"]):
        a = rng.uniform(0, 2 * np.pi)
        outd = np.array([math.cos(a), math.sin(a), 0.0])
        r0 = rng.uniform(0, 0.09) ** 0.8
        base = outd * r0
        h = rng.uniform(*st["h"])
        lean = math.radians(rng.uniform(*st["lean"])) * (0.6 + r0 * 6)
        droop = rng.uniform(0.0, 0.35) * h
        t = np.linspace(0, 1, nseg + 1)
        centre = (base[None] + outd[None] * (math.sin(lean) * h * t ** 1.6)[:, None]
                  + Z[None] * (math.cos(lean) * h * t - droop * t ** 3)[:, None])
        tw = rng.uniform(-1.2, 1.2) * t
        side0 = np.array([-math.sin(a), math.cos(a), 0.0])
        side = np.cos(tw)[:, None] * side0[None] + np.sin(tw)[:, None] * outd[None]
        w = st["w"] * rng.uniform(0.7, 1.4) * (1 - t ** 1.8) + 0.0004
        tang = _n(np.gradient(centre, axis=0))
        nrm = _n(np.cross(side, tang))
        V = np.concatenate([centre + nrm * (w * 0.35)[:, None], centre + side * w[:, None], centre - side * w[:, None]])
        n1 = nseg + 1
        F = []
        for i in range(nseg):
            F.append([i, n1 + i, n1 + i + 1, i + 1])
            F.append([i, i + 1, 2 * n1 + i + 1, 2 * n1 + i])
        dry = rng.random() < st["dry"]
        if dry:
            cb, cm, ct = np.array([0.1, 0.12, 0.04]), np.array([0.42, 0.38, 0.18]), np.array([0.7, 0.62, 0.36])
        else:
            j = rng.normal(1, 0.1)
            cb, cm, ct = np.array(st["base"]) * j, np.array(st["mid"]) * j, np.array(st["tip"]) * rng.normal(1, 0.12)
        tt = np.concatenate([t, t, t])[:, None]
        C = np.where(tt < 0.5, cb * (1 - tt * 2) + cm * tt * 2, cm * (2 - tt * 2) + ct * (tt * 2 - 1))
        C[: n1] *= 1.1
        mb.add(V, F, C=np.clip(C, 0, 1), mi=0)
        if rng.random() < st["seed_frac"]:
            _seed_head(mb, rng, base, h * rng.uniform(1.05, 1.3), outd)
    return mb


def _seed_head(mb, rng, base, h, outd):
    t = np.linspace(0, 1, 6)
    bend = rng.uniform(0.05, 0.2)
    pts = base[None] + outd[None] * (bend * h * t ** 2)[:, None] + Z[None] * (h * t)[:, None]
    V, F, UV, C = tube(pts, np.linspace(0.0014, 0.0008, 6), 3, np.repeat([[0.4, 0.42, 0.2]], 6, 0))
    mb.add(V, F, UV, C, mi=0)
    top = pts[-1]
    k = rng.integers(10, 18)
    tt = rng.uniform(0, 1, k)
    anchors = pts[-3][None] * (1 - tt[:, None]) + top[None] * tt[:, None]
    d = _n(rng.normal(0, 1, (k, 3)) * [1, 1, 0.3] - Z * rng.uniform(0.4, 1.2, (k, 1)) + outd * 0.5)
    cols = palette(rng, k, [(0.78, 0.66, 0.4), (0.7, 0.6, 0.35), (0.85, 0.75, 0.5)], jitter=0.05)
    place_leaves(mb, leaf_tpl("oval", n=5, fold=0.4, curl=0.05), anchors, d, _n(rng.normal(0, 1, (k, 3))),
                 rng.uniform(0.012, 0.02, k), cols, 0, width_scale=0.7)


# =====================================================================
# wildflowers (material slots: 0 petals, 1 green parts, 2 centre)
# =====================================================================
def _stem(mb, rng, h, curve=0.06, r=0.0025, col=(0.12, 0.26, 0.05)):
    t = np.linspace(0, 1, 7)
    ph = rng.uniform(0, 6)
    pts = np.stack([curve * h * np.sin(t * 2 + ph) * t, curve * h * np.cos(t * 1.7 + ph) * t * 0.6, h * t], -1)
    V, F, UV, C = tube(pts, np.linspace(r, r * 0.7, 7), 5, np.repeat([col], 7, 0))
    mb.add(V, F, UV, C, mi=1)
    top_dir = _n(pts[-1] - pts[-2])
    return pts[-1], top_dir, pts


def _stem_leaves(mb, rng, pts, n, size, kind="lance", col=(0.1, 0.24, 0.04)):
    if n <= 0:
        return
    idx = rng.integers(1, len(pts) - 2, n)
    base = pts[idx]
    d = _n(rng.normal(0, 1, (n, 3)) * [1, 1, 0] + Z * 0.8)
    place_leaves(mb, leaf_tpl(kind, fold=0.3, curl=0.3), base, d, np.repeat(Z[None], n, 0),
                 rng.uniform(0.7, 1.2, n) * size, palette(rng, n, [col]), 1)


def _radial_petals(mb, rng, centre, axis, count, size, kind, colors, cup_deg, mi=0, tip_col=None, jitter_deg=6,
                   fold=0.15, curl=0.1, width_scale=1.0, tip_pow=2.0):
    u, w = perp_frame(axis)
    az = np.arange(count) * 2 * np.pi / count + rng.normal(0, 0.05, count)
    cup = np.radians(cup_deg + rng.normal(0, jitter_deg, count))
    rad = np.cos(az)[:, None] * u + np.sin(az)[:, None] * w
    d = _n(rad * np.cos(cup)[:, None] + axis[None] * np.sin(cup)[:, None])
    nrm = _n(axis[None] - d * np.sum(axis[None] * d, -1, keepdims=True))
    base = centre[None] + rad * size * 0.08
    cols = np.repeat(np.array(colors, float)[None], count, 0) if np.ndim(colors) == 1 else colors
    tc = None if tip_col is None else np.repeat(np.array(tip_col)[None], count, 0)
    place_leaves(mb, leaf_tpl(kind, fold=fold, curl=curl), base, d, nrm,
                 np.full(count, size) * rng.uniform(0.9, 1.1, count), cols, mi, width_scale=width_scale, tip_col=tc,
                 tip_pow=tip_pow)


def make_flower(kind, seed):
    rng = np.random.default_rng(seed)
    mb = MB()
    if kind == "poppy":
        h = rng.uniform(0.45, 0.7)
        top, ax, pts = _stem(mb, rng, h, r=0.0022)
        ax = _n(ax + rng.normal(0, 0.2, 3) + Z * 0.5)
        base_col = np.array([0.82, 0.05, 0.02]) * rng.uniform(0.9, 1.1)
        # dark basal blotch -> base of petal colour, bright to the rim
        _radial_petals(mb, rng, top, ax, 4, 0.05, "petal_round", np.array([0.08, 0.01, 0.01]), 38, tip_col=base_col,
                       fold=0.05, curl=-0.5, tip_pow=0.3)
        V0, F0 = ico(1)
        mb.add(top + V0 * [0.007, 0.007, 0.006] + ax * 0.006, F0, C=(0.35, 0.4, 0.3), mi=2)
        ring = rng.normal(0, 1, (18, 3))
        ring = top + _n(ring * [1, 1, 0.2]) * 0.009 + ax * 0.004
        for p in ring:
            mb.add(p + V0 * 0.0016, F0, C=(0.02, 0.02, 0.03), mi=2)
        _stem_leaves(mb, rng, pts, 2, 0.08, "oak")
        if rng.random() < 0.5:  # nodding bud
            bt = pts[3] + rng.normal(0, 0.02, 3)
            V0, F0 = ico(2)
            mb.add(bt + V0 * [0.008, 0.008, 0.013], F0, C=(0.18, 0.28, 0.08), mi=1)
    elif kind == "daisy":
        h = rng.uniform(0.35, 0.6)
        top, ax, pts = _stem(mb, rng, h, r=0.0018)
        ax = _n(ax + Z * 0.6)
        _radial_petals(mb, rng, top, ax, 24, 0.032, "ray", (0.95, 0.95, 0.92), -6, fold=0.2, curl=0.25)
        V0, F0 = ico(2)
        mb.add(top + V0 * [0.011, 0.011, 0.005] + ax * 0.003, F0, C=(1.0, 0.72, 0.05), mi=2)
        _stem_leaves(mb, rng, pts, 3, 0.05, "lance")
    elif kind == "cornflower":
        h = rng.uniform(0.4, 0.65)
        top, ax, pts = _stem(mb, rng, h, r=0.0016, col=(0.25, 0.35, 0.22))
        ax = _n(ax + Z * 0.4)
        _radial_petals(mb, rng, top, ax, 11, 0.022, "petal_notch", (0.18, 0.3, 0.95), 25, fold=0.35, curl=0.0,
                       width_scale=0.6, tip_col=(0.3, 0.45, 1.0))
        V0, F0 = ico(1)
        mb.add(top + V0 * 0.006, F0, C=(0.25, 0.1, 0.45), mi=2)
        _stem_leaves(mb, rng, pts, 3, 0.07, "lance", col=(0.2, 0.3, 0.15))
    elif kind == "buttercup":
        for _ in range(rng.integers(2, 4)):
            h = rng.uniform(0.3, 0.5)
            top, ax, pts = _stem(mb, rng, h, curve=0.12, r=0.0014)
            ax = _n(ax + Z * 0.8)
            _radial_petals(mb, rng, top, ax, 5, 0.014, "petal_round", (1.0, 0.78, 0.02), 45, fold=0.05, curl=-0.6)
            V0, F0 = ico(1)
            mb.add(top + V0 * 0.004 + ax * 0.002, F0, C=(0.9, 0.6, 0.0), mi=2)
        _stem_leaves(mb, rng, pts, 2, 0.05, "oak")
    elif kind == "lupine":
        h = rng.uniform(0.55, 0.85)
        top, ax, pts = _stem(mb, rng, h, curve=0.03, r=0.003)
        k = 34
        tt = np.linspace(0, 1, k)
        spike = pts[-1] - ax * 0.18 * (1 - tt)[:, None]
        a = tt * 2 * np.pi * 7
        u, w = perp_frame(ax)
        d = _n(np.cos(a)[:, None] * u + np.sin(a)[:, None] * w + ax * 0.2)
        c = np.array([0.42, 0.24, 0.85]) * (1 - tt[:, None]) + np.array([0.85, 0.55, 0.95]) * tt[:, None]
        place_leaves(mb, leaf_tpl("petal_round", n=5, fold=0.5, curl=0.2), spike, d, np.repeat(ax[None], k, 0),
                     0.016 * (1 - 0.5 * tt), c, 0)
        _stem_leaves(mb, rng, pts, 4, 0.08, "lance")
    elif kind == "dandelion":
        h = rng.uniform(0.18, 0.32)
        top, ax, pts = _stem(mb, rng, h, curve=0.1, r=0.0022, col=(0.2, 0.3, 0.12))
        ax = _n(ax + Z * 0.6)
        _radial_petals(mb, rng, top, ax, 26, 0.02, "ray", (1.0, 0.72, 0.02), 15, fold=0.2, curl=0.1)
        _radial_petals(mb, rng, top + ax * 0.003, ax, 14, 0.013, "ray", (1.0, 0.62, 0.0), 45, fold=0.2, curl=0.1)
        _rosette(mb, rng, 6, 0.14)
    elif kind == "clock":  # dandelion seed head
        h = rng.uniform(0.25, 0.4)
        top, ax, pts = _stem(mb, rng, h, curve=0.08, r=0.0021, col=(0.3, 0.38, 0.2))
        dandelion_head(mb, rng, top, radius=0.02, seeds=70, hairs=10, mi_hair=0, mi_seed=2)
        _rosette(mb, rng, 5, 0.12)
    return mb


def _rosette(mb, rng, n, size):
    a = np.arange(n) * 2 * np.pi / n + rng.normal(0, 0.2, n)
    d = _n(np.stack([np.cos(a), np.sin(a), np.full(n, 0.35)], -1))
    place_leaves(mb, leaf_tpl("oak", fold=0.2, curl=0.4), np.zeros((n, 3)) + [0, 0, 0.01], d,
                 np.repeat(Z[None], n, 0), rng.uniform(0.8, 1.2, n) * size,
                 palette(rng, n, [(0.08, 0.2, 0.04), (0.14, 0.26, 0.05)]), 1, width_scale=0.7)


def dandelion_head(mb, rng, centre, radius=0.02, seeds=70, hairs=10, mi_hair=0, mi_seed=2, pappus_r=0.009):
    """Sphere of seeds, each a beak plus an umbrella of hair ribbons."""
    g = math.pi * (3 - math.sqrt(5))
    i = np.arange(seeds)
    z = 1 - 2 * (i + 0.5) / seeds
    keep = z > -0.7
    z = z[keep]
    r = np.sqrt(1 - z * z)
    ang = g * i[keep]
    D = _n(np.stack([r * np.cos(ang), r * np.sin(ang), z], -1) + rng.normal(0, 0.05, (len(z), 3)))
    tips = centre + D * radius
    # beaks as thin quads (cheap)
    base = centre + D * radius * 0.2
    w = 0.0003
    for dvec, b, tp in zip(D, base, tips):
        u, _ = perp_frame(dvec)
        V = [b - u * w, b + u * w, tp + u * w * 0.5, tp - u * w * 0.5]
        mb.add(V, [[0, 1, 2, 3]], C=(0.75, 0.72, 0.62), mi=mi_seed)
    # pappus hairs
    B, X, N, S_ = [], [], [], []
    for dvec, tp in zip(D, tips):
        u, w2 = perp_frame(dvec)
        a = np.arange(hairs) * 2 * np.pi / hairs + rng.uniform(0, 1)
        rad = np.cos(a)[:, None] * u + np.sin(a)[:, None] * w2
        d = _n(rad * 0.9 + dvec * 0.45)
        B.append(np.repeat(tp[None], hairs, 0))
        X.append(d)
        N.append(np.repeat(dvec[None], hairs, 0))
        S_.append(np.full(hairs, pappus_r))
    place_leaves(mb, leaf_tpl("needle", n=3, fold=0.0, curl=-0.3), np.concatenate(B), np.concatenate(X),
                 np.concatenate(N), np.concatenate(S_), np.full((len(D) * hairs, 3), 0.95), mi_hair, width_scale=0.35)


def make_player_seed(seed=3, hairs=90):
    """The hero seed (achene + beak + fine pappus). Slots: 0 hairs, 1 seed body."""
    rng = np.random.default_rng(seed)
    mb = MB()
    t = np.linspace(0, 1, 8)
    ach = np.stack([np.zeros(8), np.zeros(8), -0.004 + 0.004 * t], -1)
    V, F, UV, C = tube(ach, 0.00045 * np.sin(np.pi * (0.1 + 0.8 * t)) + 0.0001, 8, np.repeat([[0.36, 0.3, 0.16]], 8, 0))
    mb.add(V, F, UV, C, mi=1)
    beak = np.stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.009, 6)], -1)
    V, F, UV, C = tube(beak, 0.0001, 4, np.repeat([[0.85, 0.83, 0.75]], 6, 0))
    mb.add(V, F, UV, C, mi=1)
    tip = beak[-1]
    a = np.arange(hairs) * 2 * np.pi / hairs + rng.normal(0, 0.03, hairs)
    for k in range(hairs):
        rad = np.array([math.cos(a[k]), math.sin(a[k]), 0.0])
        L = rng.uniform(0.0055, 0.0068)
        el0 = math.radians(rng.uniform(8, 20))
        curl = math.radians(rng.uniform(15, 35))
        pts = [tip]
        p = tip.copy()
        for s in range(1, 6):
            e = el0 + curl * (s / 5) ** 1.6
            p = p + (math.cos(e) * rad + math.sin(e) * Z) * (L / 5)
            pts.append(p)
        pts = np.array(pts)
        side = _n(np.cross(rad, Z))
        tw = rng.uniform(0, np.pi)
        side = math.cos(tw) * side + math.sin(tw) * Z
        widths = np.linspace(0.00005, 0.00002, 6)
        V = np.concatenate([[pp - side * ww, pp + side * ww] for pp, ww in zip(pts, widths)])
        F = [[2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2] for i in range(5)]
        mb.add(V, F, C=(0.97, 0.97, 0.95), mi=0)
    return mb


# =====================================================================
# ferns, reeds, clover, fallen petals
# =====================================================================
def make_fern(seed, fronds=12, size=1.0):
    """Slots: 1 leaves (pinnae + rachis share the leaf material)."""
    rng = np.random.default_rng(seed)
    mb = MB()
    for f in range(fronds):
        a = 2 * np.pi * f / fronds + rng.normal(0, 0.2)
        L = size * rng.uniform(0.7, 1.15)
        el = math.radians(rng.uniform(35, 70))
        t = np.linspace(0, 1, 14)
        outd = np.array([math.cos(a), math.sin(a), 0.0])
        pts = outd[None] * (L * t * math.cos(el))[:, None] + Z[None] * (L * t * math.sin(el) - L * 0.55 * t ** 2)[:, None]
        V, F, UV, C = tube(pts, np.linspace(0.004, 0.001, 14) * size, 3, np.repeat([[0.12, 0.22, 0.05]], 14, 0))
        mb.add(V, F, UV, C, mi=1)
        tang = _n(np.gradient(pts, axis=0))
        side = _n(np.cross(tang, Z))
        nrm = _n(np.cross(side, tang))
        k = len(t) - 2
        idx = np.arange(1, len(t) - 1)
        prof = np.sin(np.pi * (0.15 + 0.85 * t[idx])) ** 0.8
        for sgn in (1, -1):
            d = _n(side[idx] * sgn + tang[idx] * 0.45)
            cols = palette(rng, k, [(0.1, 0.27, 0.04), (0.18, 0.36, 0.06), (0.26, 0.42, 0.08)], jitter=0.06)
            place_leaves(mb, leaf_tpl("pinna", fold=0.3, curl=0.15), pts[idx], d, nrm[idx], prof * 0.24 * L, cols, 1)
    return mb


def make_reeds(seed, blades=26):
    """Slots: 1 leaves, 2 cattail spikes."""
    rng = np.random.default_rng(seed)
    mb = MB()
    for b in range(blades):
        a = rng.uniform(0, 2 * np.pi)
        outd = np.array([math.cos(a), math.sin(a), 0.0])
        h = rng.uniform(1.0, 1.9)
        lean = math.radians(rng.uniform(2, 18))
        t = np.linspace(0, 1, 9)
        c = outd[None] * (math.sin(lean) * h * t ** 2)[:, None] + Z[None] * (math.cos(lean) * h * t)[:, None]
        side = np.array([-math.sin(a), math.cos(a), 0.0])
        w = 0.012 * (1 - t ** 2) + 0.0005
        V = np.concatenate([[p - side * ww, p + side * ww] for p, ww in zip(c, w)])
        F = [[2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2] for i in range(8)]
        tt = np.repeat(t, 2)[:, None]
        C = np.array([0.1, 0.18, 0.05]) * (1 - tt) + np.array([0.45, 0.48, 0.2]) * tt
        mb.add(V, F, C=C, mi=1)
    for k in range(rng.integers(3, 6)):
        h = rng.uniform(1.3, 1.9)
        off = rng.normal(0, 0.06, 3) * [1, 1, 0]
        t = np.linspace(0, 1, 6)
        pts = off + Z[None] * (h * t)[:, None] + np.array([0.03, 0.0, 0.0]) * t[:, None] ** 2
        V, F, UV, C = tube(pts, 0.004, 4, np.repeat([[0.2, 0.28, 0.1]], 6, 0))
        mb.add(V, F, UV, C, mi=1)
        sp = pts[-1] - Z * 0.22
        V, F, UV, C = tube(np.stack([sp, sp + Z * 0.16], 0), 0.013, 8, np.repeat([[0.26, 0.15, 0.07]], 2, 0))
        mb.add(V, F, UV, C, mi=2)
    return mb


def make_petal_bits(seed, n=1):
    rng = np.random.default_rng(seed)
    mb = MB()
    place_leaves(mb, leaf_tpl("petal_notch", n=6, fold=0.1, curl=-0.4), np.zeros((n, 3)),
                 _n(rng.normal(0, 1, (n, 3))), _n(rng.normal(0, 1, (n, 3))), np.full(n, 0.035),
                 palette(rng, n, [(1.0, 0.75, 0.85), (1.0, 0.85, 0.9)]), 0)
    return mb
