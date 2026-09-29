"""Rocks and ruins: fractured boulders, weathered standing stones, an overgrown
arch, fluted broken columns, lily pads. Material slots are documented per function."""
import math

import bmesh
import bpy
import mathutils
import numpy as np

from blx import MB, ico, noise3, perp_frame, tube
from flora import leaf_tpl, palette, place_leaves, _n

Z = np.array([0.0, 0.0, 1.0])


def _fracture(V, rng, cuts, bias_up=0.0):
    """Flatten the mesh against random planes -> crisp fractured facets."""
    for _ in range(cuts):
        n = _n(rng.normal(0, 1, 3) + Z * bias_up)
        proj = V @ n
        lim = np.quantile(proj, rng.uniform(0.72, 0.93))
        m = proj > lim
        V[m] -= (proj[m] - lim)[:, None] * n
    return V


def make_boulder(seed, size=(1.6, 1.2, 0.9), cuts=9, sub=5, sink=0.3):
    """Slots: 0 rock."""
    rng = np.random.default_rng(seed)
    V0, F = ico(sub)
    V = V0 * np.array(size) * rng.uniform(0.9, 1.1, 3)
    V = V * (1 + (noise3(V0, 1.2, 4, seed) - 0.5) * 0.5)[:, None]
    V = _fracture(V, rng, cuts)
    V = V + V0 * ((noise3(V0, 4.0, 4, seed + 9) - 0.5) * 0.08 * max(size))[:, None]
    V[:, 2] -= size[2] * sink
    mb = MB()
    mb.add(V, F, mi=0)
    return mb


def make_slab(seed, size=(2.5, 1.8, 0.5)):
    return make_boulder(seed, size, cuts=6, sub=4, sink=0.45)


def _block(center, size, rot, rng, chip=0.08, weather=0.05, sub=3):
    """Weathered stone block: bevelled cube, subdivided, noise-eroded, corner chips."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.06, segments=2, affect="EDGES", profile=0.5)
    bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=sub, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    V = np.array([v.co[:] for v in bm.verts])
    F = np.array([[v.index for v in f.verts] for f in bm.faces])
    bm.free()
    V = V * np.array(size)
    nrm = _n(V / np.array(size))
    V = V + nrm * ((noise3(V, 1.6, 4, int(rng.integers(1e6))) - 0.5) * weather * 2)[:, None]
    V = _fracture(V, rng, rng.integers(0, 3)) if chip > 0 else V
    R = np.array(mathutils.Euler(rot).to_matrix())
    return V @ R.T + center, F


def make_monolith(seed, h=6.5, rune=True):
    """Slots: 0 stone, 1 rune glow."""
    rng = np.random.default_rng(seed)
    V0, F = ico(5)
    V = V0 * [0.95, 0.6, h / 2]
    V[:, :2] *= (1 - 0.3 * np.clip((V[:, 2] + h / 2) / h, 0, 1) ** 1.5)[:, None]
    V = _fracture(V, rng, 10, bias_up=0.3)
    V = V + V0 * ((noise3(V0 * [1, 1, 3], 1.5, 5, seed) - 0.5) * 0.12)[:, None]
    V[:, 2] += h / 2 - 0.5
    mb = MB()
    mb.add(V, F, mi=0)
    if rune:
        t = np.linspace(0, 4 * np.pi, 90)
        r = 0.05 + 0.045 * t
        face_y = V[:, 1].min() - 0.005
        pts = np.stack([r * np.cos(t) * 0.8, np.full(90, face_y), h * 0.66 + r * np.sin(t)], -1)
        Vt, Ft, UV, _ = tube(pts, 0.011, 5)
        mb.add(Vt, Ft, mi=1)
    return mb


def make_arch(seed=5, span=19.0, pillar_h=13.2):
    """Slots: 0 stone, 1 rune glow, 2 ivy leaves."""
    rng = np.random.default_rng(seed)
    mb = MB()
    half = span / 2 + 1.5
    for sx in (-half, half):
        z = 0.0
        for k in range(6):
            h = rng.uniform(2.0, 2.5)
            V, F = _block([sx + rng.normal(0, 0.12), rng.normal(0, 0.12), z + h / 2], [3.2, 3.2, h],
                          (0, 0, rng.normal(0, 0.05)), rng, weather=0.07)
            mb.add(V, F, mi=0)
            z += h
        top = z
    R = half
    zc = top
    n = 15
    for k in range(n):
        if k in (10,):
            continue  # a fallen voussoir
        a0, a1 = math.pi * k / n, math.pi * (k + 1) / n
        am = (a0 + a1) / 2
        c = np.array([-R * math.cos(am), 0, zc + R * math.sin(am)])
        V, F = _block(c, [2 * R * math.sin((a1 - a0) / 2) * 0.97, 3.0, 2.4], (0, am - math.pi / 2, 0), rng, weather=0.06)
        mb.add(V, F, mi=0)
    # fallen voussoir at the foot of a pillar
    V, F = _block([half + 3.5, 2.5, 0.8], [2.4, 3.0, 2.2], (0.3, 0.2, 0.7), rng, weather=0.09)
    mb.add(V, F, mi=0)
    # keystone rune
    t = np.linspace(0, 2 * np.pi, 64)
    pts = np.stack([0.6 * np.cos(t), np.full(64, -1.56), zc + R + 0.1 + 0.6 * np.sin(t)], -1)
    V, F, UV, _ = tube(pts, 0.05, 5)
    mb.add(V, F, mi=1)
    t2 = np.linspace(0, 3 * np.pi, 60)
    r2 = 0.04 + 0.05 * t2
    pts2 = np.stack([r2 * np.cos(t2) * 0.9, np.full(60, -1.57), zc + R + 0.1 + r2 * np.sin(t2) * 0.9], -1)
    V, F, UV, _ = tube(pts2, 0.03, 4)
    mb.add(V, F, mi=1)
    # ivy curtains hanging from the arch
    for k in range(34):
        a = rng.uniform(0.12, 0.88) * math.pi
        side = rng.choice([-1.5, 1.5])
        start = np.array([-R * math.cos(a) * 1.05, side * rng.uniform(0.6, 1.0), zc + (R + 1.2) * math.sin(a)])
        L = rng.uniform(2.0, 9.0)
        m = int(L * 9)
        tt = np.linspace(0, 1, m)
        sway = rng.normal(0, 0.25, 2)
        pts = start + np.stack([sway[0] * tt ** 2, sway[1] * tt ** 2 + side * 0.1 * tt, -L * tt], -1)
        V, F, UV, _ = tube(pts, 0.012, 3)
        mb.add(V, F, C=(0.12, 0.1, 0.06), mi=2)
        cnt = m * 5
        idx = rng.integers(0, m, cnt)
        base = pts[idx]
        d = _n(rng.normal(0, 1, (cnt, 3)) - Z * 0.4)
        nrm = _n(np.array([0, side, 0.3])[None] + rng.normal(0, 0.4, (cnt, 3)))
        cols = palette(rng, cnt, [(0.05, 0.14, 0.03), (0.09, 0.2, 0.04), (0.16, 0.24, 0.05)])
        place_leaves(mb, leaf_tpl("oak", fold=0.25, curl=0.25), base, d, nrm, rng.uniform(0.12, 0.22, cnt), cols, 2,
                     width_scale=1.3)
    return mb


def make_column(seed, broken=True):
    """Slots: 0 stone."""
    rng = np.random.default_rng(seed)
    mb = MB()
    h = rng.uniform(3.0, 7.0) if broken else 8.0
    n_ring, n_seg = 40, 64
    t = np.linspace(0, 1, n_ring)
    a = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    r = 0.85 * (1 - 0.06 * t)[:, None] * (1 + 0.035 * np.cos(20 * a)[None])
    V = np.stack([r * np.cos(a)[None], r * np.sin(a)[None], np.repeat(t[:, None] * h, n_seg, 1)], -1).reshape(-1, 3)
    if broken:
        top = V[:, 2] > h * 0.8
        V[top, 2] = np.minimum(V[top, 2], h * 0.8 + (noise3(V[top] * [1, 1, 0], 1.8, 3, seed) - 0.3) * h * 0.35)
    V = V + _n(V * [1, 1, 0] + 1e-6) * ((noise3(V, 2.0, 4, seed + 1) - 0.5) * 0.06)[:, None]
    from blx import grid_faces
    F = grid_faces(n_ring, n_seg, wrap=True)
    mb.add(V, F, mi=0)
    # capital-less base plinth
    Vb, Fb = _block([0, 0, 0.3], [2.3, 2.3, 0.7], (0, 0, 0), rng, weather=0.05)
    mb.add(Vb, Fb, mi=0)
    return mb


def make_drum(seed):
    """Fallen column drum. Slots: 0 stone."""
    rng = np.random.default_rng(seed)
    mb = make_column(seed, broken=False)
    ob_mb = MB()
    for V, F in zip(mb.V[:1], mb.F[:1]):
        V = V.copy()
        V[:, 2] *= rng.uniform(0.15, 0.3)
        R = np.array(mathutils.Euler((math.pi / 2, 0, rng.uniform(0, 6))).to_matrix())
        V = V @ R.T + [0, 0, 0.75]
        ob_mb.add(V, F - F.min(), mi=0)
    return ob_mb


def make_lily(seed, lotus=False):
    """Slots: 0 pad (leaf mat), 1 lotus petals."""
    rng = np.random.default_rng(seed)
    mb = MB()
    n = 28
    ang = np.linspace(0.25, 2 * np.pi - 0.25, n)
    rim = np.stack([np.cos(ang) * 0.5, np.sin(ang) * 0.5, 0.012 + 0.01 * np.sin(ang * 5)], -1)
    V = np.vstack([[[0, 0, 0.0]], rim])
    F = [[0, k + 1, k + 2] for k in range(n - 1)]
    C = np.vstack([[[0.12, 0.26, 0.05]], np.repeat([[0.2, 0.36, 0.07]], n, 0)]) * rng.uniform(0.8, 1.2)
    C[1:, 0] += 0.08 * (rng.random() < 0.3)
    mb.add(V, F, C=C, mi=0)
    if lotus:
        for ring, (cnt, L, el) in enumerate([(8, 0.13, 45), (8, 0.11, 62), (6, 0.08, 75)]):
            a = np.arange(cnt) * 2 * np.pi / cnt + ring * 0.35
            d = _n(np.stack([np.cos(a) * math.cos(math.radians(el)), np.sin(a) * math.cos(math.radians(el)),
                             np.full(cnt, math.sin(math.radians(el)))], -1))
            nrm = _n(Z[None] - d * (d @ Z)[:, None])
            base = np.array([0, 0, 0.03]) + d * 0.02
            place_leaves(mb, leaf_tpl("petal_round", fold=0.35, curl=-0.3), base, d, nrm, np.full(cnt, L),
                         np.repeat([[1.0, 0.94, 0.95]], cnt, 0), 1, tip_col=(0.95, 0.5, 0.7), tip_pow=1.5)
    return mb
