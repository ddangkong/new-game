"""Stage 1 "The Valley of the First Wind" -- procedural Blender build.

A dandelion seed leaves its mother on a dawn hilltop and rides the wind for
~5 minutes (about 2.1 km of flight path) through six zones until it lands in
a sacred glade on a high plateau.

Outputs (next to this file):
  stage1_terrain.glb   terrain (8 m LOD, vertex colours) + all water meshes
  stage1_props.glb     asset library: one node per prop, at the origin
  stage1_layout.json   flight path (timed), zones, wind, landmarks, instances
  renders/*.png        Cycles preview shots

Usage:
  python3 build_stage1.py              full build + preview renders
  python3 build_stage1.py --fast       low-res/low-sample renders (composition check)
  python3 build_stage1.py --no-render  data export only
  python3 build_stage1.py --shots 3,4  render only some shots
"""
import json
import math
import os
import sys
import time

import bpy
import bmesh  # noqa: E402  (bmesh is only importable after bpy)
import mathutils
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DANDELION_GLB = os.path.join(HERE, "..", "dandelion", "dandelion.glb")
RENDER_DIR = os.path.join(HERE, "renders")
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
FAST = "--fast" in ARGS
NO_RENDER = "--no-render" in ARGS
ONLY_SHOTS = None
if "--shots" in ARGS:
    ONLY_SHOTS = {int(v) for v in ARGS[ARGS.index("--shots") + 1].split(",")}

RNG = np.random.default_rng(2026)
T0 = time.time()


def log(*a):
    print(f"[stage1 {time.time() - T0:6.1f}s]", *a, flush=True)


# =====================================================================
# noise
# =====================================================================
def _hash(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h / 4294967296.0


def vnoise(x, y, seed=0):
    xf, yf = np.floor(x), np.floor(y)
    fx, fy = x - xf, y - yf
    xi, yi = xf.astype(np.int64), yf.astype(np.int64)
    u, v = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash(xi, yi, seed), _hash(xi + 1, yi, seed)
    c, d = _hash(xi, yi + 1, seed), _hash(xi + 1, yi + 1, seed)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def fbm(x, y, octaves=4, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        s = s + amp * vnoise(x, y, seed + o * 17)
        norm += amp
        amp *= 0.5
        x, y = x * 2.03 + 13.7, y * 2.03 - 7.3
    return s / norm


def ridged(x, y, octaves=5, seed=0):
    return 1 - np.abs(2 * fbm(x, y, octaves, seed) - 1)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# =====================================================================
# flight path
# =====================================================================
CTRL = np.array([
    (0, 0), (90, -15), (200, 10), (320, 40), (450, 5), (580, -30), (700, -5),
    (820, 30), (930, 15), (1050, -10), (1160, 0), (1260, -25), (1360, 5),
    (1460, 20), (1560, 0), (1650, 10), (1720, 12), (1800, -10), (1900, -25),
    (1980, -15), (2045, -2)], float)


def catmull_rom(C, samples=60):
    C = np.vstack([2 * C[0] - C[1], C, 2 * C[-1] - C[-2]])
    out = []
    t = np.linspace(0, 1, samples, endpoint=False)[:, None]
    for i in range(1, len(C) - 2):
        p0, p1, p2, p3 = C[i - 1], C[i], C[i + 1], C[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(C[-2][None])
    return np.vstack(out)


_dense = catmull_rom(CTRL)
_seg = np.linalg.norm(np.diff(_dense, axis=0), axis=1)
_cum = np.concatenate([[0], np.cumsum(_seg)])
PATH_LEN = _cum[-1]
S = np.arange(0, PATH_LEN, 1.0)
P = np.stack([np.interp(S, _cum, _dense[:, 0]), np.interp(S, _cum, _dense[:, 1])], -1)
TAN = np.gradient(P, axis=0)
TAN /= np.linalg.norm(TAN, axis=1, keepdims=True)
NRM = np.stack([-TAN[:, 1], TAN[:, 0]], -1)  # left of travel direction
PQ, SQ = P[::4], S[::4]


def path_query(x, y):
    """Unsigned distance to the path centreline and arc length of the nearest point."""
    x, y = np.ravel(x), np.ravel(y)
    d = np.empty(len(x))
    s = np.empty(len(x))
    for i in range(0, len(x), 16384):
        dx = x[i:i + 16384, None] - PQ[None, :, 0]
        dy = y[i:i + 16384, None] - PQ[None, :, 1]
        d2 = dx * dx + dy * dy
        k = d2.argmin(1)
        d[i:i + 16384] = np.sqrt(d2[np.arange(len(k)), k])
        s[i:i + 16384] = SQ[k]
    return d, s


def s_at_x(xv):
    return float(S[np.argmax(P[:, 0] >= xv)])


# =====================================================================
# terrain
# =====================================================================
LAKE_C, LAKE_R = (930.0, 15.0), (240.0, 165.0)
ISLANDS = [(880, 62, 30), (995, -45, 20), (1045, 72, 15)]
PLAT_H = 88.0
GLADE_C = (2040.0, 0.0)
POND_C, POND_R = (2028.0, -22.0), 15.0
SACRED = (2060.0, 16.0)
LANDING = (2045.0, -2.0)
RING_X, ARCH_X = 470.0, 1400.0


def floor_z(x):
    return np.interp(x, [1100, 1200, 1690], [2.5, 3.0, 9.0])


def river_z(x):
    return floor_z(x) - 1.4


STREAM_Z = PLAT_H - 0.7


def lake_mask(x, y):
    r = np.sqrt(((x - LAKE_C[0]) / LAKE_R[0]) ** 2 + ((y - LAKE_C[1]) / LAKE_R[1]) ** 2)
    r = r + (fbm(x / 90, y / 90, 3, 5) - 0.5) * 0.35
    return 1 - smoothstep(0.82, 1.0, r)


def plateau_mask(x, y):
    nx = x + (fbm(y / 60, x / 60, 3, 9) - 0.5) * 40
    return smoothstep(1690, 1712, nx)


def height(x, y, d):
    roll = (fbm(x / 220, y / 220, 5, 1) - 0.5) * 32 + (fbm(x / 45, y / 45, 3, 2) - 0.5) * 5
    near = smoothstep(0, 160, d)
    mount = smoothstep(110, 480, d) * (70 + 150 * ridged(x / 380, y / 380, 5, 3))
    h = 5 + roll * (0.35 + 0.65 * near) + mount
    # origin hill
    h = h + 36 * np.exp(-(((x + 20) / 130) ** 2 + (y / 120) ** 2))
    # mist lake + islands
    lm = lake_mask(x, y)
    lake_floor = -5 + 3 * fbm(x / 40, y / 40, 2, 4)
    for ix, iy, r in ISLANDS:
        lake_floor = lake_floor + 10 * np.exp(-((x - ix) ** 2 + (y - iy) ** 2) / (r * r))
    h = h * (1 - lm) + lake_floor * lm
    shore = 1 - smoothstep(0.02, 0.4, lm)
    h = np.where(shore > 0.99, np.maximum(h, 0.8), h)
    # ancient gorge (river valley with high walls)
    g = smoothstep(1120, 1220, x) * (1 - plateau_mask(x, y))
    gz = (floor_z(x)
          + smoothstep(28, 85, d) * (50 + 35 * fbm(x / 70, y / 70, 4, 6))
          + smoothstep(85, 200, d) * 30 + mount * 0.6
          + (fbm(x / 15, y / 15, 3, 7) - 0.5) * 3 * smoothstep(20, 40, d)
          - 3 * np.exp(-(d / 7) ** 2))
    h = h * (1 - g) + gz * g
    # plateau above the waterfall cliff
    p = plateau_mask(x, y)
    pz = (PLAT_H + (fbm(x / 120, y / 120, 4, 8) - 0.5) * 12 * smoothstep(0, 60, d)
          + mount * 0.8 + smoothstep(40, 110, d) * 20)
    gd = np.hypot(x - GLADE_C[0], y - GLADE_C[1])
    gm = 1 - smoothstep(55, 90, gd)
    pz = pz * (1 - gm) + (PLAT_H + 0.5) * gm
    pd = np.hypot(x - POND_C[0], y - POND_C[1])
    pz = pz - 2.5 * (1 - smoothstep(POND_R - 3, POND_R + 3, pd))
    stream = np.exp(-(d / 5) ** 2) * smoothstep(1700, 1712, x) * (1 - smoothstep(2020, 2035, x))
    pz = pz - 1.6 * stream
    h = h * (1 - p) + pz * p
    return h


def is_wet(x, y, h, d):
    lake = (h < 0.25) & (x > 600) & (x < 1230)
    river = (x > 1150) & (x < 1705) & (d < 9)
    strm = (x > 1700) & (x < 2030) & (d < 5)
    pond = np.hypot(x - POND_C[0], y - POND_C[1]) < POND_R + 1
    return lake | river | strm | pond


# ---- heightfield grid (4 m) ------------------------------------------
GX0, GX1, GY0, GY1, STEP = -600.0, 2700.0, -1000.0, 1000.0, 4.0
GXS = np.arange(GX0, GX1 + 0.1, STEP)
GYS = np.arange(GY0, GY1 + 0.1, STEP)
log("computing heightfield", len(GXS), "x", len(GYS))
_XX, _YY = np.meshgrid(GXS, GYS)
_D, _SS = path_query(_XX, _YY)
DG = _D.reshape(_XX.shape)
SG = _SS.reshape(_XX.shape)
HG = height(_XX, _YY, DG)
_gy, _gx = np.gradient(HG, STEP)
NZG = 1 / np.sqrt(1 + _gx ** 2 + _gy ** 2)


def bil(G, x, y):
    fx = np.clip((np.asarray(x) - GX0) / STEP, 0, len(GXS) - 1.001)
    fy = np.clip((np.asarray(y) - GY0) / STEP, 0, len(GYS) - 1.001)
    ix, iy = fx.astype(int), fy.astype(int)
    tx, ty = fx - ix, fy - iy
    return (G[iy, ix] * (1 - tx) * (1 - ty) + G[iy, ix + 1] * tx * (1 - ty)
            + G[iy + 1, ix] * (1 - tx) * ty + G[iy + 1, ix + 1] * tx * ty)


def ground(x, y):
    return bil(HG, x, y)


# =====================================================================
# flight altitude, timing, zones
# =====================================================================
ZONES = [  # id, name_ko, name_en, x_end, relative cruise speed
    ("origin_hill", "새벽 언덕", "Dawn Hill", 200, 4.5),
    ("whisper_meadow", "속삭이는 초원", "Whispering Meadow", 700, 7.5),
    ("mist_lake", "안개 호수", "Mist Lake", 1160, 6.0),
    ("ancient_gorge", "고대 숲 협곡", "Ancient Gorge", 1560, 7.0),
    ("waterfall_ascent", "빛의 폭포", "Waterfall of Light", 1800, 6.5),
    ("sanctuary_glade", "성소의 숲", "Sanctuary Glade", 1e9, 5.0),
]
zone_idx = np.zeros(len(S), int)
for zi in range(len(ZONES) - 1, -1, -1):
    zone_idx[P[:, 0] < ZONES[zi][3]] = zi

gp = ground(P[:, 0], P[:, 1])
river_mask = (P[:, 0] > 1150) & (P[:, 0] < 1700)
gp = np.where(river_mask, np.maximum(gp, river_z(P[:, 0])), gp)
gp = np.maximum(gp, 0.0)
# look-ahead envelope so the seed rises *before* the cliff instead of into it
env = np.array([gp[max(0, i - 40):i + 90].max() for i in range(len(S))])
clear = np.select(
    [zone_idx == 0, zone_idx == 1, zone_idx == 2, zone_idx == 3, zone_idx == 4],
    [9.0, 7 + 4 * np.sin(S / 60), 4.5, 11.0, 18.0], 16.0)
# dip through the standing-stone ring
clear = clear - 5.0 * np.exp(-((P[:, 0] - RING_X) / 25) ** 2)
alt = env + clear
k = np.exp(-0.5 * (np.arange(-60, 61) / 20.0) ** 2)
k /= k.sum()
alt = np.convolve(np.pad(alt, 60, mode="edge"), k, mode="valid")
alt = np.maximum(alt, gp + 2.5)
# take-off from the mother dandelion and the final landing
g_end = gp[-1]
a0 = smoothstep(0, 70, S)
alt = alt * a0 + (gp[0] + 0.26) * (1 - a0)
a1 = smoothstep(PATH_LEN - 140, PATH_LEN, S)
alt = alt * (1 - a1) + (g_end + 0.3) * a1
ALT = alt

rel_v = np.array([ZONES[z][4] for z in zone_idx], float)
rel_v = np.convolve(np.pad(rel_v, 30, mode="edge"), np.ones(61) / 61, mode="valid")
# include vertical travel in the time budget
seg3 = np.sqrt(1 + np.gradient(ALT) ** 2)
t_raw = np.concatenate([[0], np.cumsum(seg3[1:] / rel_v[1:])])
SPEED_SCALE = t_raw[-1] / 300.0
TIME = t_raw / SPEED_SCALE
SPEED = rel_v * SPEED_SCALE
log(f"path {PATH_LEN:.0f} m, 300 s, speeds {SPEED.min():.1f}-{SPEED.max():.1f} m/s")


def path_point(xv, lat=0.0, up=0.0, ref="path"):
    i = int(np.argmax(P[:, 0] >= xv))
    b = P[i] + NRM[i] * lat
    z = (ALT[i] if ref == "path" else float(ground(b[0], b[1]))) + up
    return np.array([b[0], b[1], z])


# =====================================================================
# mesh helpers
# =====================================================================
bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene


class MB:
    """Mesh builder: numpy verts/faces, per-corner UVs, per-vertex colours, material slots."""

    def __init__(self):
        self.V, self.F, self.UV, self.C, self.M = [], [], [], [], []
        self.n = 0

    def add(self, V, F, UV=None, C=None, mi=0):
        V = np.asarray(V, float).reshape(-1, 3)
        F = np.asarray(F, np.int64)
        if F.ndim == 1:
            F = F[None]
        self.V.append(V)
        self.F.append(F + self.n)
        self.n += len(V)
        self.UV.append(np.zeros(F.shape + (2,)) if UV is None else np.asarray(UV, float).reshape(F.shape + (2,)))
        if C is None:
            C = np.ones((len(V), 4))
        C = np.asarray(C, float)
        if C.ndim == 1:
            C = np.broadcast_to(np.append(C[:3], 1.0) if len(C) == 3 else C, (len(V), 4))
        elif C.shape[1] == 3:
            C = np.hstack([C, np.ones((len(C), 1))])
        self.C.append(np.asarray(C))
        self.M.append(np.full(len(F), mi))

    def build(self, name, mats, smooth=True, link=True):
        V = np.vstack(self.V)
        me = bpy.data.meshes.new(name)
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        sizes = np.concatenate([np.full(len(f), f.shape[1]) for f in self.F])
        loops = np.concatenate([f.ravel() for f in self.F])
        me.loops.add(len(loops))
        me.loops.foreach_set("vertex_index", loops.astype(np.int32))
        me.polygons.add(len(sizes))
        me.polygons.foreach_set("loop_start", np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32))
        uv = me.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv", np.concatenate([u.reshape(-1, 2) for u in self.UV]).astype(np.float32).ravel())
        col = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        col.data.foreach_set("color", np.vstack(self.C).astype(np.float32).ravel())
        me.polygons.foreach_set("material_index", np.concatenate(self.M).astype(np.int32))
        me.polygons.foreach_set("use_smooth", np.full(len(sizes), smooth))
        me.update()
        me.validate()
        for m in mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(name, me)
        if link:
            SC.collection.objects.link(ob)
        return ob


def grid_faces(nr, nc, wrap=False):
    cc = nc if wrap else nc - 1
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(cc), indexing="ij")
    j2 = (j + 1) % nc
    return np.stack([i * nc + j, i * nc + j2, (i + 1) * nc + j2, (i + 1) * nc + j], -1).reshape(-1, 4)


def tube(pts, radii, segs=8, cols=None):
    pts = np.asarray(pts, float)
    n = len(pts)
    radii = np.broadcast_to(np.asarray(radii, float), (n,))
    T = np.gradient(pts, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(T[0], a)
    u /= np.linalg.norm(u)
    U, W = [], []
    for i in range(n):
        u = u - T[i] * np.dot(u, T[i])
        u /= np.linalg.norm(u)
        U.append(u)
        W.append(np.cross(T[i], u))
    U, W = np.array(U), np.array(W)
    ang = 2 * np.pi * np.arange(segs) / segs
    V = pts[:, None] + radii[:, None, None] * (np.cos(ang)[None, :, None] * U[:, None] + np.sin(ang)[None, :, None] * W[:, None])
    F = grid_faces(n, segs, wrap=True)
    C = None if cols is None else np.repeat(np.asarray(cols, float).reshape(n, -1), segs, 0)
    return V.reshape(-1, 3), F, C


_ICO = {}


def ico(sub):
    if sub not in _ICO:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
        V = np.array([v.co[:] for v in bm.verts])
        F = np.array([[v.index for v in f.verts] for f in bm.faces])
        bm.free()
        _ICO[sub] = (V, F)
    return _ICO[sub]


def blob(center, radius, squash=0.8, seed=0, dark=(0.07, 0.18, 0.04), light=(0.42, 0.55, 0.12), sub=2, rough=0.28):
    V0, F = ico(sub)
    r = 1 + rough * (fbm(V0[:, 0] * 1.7 + seed * 3.1, V0[:, 1] * 1.7 + V0[:, 2] * 1.3, 3, seed) - 0.5) * 2
    V = np.asarray(center) + V0 * r[:, None] * radius * np.array([1, 1, squash])
    t = np.clip((V0[:, 2] + 1) / 2 + (fbm(V0[:, 0] * 3, V0[:, 1] * 3 + seed, 2, seed + 1) - 0.5) * 0.5, 0, 1) ** 1.3
    C = np.array(dark) * (1 - t[:, None]) + np.array(light) * t[:, None]
    return V, F, C


def box(center, size, rotz=0.0, top=(0.3, 0.3, 0.3), side=(0.25, 0.25, 0.25), tilt=(0, 0)):
    sx, sy, sz = np.asarray(size) / 2
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * [sx, sy, sz]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    R = np.array(mathutils.Euler((tilt[0], tilt[1], rotz)).to_matrix())
    V, F, C = [], [], []
    for fi, f in enumerate(faces):
        base = len(V)
        for vi in f:
            V.append(R @ corners[vi] + center)
        F.append([base, base + 1, base + 2, base + 3])
        c = top if fi == 1 else side
        C += [c] * 4
    return np.array(V), np.array(F), np.array(C)


# =====================================================================
# materials (glTF-friendly; render-only extras are added after export)
# =====================================================================
VC_MATS = []


def mat_vc(name, rough=0.6, sheen=0.0, emit=0.0, foliage=False, alpha=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Col"
    nt.links.new(vc.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    if sheen:
        b.inputs["Sheen Weight"].default_value = sheen
    if emit:
        nt.links.new(vc.outputs["Color"], b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = emit
    if alpha is not None:
        b.inputs["Alpha"].default_value = alpha
    m.use_backface_culling = False
    m["foliage"] = foliage
    VC_MATS.append(m)
    return m


M_BARK = mat_vc("bark", 0.85)
M_LEAF = mat_vc("foliage", 0.55, sheen=0.3, foliage=True)
M_BLOSSOM = mat_vc("blossom", 0.5, sheen=0.4, emit=0.35, foliage=True)
M_ROCK = mat_vc("rock", 0.8)
M_STONE = mat_vc("ruin_stone", 0.75)
M_RUNE = mat_vc("rune_glow", 0.4, emit=6.0)
M_PETAL = mat_vc("petal", 0.45, sheen=0.3, foliage=True)
M_GRASS = mat_vc("grass", 0.5, sheen=0.25, foliage=True)
M_PUFF = mat_vc("puff", 0.9, sheen=1.0)
M_LILY = mat_vc("lily", 0.3)
M_GLOW = mat_vc("glow", 0.5, emit=12.0)
M_TERRAIN = mat_vc("terrain", 0.9)


def mat_water(name, alpha=None, foam=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    if foam:
        b.inputs["Base Color"].default_value = (0.9, 0.95, 1.0, 1)
        b.inputs["Roughness"].default_value = 0.3
        b.inputs["Emission Color"].default_value = (0.8, 0.9, 1.0, 1)
        b.inputs["Emission Strength"].default_value = 0.4
    else:
        b.inputs["Base Color"].default_value = (0.02, 0.07, 0.08, 1)
        b.inputs["Roughness"].default_value = 0.04
        b.inputs["IOR"].default_value = 1.33
    if alpha is not None:
        b.inputs["Alpha"].default_value = alpha
    m.use_backface_culling = False
    return m


M_WATER = mat_water("water")
M_FALL = mat_water("waterfall", alpha=0.75, foam=True)

# =====================================================================
# asset library (all built at the origin, Z-up, metres)
# =====================================================================
LIB = {}


def lib(name, mb, mats, smooth=True):
    ob = mb.build(name, mats, smooth)
    LIB[name] = ob
    return ob


def make_oak(name, seed, H=11.0):
    r = np.random.default_rng(seed)
    mb = MB()
    zt = H * 0.45
    t = np.linspace(0, 1, 10)
    trunk = np.stack([0.4 * np.sin(t * 2 + seed), 0.3 * np.sin(t * 1.7 + seed * 2), t * zt], -1)
    rad = 0.45 * (1 - 0.35 * t) * (1 + 1.2 * np.exp(-t * zt / 0.7))
    bark = np.array([0.22, 0.17, 0.12]) * (0.8 + 0.4 * t[:, None])
    V, F, C = tube(trunk, rad, 12, bark)
    mb.add(V, F, C=C, mi=0)
    ends = []
    for b in range(r.integers(5, 7)):
        a = 2 * math.pi * b / 6 + r.uniform(-0.3, 0.3)
        el = math.radians(r.uniform(30, 55))
        L = H * r.uniform(0.3, 0.42)
        st = trunk[-1] - [0, 0, r.uniform(0, 1.2)]
        dvec = np.array([math.cos(a) * math.cos(el), math.sin(a) * math.cos(el), math.sin(el)])
        tt = np.linspace(0, 1, 6)[:, None]
        pts = st + dvec * L * tt + np.array([0, 0, -0.08 * L]) * tt ** 2
        V, F, C = tube(pts, np.linspace(0.22, 0.07, 6), 7, np.repeat([[0.24, 0.18, 0.13]], 6, 0))
        mb.add(V, F, C=C, mi=0)
        ends.append(pts[-1])
    crown = np.array([trunk[-1][0], trunk[-1][1], H * 0.66])
    for e in ends:
        mb.add(*_blob_args(blob(e + [0, 0, 0.6], r.uniform(2.0, 2.8), 0.75, r.integers(1e6))), mi=1)
    for _ in range(r.integers(5, 8)):
        off = r.normal(0, 1, 3) * [H * 0.18, H * 0.18, H * 0.08]
        mb.add(*_blob_args(blob(crown + off, r.uniform(2.2, 3.2), 0.75, r.integers(1e6))), mi=1)
    return lib(name, mb, [M_BARK, M_LEAF])


def _blob_args(vfc):
    V, F, C = vfc
    return V, F, None, C


def make_birch(name, seed, H=14.0):
    r = np.random.default_rng(seed)
    mb = MB()
    t = np.linspace(0, 1, 16)
    trunk = np.stack([0.5 * np.sin(t * 2.5 + seed), 0.3 * np.sin(t * 3 + seed), t * H], -1)
    band = (r.random(16) < 0.3)
    cols = np.where(band[:, None], [0.12, 0.11, 0.1], [0.85, 0.83, 0.78])
    V, F, C = tube(trunk, np.linspace(0.2, 0.05, 16), 9, cols)
    mb.add(V, F, C=C, mi=0)
    for _ in range(r.integers(8, 12)):
        z = r.uniform(0.45, 0.98)
        c = np.interp(z, t, trunk[:, 0]), np.interp(z, t, trunk[:, 1]), z * H
        off = np.array([r.normal(0, 1.2), r.normal(0, 1.2), r.normal(0, 0.5)])
        mb.add(*_blob_args(blob(np.array(c) + off, r.uniform(1.0, 1.7), 0.9, r.integers(1e6),
                                dark=(0.12, 0.2, 0.03), light=(0.55, 0.6, 0.12))), mi=1)
    return lib(name, mb, [M_BARK, M_LEAF])


def make_pine(name, seed, H=16.0):
    r = np.random.default_rng(seed)
    mb = MB()
    t = np.linspace(0, 1, 6)
    V, F, C = tube(np.stack([0 * t, 0 * t, t * H * 0.9], -1), np.linspace(0.35, 0.05, 6), 8,
                   np.repeat([[0.18, 0.12, 0.09]], 6, 0))
    mb.add(V, F, C=C, mi=0)
    layers = 8
    for li in range(layers):
        f = li / (layers - 1)
        z0 = H * (0.18 + 0.72 * f)
        R0 = (1 - f) * 3.6 + 0.5
        segs = 14
        ang = 2 * np.pi * np.arange(segs) / segs
        jag = 1 + 0.25 * r.standard_normal(segs)
        ring = np.stack([np.cos(ang) * R0 * jag, np.sin(ang) * R0 * jag, np.full(segs, z0 - 0.6)], -1)
        tip = np.array([[0, 0, z0 + H * 0.16]])
        V = np.vstack([ring, tip])
        F = np.array([[k, (k + 1) % segs, segs] for k in range(segs)])
        C = np.vstack([np.repeat([[0.03, 0.09, 0.06]], segs, 0), [[0.12, 0.24, 0.14]]])
        mb.add(V, F, C=C, mi=1)
    return lib(name, mb, [M_BARK, M_LEAF], smooth=False)


def make_sacred():
    r = np.random.default_rng(77)
    mb = MB()
    H = 34.0
    # three trunks spiralling into one
    for k in range(3):
        t = np.linspace(0, 1, 24)
        a = 2 * np.pi * k / 3 + t * 2.4
        rr = 2.2 * (1 - t) + 0.4
        pts = np.stack([rr * np.cos(a), rr * np.sin(a), t * 16], -1)
        V, F, C = tube(pts, np.linspace(1.5, 0.9, 24), 12, np.array([0.3, 0.26, 0.24]) * (0.7 + 0.5 * t[:, None]))
        mb.add(V, F, C=C, mi=0)
    # roots
    for k in range(9):
        a = 2 * np.pi * k / 9 + r.uniform(-0.2, 0.2)
        t = np.linspace(0, 1, 8)
        L = r.uniform(7, 11)
        pts = np.stack([np.cos(a) * (1.8 + L * t), np.sin(a) * (1.8 + L * t), 2.5 * (1 - t) ** 2 - 0.4], -1)
        V, F, C = tube(pts, np.linspace(0.9, 0.15, 8), 8, np.repeat([[0.26, 0.22, 0.19]], 8, 0))
        mb.add(V, F, C=C, mi=0)
    ends = []
    for k in range(8):
        a = 2 * np.pi * k / 8 + r.uniform(-0.25, 0.25)
        t = np.linspace(0, 1, 10)[:, None]
        L = r.uniform(12, 16)
        d = np.array([math.cos(a), math.sin(a), 0])
        pts = np.array([0, 0, 15]) + d * L * t + np.array([0, 0, 1]) * (9 * t - 3 * t ** 2)
        V, F, C = tube(pts, np.linspace(1.0, 0.25, 10), 9, np.repeat([[0.3, 0.25, 0.22]], 10, 0))
        mb.add(V, F, C=C, mi=0)
        ends.append(pts[-1])
    for e in ends:
        for _ in range(3):
            mb.add(*_blob_args(blob(e + r.normal(0, 2.2, 3) + [0, 0, 1.5], r.uniform(3.2, 4.6), 0.6, r.integers(1e6),
                                    dark=(0.55, 0.32, 0.42), light=(1.0, 0.86, 0.9))), mi=1)
    for _ in range(10):
        c = np.array([0, 0, 28]) + r.normal(0, 1, 3) * [7, 7, 2]
        mb.add(*_blob_args(blob(c, r.uniform(4, 6), 0.6, r.integers(1e6),
                                dark=(0.55, 0.32, 0.42), light=(1.0, 0.88, 0.92))), mi=1)
    return lib("sacred_tree", mb, [M_BARK, M_BLOSSOM])


def make_rock(name, seed, size=(2.2, 1.7, 1.2)):
    V0, F = ico(3)
    n = fbm(V0[:, 0] * 1.3 + seed, V0[:, 1] * 1.3 + V0[:, 2], 4, seed)
    r = 1 + (n - 0.5) * 0.7
    V = V0 * r[:, None] * size
    V[:, 2] -= size[2] * 0.3
    moss = smoothstep(0.25, 0.7, V0[:, 2] + (fbm(V0[:, 0] * 4, V0[:, 1] * 4, 2, seed + 3) - 0.5))
    base = np.array([0.32, 0.30, 0.28]) * (0.75 + 0.5 * n[:, None])
    C = base * (1 - moss[:, None]) + np.array([0.12, 0.2, 0.05]) * moss[:, None]
    mb = MB()
    mb.add(V, F, C=C)
    return lib(name, mb, [M_ROCK])


def make_monolith(name, seed, rune=True):
    V0, F = ico(3)
    n = fbm(V0[:, 0] * 2 + seed, V0[:, 1] * 2 + V0[:, 2] * 2, 3, seed)
    V = V0 * (1 + (n[:, None] - 0.5) * 0.15) * [0.9, 0.55, 3.6]
    V[:, 2] += 3.0
    V[:, :2] *= (1 - 0.25 * np.clip(V[:, 2] / 6.6, 0, 1))[:, None]
    moss = smoothstep(0.4, 0.9, V0[:, 2]) + 0.6 * smoothstep(-0.6, -0.95, V0[:, 2])
    C = np.array([0.36, 0.35, 0.33]) * (0.8 + 0.4 * n[:, None]) * (1 - moss[:, None]) + np.array([0.1, 0.2, 0.06]) * moss[:, None]
    mb = MB()
    mb.add(V, F, C=C, mi=0)
    if rune:  # thin glowing spiral carved on the face
        t = np.linspace(0, 3 * np.pi, 40)
        pts = np.stack([np.full(40, -0.58), 0.25 * t / 9 * np.cos(t), 4.2 + 0.25 * t / 9 * np.sin(t) * 1.3], -1)
        V2, F2, _ = tube(pts, 0.03, 4)
        mb.add(V2, F2, C=(0.5, 0.95, 1.0), mi=1)
    return lib(name, mb, [M_STONE, M_RUNE])


def make_arch():
    r = np.random.default_rng(5)
    mb = MB()
    top, side = (0.16, 0.25, 0.07), (0.42, 0.40, 0.36)
    for sx in (-10.5, 10.5):
        z = 0
        for k in range(6):
            h = 2.3
            V, F, C = box([sx + r.uniform(-0.2, 0.2), r.uniform(-0.2, 0.2), z + h / 2], [3.2, 3.2, h],
                          r.uniform(-0.1, 0.1), top, np.array(side) * r.uniform(0.85, 1.1))
            mb.add(V, F, C=C)
            z += h
    R = 10.5
    zc = 13.8
    n = 13
    for k in range(n):
        if k == 9:
            continue  # one fallen voussoir: the ruin is old
        a = math.pi * (k + 0.5) / n
        c = np.array([-R * math.cos(a), 0, zc + R * math.sin(a)])
        V, F, C = box(c, [2.6, 3.0, 2.3], 0, top, np.array(side) * r.uniform(0.85, 1.1),
                      tilt=(0, -(a - math.pi / 2)))
        mb.add(V, F, C=C)
    # glowing keystone rune (the wind gate)
    t = np.linspace(0, 2 * np.pi, 48)
    pts = np.stack([0.7 * np.cos(t), np.full(48, -1.55), zc + R - 0.3 + 0.7 * np.sin(t)], -1)
    V, F, _ = tube(pts, 0.06, 5)
    mb.add(V, F, C=(0.5, 0.95, 1.0), mi=1)
    return lib("ruin_arch", mb, [M_STONE, M_RUNE], smooth=False)


def make_column(name, seed):
    r = np.random.default_rng(seed)
    mb = MB()
    h = r.uniform(2.5, 6.5)
    t = np.linspace(0, 1, 8)
    V, F, C = tube(np.stack([0 * t, 0 * t, t * h], -1), 0.9 + 0.05 * np.sin(t * 20), 14,
                   np.array([0.45, 0.43, 0.39]) * (0.8 + 0.3 * t[:, None]))
    V = V.reshape(8, 14, 3)
    V[-1, :, 2] += r.uniform(-0.6, 0.4, 14)  # broken top
    mb.add(V.reshape(-1, 3), F, C=C)
    V, F, C = box([0, 0, 0.3], [2.4, 2.4, 0.6], 0, (0.16, 0.25, 0.07), (0.4, 0.38, 0.34))
    mb.add(V, F, C=C)
    return lib(name, mb, [M_STONE])


FLOWER_COLS = {
    "red": ((0.85, 0.08, 0.04), (0.05, 0.03, 0.02)),
    "yellow": ((1.0, 0.75, 0.08), (0.8, 0.45, 0.02)),
    "white": ((0.95, 0.95, 0.92), (1.0, 0.7, 0.05)),
    "violet": ((0.55, 0.28, 0.85), (0.95, 0.85, 0.3)),
    "blue": ((0.25, 0.4, 0.95), (0.95, 0.95, 0.9)),
}


def make_flower(col):
    petal, centre = FLOWER_COLS[col]
    mb = MB()
    t = np.linspace(0, 1, 4)
    stem = np.stack([0.02 * np.sin(t * 3), 0 * t, t * 0.5], -1)
    V, F, C = tube(stem, 0.009, 4, np.repeat([[0.12, 0.3, 0.05]], 4, 0))
    mb.add(V, F, C=C, mi=1)
    top = stem[-1]
    for k in range(6):
        a = 2 * np.pi * k / 6
        d = np.array([math.cos(a), math.sin(a), 0])
        s = np.array([-math.sin(a), math.cos(a), 0])
        rs = np.array([0.012, 0.045, 0.08])
        ws = np.array([0.012, 0.042, 0.02])
        zs = np.array([0.0, 0.012, 0.03])
        V = []
        for rr, ww, zz in zip(rs, ws, zs):
            c = top + d * rr + [0, 0, zz]
            V += [c - s * ww / 2, c + s * ww / 2]
        F = [[0, 1, 3, 2], [2, 3, 5, 4]]
        C = [np.array(petal) * 0.8] * 2 + [petal] * 4
        mb.add(V, F, C=C, mi=0)
    V0, F0 = ico(1)
    mb.add(top + V0 * [0.018, 0.018, 0.012] + [0, 0, 0.008], F0, C=centre, mi=0)
    for sgn in (1, -1):
        pts = np.array([[0, 0, 0.02], [sgn * 0.08, 0.02, 0.07], [sgn * 0.16, 0.03, 0.06]])
        s = np.array([0, 1.0, 0])
        V = np.concatenate([[p - s * w, p + s * w] for p, w in zip(pts, [0.005, 0.02, 0.001])])
        mb.add(V, [[0, 1, 3, 2], [2, 3, 5, 4]], C=(0.1, 0.28, 0.05), mi=1)
    return lib(f"flower_{col}", mb, [M_PETAL, M_GRASS])


def make_grass(name, seed, blades=11):
    r = np.random.default_rng(seed)
    mb = MB()
    for _ in range(blades):
        a = r.uniform(0, 2 * np.pi)
        d = np.array([math.cos(a), math.sin(a), 0])
        s = np.array([-math.sin(a), math.cos(a), 0])
        base = d * r.uniform(0, 0.1)
        h = r.uniform(0.45, 0.9)
        bend = r.uniform(0.1, 0.4)
        t = np.linspace(0, 1, 5)
        pts = base + d[None] * (bend * h * t ** 2)[:, None] + np.array([0, 0, 1.0]) * (h * t)[:, None]
        w = 0.03 * (1 - t) ** 0.8 + 0.001
        V = np.concatenate([[p - s * ww, p + s * ww] for p, ww in zip(pts, w)])
        F = [[2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2] for i in range(4)]
        tc = np.repeat(t, 2)[:, None]
        C = np.array([0.05, 0.14, 0.03]) * (1 - tc) + np.array([0.42, 0.52, 0.14]) * tc
        mb.add(V, F, C=C)
    return lib(name, mb, [M_GRASS])


def make_puff():
    mb = MB()
    t = np.linspace(0, 1, 4)
    V, F, C = tube(np.stack([0.015 * np.sin(t * 2), 0 * t, t * 0.3], -1), 0.004, 4, np.repeat([[0.3, 0.45, 0.2]], 4, 0))
    mb.add(V, F, C=C, mi=1)
    V0, F0 = ico(2)
    mb.add(V0 * 0.024 + [0.015 * math.sin(2), 0, 0.3], F0, C=(0.95, 0.95, 0.92), mi=0)
    return lib("dandelion_puff", mb, [M_PUFF, M_GRASS])


def make_lily(lotus=False):
    mb = MB()
    n = 22
    ang = np.linspace(0.3, 2 * np.pi - 0.3, n)
    ring = np.stack([np.cos(ang) * 0.6, np.sin(ang) * 0.6, np.full(n, 0.02)], -1)
    V = np.vstack([[[0, 0, 0.0]], ring])
    F = [[0, k + 1, k + 2] for k in range(n - 1)]
    C = np.vstack([[[0.1, 0.25, 0.06]], np.repeat([[0.18, 0.38, 0.08]], n, 0)])
    mb.add(V, F, C=C, mi=0)
    mats = [M_LILY]
    if lotus:
        for k in range(10):
            a = 2 * np.pi * k / 10
            d = np.array([math.cos(a), math.sin(a), 0])
            s = np.array([-math.sin(a), math.cos(a), 0])
            b = np.array([0, 0, 0.05]) + d * 0.04
            tip = b + d * 0.16 + [0, 0, 0.2]
            mid = (b + tip) / 2 + d * 0.03
            V = [b - s * 0.02, b + s * 0.02, mid - s * 0.06, mid + s * 0.06, tip, tip]
            mb.add(V, [[0, 1, 3, 2], [2, 3, 5, 4]], C=[(1, 0.95, 0.95)] * 2 + [(0.95, 0.55, 0.72)] * 4, mi=1)
        mats.append(M_PETAL)
    return lib("lotus" if lotus else "lily_pad", mb, mats)


def make_glow(name, radius, col):
    V0, F0 = ico(1)
    mb = MB()
    mb.add(V0 * radius, F0, C=col)
    return lib(name, mb, [M_GLOW])


log("building asset library")
for i in range(3):
    make_oak(f"oak_{'abc'[i]}", 10 + i)
for i in range(2):
    make_birch(f"birch_{'ab'[i]}", 20 + i)
for i in range(3):
    make_pine(f"pine_{'abc'[i]}", 30 + i)
make_sacred()
for i in range(4):
    make_rock(f"rock_{'abcd'[i]}", 40 + i, size=[(2.2, 1.7, 1.2), (1.4, 1.2, 1.1), (3.5, 2.6, 1.6), (1.0, 0.8, 0.6)][i])
make_monolith("monolith_a", 50)
make_monolith("monolith_b", 51, rune=False)
make_arch()
make_column("column_a", 60)
make_column("column_b", 61)
for c in FLOWER_COLS:
    make_flower(c)
make_grass("grass_a", 70)
make_grass("grass_b", 71, blades=7)
make_puff()
make_lily()
make_lily(lotus=True)
make_glow("firefly", 0.03, (1.0, 0.85, 0.35))
make_glow("spirit_mote", 0.05, (0.6, 0.95, 1.0))

# =====================================================================
# terrain meshes + vertex colours
# =====================================================================


def terrain_colors(X, Y, H, NZ, D):
    lush = np.array([0.10, 0.26, 0.05])
    warm = np.array([0.30, 0.38, 0.09])
    pat = fbm(X / 60, Y / 60, 3, 11)[..., None]
    grass = lush * (1 - pat) + warm * pat
    moss = np.array([0.06, 0.16, 0.04])
    g = (smoothstep(1150, 1250, X) * (1 - smoothstep(1690, 1700, X)) * (1 - smoothstep(30, 60, D)))[..., None]
    grass = grass * (1 - g) + moss * g
    rock = np.array([0.27, 0.25, 0.22]) * (0.75 + 0.35 * fbm(X / 20, Y / 20, 3, 12) + 0.12 * np.sin(H / 2.3 + fbm(X / 40, Y / 40, 2, 13) * 4))[..., None]
    rk = smoothstep(0.82, 0.6, NZ)[..., None]
    rk = np.maximum(rk, smoothstep(300, 380, H)[..., None])
    col = grass * (1 - rk) + rock * rk
    mud = np.array([0.30, 0.26, 0.18])
    wet = (1 - smoothstep(0.4, 1.6, H))[..., None] * (H > -1)[..., None]
    col = col * (1 - wet) + mud * wet
    col = np.where((H < -0.5)[..., None], np.array([0.12, 0.11, 0.08]), col)
    return np.concatenate([col, np.ones(col.shape[:-1] + (1,))], -1)


def terrain_mesh(name, stride):
    Hs, Xs, Ys = HG[::stride, ::stride], _XX[::stride, ::stride], _YY[::stride, ::stride]
    ny, nx = Hs.shape
    V = np.stack([Xs, Ys, Hs], -1).reshape(-1, 3)
    F = grid_faces(ny, nx)
    C = terrain_colors(Xs, Ys, Hs, NZG[::stride, ::stride], DG[::stride, ::stride]).reshape(-1, 4)
    uv_v = (np.stack([Xs, Ys], -1).reshape(-1, 2) / 10.0)
    mb = MB()
    mb.add(V, F, UV=uv_v[F], C=C)
    return mb.build(name, [M_TERRAIN])


log("building terrain meshes")
terrain_lod = terrain_mesh("terrain", 2)


def ribbon_along(mask, zfun, width, name):
    idx = np.where(mask)[0]
    c = P[idx]
    n = NRM[idx]
    z = zfun(c[:, 0])
    Vl = np.column_stack([c - n * width / 2, z])
    Vr = np.column_stack([c + n * width / 2, z])
    V = np.stack([Vl, Vr], 1).reshape(-1, 3)
    F = grid_faces(len(idx), 2)
    mb = MB()
    mb.add(V, F, UV=np.stack([V[:, 0], V[:, 1]], -1)[F] / 10)
    return mb.build(name, [M_WATER], smooth=True)


log("building water")
water = []
lk = MB()
xs, ys = np.linspace(640, 1240, 40), np.linspace(-200, 230, 30)
X, Y = np.meshgrid(xs, ys)
lk.add(np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3), grid_faces(30, 40))
water.append(lk.build("water_lake", [M_WATER]))
water.append(ribbon_along((P[:, 0] > 1180) & (P[:, 0] < 1712), river_z, 16, "water_river"))
pd = MB()
ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
pd.add(np.vstack([[POND_C[0], POND_C[1], STREAM_Z],
                  np.stack([POND_C[0] + np.cos(ang) * (POND_R + 3), POND_C[1] + np.sin(ang) * (POND_R + 3),
                            np.full(48, STREAM_Z)], -1)]),
       [[0, k + 1, (k + 1) % 48 + 1] for k in range(48)])
water.append(pd.build("water_pond", [M_WATER]))

# waterfall: find the cliff lip on the path
ci = int(np.argmax(P[:, 0] >= 1680))
lip_i = ci + int(np.argmax(ground(P[ci:ci + 80, 0], P[ci:ci + 80, 1]) > PLAT_H - 6))
lip = P[lip_i]
fall_dir = -TAN[lip_i]
base_z = float(river_z(lip[0] - 20))
wf = MB()
# cascade that hugs the cliff: march from the lip down-slope, 1.2 m off the rock
cols_ = 9
Vw = []
u = 0.0
rows = 0
while True:
    c = lip + fall_dir * u
    zg = float(ground(c[0], c[1]))
    z = min(STREAM_Z, zg + 1.2)
    w = 14 + 12 * np.clip((STREAM_Z - z) / (STREAM_Z - base_z), 0, 1)
    for j in range(cols_):
        q = j / (cols_ - 1) - 0.5
        Vw.append([*(c + NRM[lip_i] * q * w), z])
    rows += 1
    if zg <= base_z + 0.3 or u > 150:
        break
    u += 1.5
wf.add(Vw, grid_faces(rows, cols_))
waterfall = wf.build("waterfall", [M_FALL])
water.append(waterfall)
stream_mask = (np.arange(len(S)) >= lip_i) & (P[:, 0] < 2032)
water.append(ribbon_along(stream_mask, lambda x: np.full_like(x, STREAM_Z), 9, "water_stream"))
FALL_BASE = np.array([*(lip + fall_dir * 10), base_z])

# =====================================================================
# scattering
# =====================================================================
INST = {}


def put(name, x, y, z, rotz=None, scale=None, rot3=False, tint=None):
    n = len(x)
    if n == 0:
        return
    if rotz is None:
        rotz = RNG.uniform(0, 2 * np.pi, n)
    rot = np.zeros((n, 3))
    rot[:, 2] = rotz
    if rot3:
        rot[:, 0] = RNG.normal(0, 0.25, n)
        rot[:, 1] = RNG.normal(0, 0.25, n)
    sc = np.ones(n) if scale is None else np.broadcast_to(scale, (n,))
    tint = RNG.random(n) if tint is None else tint
    arr = np.column_stack([x, y, z, rot, sc, tint])
    INST[name] = np.vstack([INST[name], arr]) if name in INST else arr


def field(x, y):
    h = ground(x, y)
    return h, bil(NZG, x, y), bil(DG, x, y)


def band(n, half_width, s0=0.0, s1=None):
    s1 = PATH_LEN - 1 if s1 is None else s1
    s = RNG.uniform(s0, s1, n)
    i = s.astype(int)
    off = RNG.uniform(-half_width, half_width, n)
    return P[i, 0] + NRM[i, 0] * off, P[i, 1] + NRM[i, 1] * off


def area(n, x0, x1, y0, y1):
    return RNG.uniform(x0, x1, n), RNG.uniform(y0, y1, n)


def keep(p):
    return RNG.random(len(p)) < p


def pick(names, n):
    return RNG.integers(0, len(names), n)


log("scattering")
spawn = np.array([P[0, 0], P[0, 1], float(ground(*P[0]))])
# -- grass everywhere near the path (render only; games use a shader/density map)
x, y = band(900000 if not FAST else 250000, 120)
h, nz, d = field(x, y)
far_ok = smoothstep(120, 60, d) * 0.7 + 0.3
p = (nz > 0.8) * (~is_wet(x, y, h, d)) * far_ok * (0.5 + 0.5 * fbm(x / 25, y / 25, 2, 21))
p *= 1 - 0.8 * np.exp(-(np.hypot(x - spawn[0], y - spawn[1]) / 0.5) ** 2)  # small clearing around the mother
m = keep(p)
x, y, h = x[m], y[m], h[m]
v = RNG.random(len(x)) < 0.6
lushness = 0.7 + 0.5 * fbm(x / 40, y / 40, 2, 22)
put("grass_a", x[v], y[v], h[v] - 0.02, scale=(lushness * RNG.uniform(0.7, 1.3, len(x)))[v])
put("grass_b", x[~v], y[~v], h[~v] - 0.02, scale=(lushness * RNG.uniform(0.8, 1.5, len(x)))[~v])

# -- extra-dense grass in the flight lane (what the camera actually sees)
x, y = band(700000 if not FAST else 200000, 30)
h, nz, d = field(x, y)
p = (nz > 0.8) * (~is_wet(x, y, h, d)) * (0.55 + 0.45 * fbm(x / 25, y / 25, 2, 23))
p *= 1 - 0.9 * np.exp(-(np.hypot(x - spawn[0], y - spawn[1]) / 0.5) ** 2)
m = keep(p)
put("grass_a", x[m], y[m], ground(x[m], y[m]) - 0.02, scale=RNG.uniform(0.9, 1.7, m.sum()))

# -- flower patches (colour chosen per patch)
x, y = band(2000000 if not FAST else 600000, 70)
h, nz, d = field(x, y)
patch = fbm(x / 35, y / 35, 3, 31)
zonex = x
zone_w = np.where(zonex < 700, 1.0, np.where(zonex < 1160, 0.35, np.where(zonex < 1690, 0.15, 1.2)))
p = smoothstep(0.48, 0.62, patch) * zone_w * (nz > 0.8) * (~is_wet(x, y, h, d))
m = keep(np.clip(p, 0, 1))
x, y, h = x[m], y[m], h[m]
hue = fbm(x / 120, y / 120, 2, 32)
cols = list(FLOWER_COLS)
ci_ = np.clip((hue * 1.6 - 0.3) * len(cols), 0, len(cols) - 1).astype(int)
ci_ = np.where(RNG.random(len(x)) < 0.15, RNG.integers(0, len(cols), len(x)), ci_)
for k, c in enumerate(cols):
    mm = ci_ == k
    put(f"flower_{c}", x[mm], y[mm], h[mm], scale=RNG.uniform(1.4, 2.4, mm.sum()))

# -- dandelion puffs on the origin hill (siblings of the mother)
x, y = area(30000, -200, 260, -160, 160)
h, nz, d = field(x, y)
p = np.exp(-(np.hypot(x + 10, y) / 110) ** 2) * (fbm(x / 20, y / 20, 2, 41) > 0.45)
p *= 1 - np.exp(-(np.hypot(x - spawn[0], y - spawn[1]) / 1.5) ** 2)
m = keep(p)
put("dandelion_puff", x[m], y[m], h[m], scale=RNG.uniform(0.8, 1.3, m.sum()))
# a few right next to the mother for the opening shot
ang = RNG.uniform(0, 2 * np.pi, 14)
rr = RNG.uniform(1.2, 4.0, 14)
xx, yy = spawn[0] + np.cos(ang) * rr + 1.0, spawn[1] + np.sin(ang) * rr
put("dandelion_puff", xx, yy, ground(xx, yy), scale=RNG.uniform(0.9, 1.3, 14))

# -- oaks: meadow groves, lake shore, gorge giants, glade ring
x, y = area(60000, -300, 2400, -500, 500)
h, nz, d = field(x, y)
grove = smoothstep(0.52, 0.62, fbm(x / 90, y / 90, 3, 51))
meadow = (x < 1150) & (d > 28) & (d < 260)
gorge = (x > 1180) & (x < 1690) & (d > 26 + 20 * (x > 1540)) & (d < 75)
plat = (x > 1720) & (d > 30) & (d < 160)
p = np.where(meadow, 0.012 + 0.10 * grove, 0.0) + np.where(gorge, 0.20, 0) + np.where(plat, 0.03 + 0.1 * grove, 0)
p *= (nz > 0.7) * (~is_wet(x, y, h, d)) * (h > 0.6)
p *= smoothstep(70, 95, np.hypot(x - GLADE_C[0], y - GLADE_C[1]) + 30 * (x > 1720))
m = keep(np.clip(p, 0, 1))
x, y, h, d = x[m], y[m], h[m], d[m]
giant = (x > 1180) & (x < 1690)
scl = np.where(giant, RNG.uniform(1.8, 2.8, len(x)), RNG.uniform(0.8, 1.4, len(x)))
v = pick(["a", "b", "c"], len(x))
for k, c in enumerate("abc"):
    mm = v == k
    put(f"oak_{c}", x[mm], y[mm], h[mm] - 0.3, scale=scl[mm])
# glade ring of oaks (the sanctuary's walls)
ang = np.linspace(0, 2 * np.pi, 26, endpoint=False) + RNG.uniform(-0.08, 0.08, 26)
rr = RNG.uniform(68, 84, 26)
gx, gy = GLADE_C[0] + np.cos(ang) * rr, GLADE_C[1] + np.sin(ang) * rr
hd, _, gd = field(gx, gy)
okr = gd > 12
put("oak_a", gx[okr], gy[okr], hd[okr] - 0.3, scale=RNG.uniform(1.3, 1.9, okr.sum()))

# -- birches: lake shore + islands
x, y = area(30000, 620, 1260, -230, 260)
h, nz, d = field(x, y)
lm = lake_mask(x, y)
shore = (lm < 0.25) & (lm > 0.0) | ((h > 1.0) & (lm > 0.5))
p = shore * (h > 0.8) * (nz > 0.75) * 0.35
m = keep(p)
v = RNG.random(m.sum()) < 0.5
xm, ym, hm = x[m], y[m], h[m]
put("birch_a", xm[v], ym[v], hm[v] - 0.2, scale=RNG.uniform(0.8, 1.2, v.sum()))
put("birch_b", xm[~v], ym[~v], hm[~v] - 0.2, scale=RNG.uniform(0.8, 1.2, (~v).sum()))

# -- pines on the mountain flanks
x, y = area(200000 if not FAST else 80000, GX0 + 50, GX1 - 50, GY0 + 50, GY1 - 50)
h, nz, d = field(x, y)
p = smoothstep(130, 220, d) * (nz > 0.6) * (h < 320) * 0.45 * (0.4 + fbm(x / 150, y / 150, 2, 61))
m = keep(np.clip(p, 0, 1))
x, y, h = x[m], y[m], h[m]
v = pick("abc", len(x))
for k, c in enumerate("abc"):
    mm = v == k
    put(f"pine_{c}", x[mm], y[mm], h[mm] - 0.3, scale=RNG.uniform(0.8, 1.5, mm.sum()))

# -- rocks
x, y = area(120000, -400, 2500, -600, 600)
h, nz, d = field(x, y)
gz = (x > 1150) & (x < 1720)
p = (0.004 + 0.05 * gz * smoothstep(12, 30, d) + 0.08 * (nz < 0.75)) * (~is_wet(x, y, h, d)) * (d > 7)
m = keep(p)
x, y, h = x[m], y[m], h[m]
v = pick("abcd", len(x))
for k, c in enumerate("abcd"):
    mm = v == k
    put(f"rock_{c}", x[mm], y[mm], h[mm], scale=RNG.uniform(0.6, 2.2, mm.sum()), rot3=True)

# -- lily pads & lotus
x, y = area(60000, 650, 1220, -190, 220)
h, nz, d = field(x, y)
lm = lake_mask(x, y)
p = (lm > 0.6) * (h < -0.6) * smoothstep(0.55, 0.7, fbm(x / 18, y / 18, 2, 71)) * 0.6
p *= smoothstep(4, 12, d)  # keep the flight lane over open water
m = keep(p)
lot = RNG.random(m.sum()) < 0.06
xm, ym = x[m], y[m]
put("lily_pad", xm[~lot], ym[~lot], np.full((~lot).sum(), 0.01), scale=RNG.uniform(0.7, 1.5, (~lot).sum()))
put("lotus", xm[lot], ym[lot], np.full(lot.sum(), 0.012), scale=RNG.uniform(0.9, 1.3, lot.sum()))

# -- landmarks
# standing-stone ring: the path dips through it (wind gate #1)
ri = int(np.argmax(P[:, 0] >= RING_X))
ring_c = P[ri]
ang = np.linspace(0, 2 * np.pi, 9, endpoint=False) + 0.2
sx, sy = ring_c[0] + np.cos(ang) * 16, ring_c[1] + np.sin(ang) * 16
put("monolith_a", sx, sy, ground(sx, sy) - 0.4, rotz=ang + np.pi, scale=RNG.uniform(0.9, 1.2, 9))
x, y = band(4000, 200, 250, 1200)
h, nz, d = field(x, y)
m = keep((d > 30) * (nz > 0.8) * (~is_wet(x, y, h, d)) * 0.006)
put("monolith_b", x[m], y[m], h[m] - 0.5, scale=RNG.uniform(0.6, 1.0, m.sum()), rot3=True)
# ruin arch across the gorge (wind gate #2)
ai = int(np.argmax(P[:, 0] >= ARCH_X))
arch_c = P[ai]
arch_rot = math.atan2(TAN[ai, 1], TAN[ai, 0]) + np.pi / 2
put("ruin_arch", [arch_c[0]], [arch_c[1]], [float(ground(*arch_c)) + 0.0], rotz=[arch_rot - np.pi / 2 + np.pi / 2], scale=1.0)
cx = arch_c[0] + RNG.uniform(-60, 60, 10)
cy = arch_c[1] + RNG.choice([-1, 1], 10) * RNG.uniform(12, 30, 10)
put("column_a", cx, cy, ground(cx, cy) - 0.2, scale=RNG.uniform(0.8, 1.2, 10), rot3=True)
# sacred tree and shrine columns
put("sacred_tree", [SACRED[0]], [SACRED[1]], [float(ground(*SACRED)) - 0.5], rotz=[0.4], scale=1.0)
ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
cx, cy = SACRED[0] + np.cos(ang) * 30, SACRED[1] + np.sin(ang) * 30
okc = np.hypot(cx - POND_C[0], cy - POND_C[1]) > POND_R + 3
okc &= np.hypot(cx - LANDING[0], cy - LANDING[1]) > 8
put("column_b", cx[okc], cy[okc], ground(cx[okc], cy[okc]) - 0.2, scale=RNG.uniform(0.9, 1.3, okc.sum()))
# glowing life: fireflies in the gorge, spirit motes around the sacred tree
x, y = band(2500, 50, s_at_x(1180), s_at_x(1690))
put("firefly", x, y, ground(x, y) + RNG.uniform(0.5, 16, len(x)) ** 1.0, scale=RNG.uniform(0.7, 1.4, len(x)))
ang = RNG.uniform(0, 2 * np.pi, 900)
rr = np.sqrt(RNG.uniform(0, 1, 900)) * 45
mx, my = SACRED[0] + np.cos(ang) * rr, SACRED[1] + np.sin(ang) * rr
put("spirit_mote", mx, my, ground(mx, my) + RNG.uniform(0.5, 30, 900), scale=RNG.uniform(0.5, 1.3, 900))
for k, v in INST.items():
    log(f"  {k:16s} {len(v):7d}")

# =====================================================================
# game data export
# =====================================================================


def to_gl(p):
    """Blender Z-up (x, y, z) -> glTF Y-up (x, z, -y)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def rnd(a, n=2):
    return np.round(np.asarray(a, float), n).tolist()


os.makedirs(RENDER_DIR, exist_ok=True)
log("exporting glb")


def export(path, objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True,
                              export_vertex_color="ACTIVE", export_all_vertex_colors=False)


export(os.path.join(HERE, "stage1_terrain.glb"), [terrain_lod] + water)
export(os.path.join(HERE, "stage1_props.glb"), list(LIB.values()))

JSON_CAP = 12000  # per prop type; games densify flowers/puffs around listed points
WIND = [
    {"type": "breeze_ribbon", "s": [s_at_x(250), s_at_x(420)], "boost": 1.4},
    {"type": "gate", "name": "stone_ring", "s": [s_at_x(RING_X - 20), s_at_x(RING_X + 20)], "boost": 1.8},
    {"type": "breeze_ribbon", "s": [s_at_x(820), s_at_x(1000)], "boost": 1.3},
    {"type": "gate", "name": "ruin_arch", "s": [s_at_x(ARCH_X - 15), s_at_x(ARCH_X + 15)], "boost": 1.8},
    {"type": "updraft", "name": "waterfall", "s": [s_at_x(1620), s_at_x(1760)], "lift": 6.0},
    {"type": "calm", "name": "landing", "s": [PATH_LEN - 140, PATH_LEN], "boost": 0.6},
]
layout = {
    "stage": "stage1_valley_of_first_wind",
    "units": "metres, glTF Y-up (x, y=up, z)",
    "duration_s": 300,
    "path_length_m": round(float(PATH_LEN), 1),
    "spawn": rnd(to_gl([*P[0], ALT[0]])),
    "landing": rnd(to_gl([*LANDING, float(ground(*LANDING)) + 0.3])),
    "lighting": {"sun_elevation_deg": 9, "sun_azimuth_deg_from_x": 15,
                 "sun_color": [1.0, 0.78, 0.55], "fog_density": 0.0006, "fog_color": [0.9, 0.82, 0.72],
                 "note": "golden hour; the player always flies toward the sun"},
    "zones": [{"id": z[0], "name_ko": z[1], "name_en": z[2],
               "s": [float(S[zone_idx == i][0]), float(S[zone_idx == i][-1])],
               "t": [round(float(TIME[zone_idx == i][0]), 1), round(float(TIME[zone_idx == i][-1]), 1)],
               "speed_mps": round(float(SPEED[zone_idx == i].mean()), 2)} for i, z in enumerate(ZONES)],
    "wind": WIND,
    "landmarks": {
        "mother_dandelion": rnd(to_gl(spawn)),
        "stone_ring": rnd(to_gl([*ring_c, float(ground(*ring_c))])),
        "ruin_arch": rnd(to_gl([*arch_c, float(ground(*arch_c))])),
        "waterfall_lip": rnd(to_gl([*lip, STREAM_Z])),
        "sacred_tree": rnd(to_gl([*SACRED, float(ground(*SACRED))])),
        "spring_pond": rnd(to_gl([*POND_C, STREAM_Z])),
    },
    "path": {"step_m": 2, "s": rnd(S[::2], 1), "t": rnd(TIME[::2], 2),
             "pos": rnd(to_gl(np.column_stack([P[::2], ALT[::2]]))),
             "zone": zone_idx[::2].tolist()},
    "instances": {},
    "grass_rule": "grass is not listed: spawn in shader within 120 m of path where slope normal.y > 0.8 and not wet",
}
for k, v in INST.items():
    if k.startswith("grass"):
        continue
    keep_n = min(len(v), JSON_CAP)
    v = v[RNG.permutation(len(v))[:keep_n]] if keep_n < len(v) else v
    layout["instances"][k] = {"pos": rnd(to_gl(v[:, 0:3]), 1), "rot_y": rnd(v[:, 5], 2), "scale": rnd(v[:, 6], 2),
                              "density_multiplier": round(len(INST[k]) / keep_n, 2)}
with open(os.path.join(HERE, "stage1_layout.json"), "w") as f:
    json.dump(layout, f, separators=(",", ":"))
log("layout written", {z["id"]: z["t"] for z in layout["zones"]})

if NO_RENDER:
    raise SystemExit

# =====================================================================
# render scene
# =====================================================================
log("setting up render scene")
bpy.data.objects.remove(terrain_lod)
terrain_hi = terrain_mesh("terrain_hi", 1)


def enhance_vc(m):
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    vc = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeVertexColor")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "INSTANCER"
    at.attribute_name = "tint"
    ma = nt.nodes.new("ShaderNodeMath")
    ma.operation = "MULTIPLY_ADD"
    nt.links.new(at.outputs["Fac"], ma.inputs[0])
    ma.inputs[1].default_value = 0.45
    ma.inputs[2].default_value = 0.78
    sc_ = nt.nodes.new("ShaderNodeVectorMath")
    sc_.operation = "SCALE"
    nt.links.new(vc.outputs["Color"], sc_.inputs[0])
    nt.links.new(ma.outputs[0], sc_.inputs["Scale"])
    nt.links.new(sc_.outputs[0], b.inputs["Base Color"])
    if m.get("foliage"):
        out = nt.nodes["Material Output"]
        tr = nt.nodes.new("ShaderNodeBsdfTranslucent")
        nt.links.new(sc_.outputs[0], tr.inputs["Color"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = 0.35
        nt.links.new(b.outputs[0], mix.inputs[1])
        nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])


for m in VC_MATS:
    if m is not M_TERRAIN:
        enhance_vc(m)

# terrain: vertex colour x large + small noise, bump
nt = M_TERRAIN.node_tree
b = nt.nodes["Principled BSDF"]
vc = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeVertexColor")
n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 0.08
n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 2.5
tc = nt.nodes.new("ShaderNodeTexCoord")
nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY_ADD"
nt.links.new(n1.outputs["Fac"], mul.inputs[0]); mul.inputs[1].default_value = 0.6; mul.inputs[2].default_value = 0.72
mul2 = nt.nodes.new("ShaderNodeMath"); mul2.operation = "MULTIPLY_ADD"
nt.links.new(n2.outputs["Fac"], mul2.inputs[0]); mul2.inputs[1].default_value = 0.35; mul2.inputs[2].default_value = 0.0
nt.links.new(mul.outputs[0], mul2.inputs[2])
scl = nt.nodes.new("ShaderNodeVectorMath"); scl.operation = "SCALE"
nt.links.new(vc.outputs["Color"], scl.inputs[0]); nt.links.new(mul2.outputs[0], scl.inputs["Scale"])
nt.links.new(scl.outputs[0], b.inputs["Base Color"])
bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.25
nt.links.new(n2.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])

# water ripples
for wm in (M_WATER,):
    nt = wm.node_tree
    b = nt.nodes["Principled BSDF"]
    nz_ = nt.nodes.new("ShaderNodeTexNoise"); nz_.inputs["Scale"].default_value = 1.5
    nz_.inputs["Detail"].default_value = 6
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.04
    nt.links.new(nz_.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
# waterfall streaks
nt = M_FALL.node_tree
b = nt.nodes["Principled BSDF"]
wv = nt.nodes.new("ShaderNodeTexNoise"); wv.inputs["Scale"].default_value = 3
mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1.2, 1.2, 0.06)
tcw = nt.nodes.new("ShaderNodeTexCoord")
nt.links.new(tcw.outputs["Object"], mp.inputs["Vector"]); nt.links.new(mp.outputs[0], wv.inputs["Vector"])
mra = nt.nodes.new("ShaderNodeMapRange"); mra.inputs[1].default_value = 0.35; mra.inputs[2].default_value = 0.65
mra.inputs[3].default_value = 0.25
nt.links.new(wv.outputs["Fac"], mra.inputs[0]); nt.links.new(mra.outputs[0], b.inputs["Alpha"])

# move library sources far away (they are referenced by geometry nodes)
for ob in LIB.values():
    ob.location = (0, 0, -5000)


def make_inst_group(src):
    ng = bpy.data.node_groups.new("inst_" + src.name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    iop = N.new("GeometryNodeInstanceOnPoints")
    oi = N.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = src
    oi.inputs["As Instance"].default_value = True
    rn = N.new("GeometryNodeInputNamedAttribute"); rn.data_type = "FLOAT_VECTOR"; rn.inputs["Name"].default_value = "rot"
    sn = N.new("GeometryNodeInputNamedAttribute"); sn.data_type = "FLOAT"; sn.inputs["Name"].default_value = "scl"
    e2r = N.new("FunctionNodeEulerToRotation")
    L.new(gi.outputs[0], iop.inputs["Points"])
    L.new(oi.outputs["Geometry"], iop.inputs["Instance"])
    L.new(rn.outputs["Attribute"], e2r.inputs["Euler"])
    L.new(e2r.outputs["Rotation"], iop.inputs["Rotation"])
    L.new(sn.outputs["Attribute"], iop.inputs["Scale"])
    L.new(iop.outputs["Instances"], go.inputs[0])
    return ng


for name, arr in INST.items():
    me = bpy.data.meshes.new(name + "_pts")
    me.vertices.add(len(arr))
    me.vertices.foreach_set("co", arr[:, 0:3].astype(np.float32).ravel())
    a = me.attributes.new("rot", "FLOAT_VECTOR", "POINT"); a.data.foreach_set("vector", arr[:, 3:6].astype(np.float32).ravel())
    a = me.attributes.new("scl", "FLOAT", "POINT"); a.data.foreach_set("value", arr[:, 6].astype(np.float32))
    a = me.attributes.new("tint", "FLOAT", "POINT"); a.data.foreach_set("value", arr[:, 7].astype(np.float32))
    ob = bpy.data.objects.new(name + "_scatter", me)
    SC.collection.objects.link(ob)
    ob.modifiers.new("inst", "NODES").node_group = make_inst_group(LIB[name])

# the hero: mother dandelion
if os.path.exists(DANDELION_GLB):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=DANDELION_GLB)
    for ob in set(bpy.data.objects) - before:
        if ob.parent is None:
            ob.location = mathutils.Vector(ob.location) + mathutils.Vector(spawn)

# ---- sky, sun, atmosphere
SUN_EL, SUN_AZ = math.radians(9), math.radians(15)  # azimuth measured from +X toward +Y
world = bpy.data.worlds.new("sky")
SC.world = world
world.use_nodes = True
wn = world.node_tree
sky = wn.nodes.new("ShaderNodeTexSky")
sky.sky_type = "MULTIPLE_SCATTERING"
sky.sun_elevation = SUN_EL
sky.sun_rotation = math.pi / 2 - SUN_AZ  # Blender sky: rotation 0 -> +Y, 90 deg -> +X
sky.sun_disc = True
sky.sun_intensity = 0.4
wn.links.new(sky.outputs[0], wn.nodes["Background"].inputs[0])
wn.nodes["Background"].inputs[1].default_value = 0.25

to_sun = mathutils.Vector((math.cos(SUN_EL) * math.cos(SUN_AZ), math.cos(SUN_EL) * math.sin(SUN_AZ), math.sin(SUN_EL)))
sun = bpy.data.lights.new("sun", "SUN")
sun.energy = 3.2
sun.color = (1.0, 0.78, 0.55)
sun.angle = math.radians(1.0)
so = bpy.data.objects.new("sun", sun)
SC.collection.objects.link(so)
so.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()


def vol_box(name, loc, size, density, color=(1, 1, 1), aniso=0.5, noise=None):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    ob = bpy.context.object
    ob.name = name
    ob.scale = [s / 2 for s in size]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.bl_idname != "ShaderNodeOutputMaterial":
            nt.nodes.remove(n)
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = (*color, 1)
    pv.inputs["Anisotropy"].default_value = aniso
    pv.inputs["Density"].default_value = density
    if noise:
        tcn = nt.nodes.new("ShaderNodeTexCoord")
        nz_ = nt.nodes.new("ShaderNodeTexNoise"); nz_.inputs["Scale"].default_value = noise
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tcn.outputs["Object"], nz_.inputs["Vector"]); nt.links.new(tcn.outputs["Object"], sep.inputs[0])
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs[1].default_value = 0.45; mr.inputs[2].default_value = 0.75
        nt.links.new(nz_.outputs["Fac"], mr.inputs[0])
        fall = nt.nodes.new("ShaderNodeMapRange"); fall.inputs[1].default_value = 1.0; fall.inputs[2].default_value = -1.0
        nt.links.new(sep.outputs["Z"], fall.inputs[0])
        m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"
        nt.links.new(mr.outputs[0], m1.inputs[0]); nt.links.new(fall.outputs[0], m1.inputs[1])
        m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "MULTIPLY"; m2.inputs[1].default_value = density
        nt.links.new(m1.outputs[0], m2.inputs[0]); nt.links.new(m2.outputs[0], pv.inputs["Density"])
    nt.links.new(pv.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
    ob.data.materials.append(m)
    ob.visible_shadow = False
    return ob


vol_box("atmosphere", (1000, 0, 150), (4200, 2800, 420), 0.00055, color=(0.95, 0.88, 0.8), aniso=0.5)
vol_box("lake_mist", (930, 15, 3), (560, 420, 8), 0.06, noise=0.012)
vol_box("fall_spray", tuple(FALL_BASE + [0, 0, 8]), (40, 40, 22), 0.04, noise=0.08)

cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
SC.collection.objects.link(cam)
SC.camera = cam
cam_data.clip_end = 6000
cam_data.clip_start = 0.02

SC.render.engine = "CYCLES"
SC.cycles.device = "CPU"
SC.cycles.use_denoising = True
SC.cycles.max_bounces = 4
SC.cycles.diffuse_bounces = 2
SC.cycles.glossy_bounces = 2
SC.cycles.transmission_bounces = 4
SC.cycles.volume_bounces = 0
SC.cycles.transparent_max_bounces = 8
SC.cycles.sample_clamp_indirect = 4
SC.cycles.volume_step_rate = 4
SC.view_settings.view_transform = "AgX"
SC.view_settings.look = "AgX - Medium High Contrast"
SC.view_settings.exposure = -0.2
SC.render.resolution_x, SC.render.resolution_y = (640, 360) if FAST else (1280, 720)
SC.cycles.samples = 16 if FAST else 64


def aim(pos, target, lens, focus=None, fstop=None):
    cam.location = pos
    d = mathutils.Vector(target) - mathutils.Vector(pos)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = lens
    cam_data.dof.use_dof = focus is not None
    if focus is not None:
        cam_data.dof.focus_distance = focus
        cam_data.dof.aperture_fstop = fstop


g0 = spawn[2]
fall_top = np.array([*lip, STREAM_Z])
SHOTS = [
    ("01_dawn_hill", spawn + [-0.17, 0.06, 0.262], spawn + [40, -3, -2.5], 26, 0.19, 5.6),
    ("02_whisper_meadow", path_point(RING_X - 75, lat=6, up=-3.0), [*ring_c, float(ground(*ring_c)) + 4], 24, None, None),
    ("03_mist_lake", path_point(760, up=-1.0), path_point(1000, up=-2.0), 22, None, None),
    ("04_ancient_gorge", path_point(ARCH_X - 70, up=-2), [*arch_c, float(ground(*arch_c)) + 12], 24, None, None),
    ("05_waterfall", path_point(1600, lat=-14, ref="ground", up=12), fall_top - [0, 0, 32], 20, None, None),
    ("06_sanctuary_glade", path_point(1975, up=12), [SACRED[0], SACRED[1], float(ground(*SACRED)) + 14], 26, None, None),
    ("07_overview", np.array([2300.0, 480.0, 400.0]), np.array([1000.0, 0.0, 0.0]), 26, None, None),
]
for i, (name, pos, tgt, lens, focus, fstop) in enumerate(SHOTS, 1):
    if ONLY_SHOTS and i not in ONLY_SHOTS:
        continue
    aim(tuple(pos), tuple(tgt), lens, focus, fstop)
    SC.render.filepath = os.path.join(RENDER_DIR, name + ("_fast" if FAST else "") + ".png")
    log("render", name)
    bpy.ops.render.render(write_still=True)
log("done")
