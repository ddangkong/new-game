"""Stage 1 world definition (numpy only, no Blender).

Shared by the game-data exporter (build_stage1.py) and the hero renderer
(render_stage1.py) so the flight path, terrain and zones never drift apart.
Coordinates: Blender Z-up, metres. x runs roughly along the flight.
"""
import numpy as np


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


def i_at_x(xv):
    return int(np.argmax(P[:, 0] >= xv))


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
WATER_Z = 0.0


def floor_z(x):
    return np.interp(x, [1100, 1200, 1690], [2.5, 3.0, 9.0])


def river_z(x):
    return floor_z(x) - 1.4


STREAM_Z = PLAT_H - 0.7


def lake_mask(x, y):
    r = np.sqrt(((x - LAKE_C[0]) / LAKE_R[0]) ** 2 + ((y - LAKE_C[1]) / LAKE_R[1]) ** 2)
    r = r + (fbm(x / 90, y / 90, 3, 5) - 0.5) * 0.35
    return 1 - smoothstep(0.82, 1.0, r)


FALL_Y = 12.0  # the waterfall pours over the cliff where the path crosses it
FALL_X = 1702.0


def plateau_mask(x, y):
    nx = x + (fbm(y / 60, x / 60, 3, 9) - 0.5) * 40
    soft = smoothstep(1690, 1712, nx)
    # near the waterfall the cliff becomes a sheer wall with a clean lip
    sheer = smoothstep(FALL_X - 1.5, FALL_X + 1.5, x + (fbm(y / 15, 3.0, 2, 19) - 0.5) * 3)
    wf = np.exp(-((y - FALL_Y) / 45.0) ** 2)
    return soft * (1 - wf) + sheer * wf


def height(x, y, d=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if d is None:
        d = path_query(x, y)[0].reshape(np.shape(x))
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
          - 3 * np.exp(-(d / 7) ** 2)
          - 3.5 * np.exp(-(((x - (FALL_X - 12)) / 14) ** 2 + ((y - FALL_Y) / 16) ** 2)))  # plunge pool
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


# ---- heightfield grid (4 m) for fast lookups --------------------------------
GX0, GX1, GY0, GY1, STEP = -600.0, 2700.0, -1000.0, 1000.0, 4.0
GXS = np.arange(GX0, GX1 + 0.1, STEP)
GYS = np.arange(GY0, GY1 + 0.1, STEP)
XX, YY = np.meshgrid(GXS, GYS)
_D, _SS = path_query(XX, YY)
DG = _D.reshape(XX.shape)
SG = _SS.reshape(XX.shape)
HG = height(XX, YY, DG)
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


def water_level(x, y, d=None):
    """Water surface height where there is water, else -inf."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if d is None:
        d = bil(DG, x, y)
    wl = np.full(np.shape(x), -np.inf)
    wl = np.where((x > 600) & (x < 1230) & (lake_mask(x, y) > 0.05), WATER_Z, wl)
    wl = np.where((x > 1150) & (x < 1705) & (d < 9), river_z(x), wl)
    wl = np.where((x > 1700) & (x < 2030) & (d < 5), STREAM_Z, wl)
    wl = np.where(np.hypot(x - POND_C[0], y - POND_C[1]) < POND_R + 3, STREAM_Z, wl)
    return wl


# =====================================================================
# zones, flight altitude and timing (exactly 300 s)
# =====================================================================
ZONES = [  # id, name_ko, name_en, x_end, relative cruise speed
    ("origin_hill", "새벽 언덕", "Dawn Hill", 200, 4.5),
    ("whisper_meadow", "속삭이는 초원", "Whispering Meadow", 700, 7.5),
    ("mist_lake", "안개 호수", "Mist Lake", 1160, 6.0),
    ("ancient_gorge", "고대 숲 협곡", "Ancient Gorge", 1560, 7.0),
    ("waterfall_ascent", "빛의 폭포", "Waterfall of Light", 1800, 6.5),
    ("sanctuary_glade", "성소의 숲", "Sanctuary Glade", 1e9, 5.0),
]
ZONE_IDX = np.zeros(len(S), int)
for _zi in range(len(ZONES) - 1, -1, -1):
    ZONE_IDX[P[:, 0] < ZONES[_zi][3]] = _zi

_gp = ground(P[:, 0], P[:, 1])
_gp = np.where((P[:, 0] > 1150) & (P[:, 0] < 1700), np.maximum(_gp, river_z(P[:, 0])), _gp)
_gp = np.maximum(_gp, 0.0)
GROUND_PATH = _gp
# look-ahead envelope so the seed rises *before* the cliff instead of into it
_env = np.array([_gp[max(0, i - 40):i + 90].max() for i in range(len(S))])
_clear = np.select(
    [ZONE_IDX == 0, ZONE_IDX == 1, ZONE_IDX == 2, ZONE_IDX == 3, ZONE_IDX == 4],
    [9.0, 7 + 4 * np.sin(S / 60), 4.5, 11.0, 18.0], 16.0)
_clear = _clear - 5.0 * np.exp(-((P[:, 0] - RING_X) / 25) ** 2)  # dip through the stone ring
_alt = _env + _clear
_k = np.exp(-0.5 * (np.arange(-60, 61) / 20.0) ** 2)
_k /= _k.sum()
_alt = np.convolve(np.pad(_alt, 60, mode="edge"), _k, mode="valid")
_alt = np.maximum(_alt, _gp + 2.5)
_a0 = smoothstep(0, 70, S)
_alt = _alt * _a0 + (_gp[0] + 0.26) * (1 - _a0)  # take-off from the mother dandelion
_a1 = smoothstep(PATH_LEN - 140, PATH_LEN, S)
_alt = _alt * (1 - _a1) + (_gp[-1] + 0.3) * _a1  # gentle landing
ALT = _alt

_rel_v = np.array([ZONES[z][4] for z in ZONE_IDX], float)
_rel_v = np.convolve(np.pad(_rel_v, 30, mode="edge"), np.ones(61) / 61, mode="valid")
_seg3 = np.sqrt(1 + np.gradient(ALT) ** 2)
_t_raw = np.concatenate([[0], np.cumsum(_seg3[1:] / _rel_v[1:])])
SPEED_SCALE = _t_raw[-1] / 300.0
TIME = _t_raw / SPEED_SCALE
SPEED = _rel_v * SPEED_SCALE


def path_point(xv, lat=0.0, up=0.0, ref="path"):
    i = i_at_x(xv)
    b = P[i] + NRM[i] * lat
    z = (ALT[i] if ref == "path" else float(ground(b[0], b[1]))) + up
    return np.array([b[0], b[1], z])


SPAWN = np.array([P[0, 0], P[0, 1], float(ground(*P[0]))])
