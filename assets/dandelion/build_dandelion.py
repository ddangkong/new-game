"""Procedural dandelion seed head ("clock") -> dandelion.glb + preview renders.

Real-world scale (metres). Every pappus hair is real geometry (thin ribbons),
not an alpha card, so the head reads as soft white fluff from any angle.

Run: python3 build_dandelion.py            (glb + both previews)
     python3 build_dandelion.py --no-render (glb only)
"""
import math
import os
import sys

import bpy
import mathutils
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11)

bpy.ops.wm.read_factory_settings(use_empty=True)


# ---------------------------------------------------------------- textures
def make_image(name, px):
    h, w, _ = px.shape
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels = px.astype(np.float32).ravel()
    img.pack()
    return img


def noise2(shape, scale, seed):
    """Cheap smooth value noise (bilinear upsampled random grid)."""
    r = np.random.default_rng(seed)
    gh, gw = max(2, shape[0] // scale), max(2, shape[1] // scale)
    g = r.random((gh + 1, gw + 1))
    y = np.linspace(0, gh, shape[0])
    x = np.linspace(0, gw, shape[1])
    y0, x0 = np.floor(y).astype(int).clip(0, gh - 1), np.floor(x).astype(int).clip(0, gw - 1)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0]; b = g[y0][:, x0 + 1]; c = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def tex_stem():
    h, w = 512, 64
    v = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    u = np.linspace(0, 1, w)[None, :] * np.ones((h, 1))
    ribs = 0.06 * np.sin(u * 2 * math.pi * 9) + 0.04 * noise2((h, w), 16, 1)
    g = 0.55 + ribs
    col = np.stack([0.48 * g, 0.78 * g, 0.36 * g], -1)
    # hollow scape: pale near the head, purplish-red flush at the base
    base = np.clip(1 - v / 0.35, 0, 1)[..., None] ** 1.5
    col = col * (1 - base) + np.array([0.42, 0.22, 0.20]) * base
    top = np.clip((v - 0.9) / 0.1, 0, 1)[..., None]
    col = col * (1 - top) + np.array([0.55, 0.62, 0.35]) * top
    return np.dstack([col, np.ones((h, w))])


def tex_leaf():
    n = 1024
    v, u = np.mgrid[0:n, 0:n] / n  # u across (0.5 = midrib), v along
    du = np.abs(u - 0.5)
    mid = np.exp(-(du / (0.012 + 0.02 * (1 - v))) ** 2)
    # secondary veins: run out from the midrib toward the tip at an angle
    phase = (v * 9 - du * 3.5) % 1.0
    veins = np.exp(-((phase - 0.5) / 0.05) ** 2) * np.clip(du / 0.03, 0, 1) * 0.6
    mott = noise2((n, n), 48, 2) * 0.10 + noise2((n, n), 8, 3) * 0.05
    base = np.stack([0.16, 0.36, 0.09]) * (0.85 + mott[..., None])
    col = base + veins[..., None] * np.array([0.05, 0.08, 0.03])
    ribc = np.array([0.62, 0.72, 0.42]) * (1 - v[..., None] ** 0.3) + np.array([0.55, 0.28, 0.25]) * v[..., None] ** 0.3
    ribc = np.array([0.62, 0.72, 0.42])
    col = col * (1 - mid[..., None]) + ribc * mid[..., None]
    # reddish midrib toward the base
    red = np.clip(1 - v / 0.3, 0, 1)[..., None] * mid[..., None]
    col = col * (1 - red) + np.array([0.5, 0.2, 0.2]) * red
    return np.dstack([col, np.ones((n, n))])


def tex_seed():
    h, w = 256, 32
    v = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    u = np.linspace(0, 1, w)[None, :] * np.ones((h, 1))
    # v in [0,0.3): achene (olive-brown, ribbed, tiny spines), rest: beak (pale)
    ribs = 0.12 * np.sin(u * 2 * math.pi * 6) + 0.08 * np.sin(v * 180) * (np.sin(u * 2 * math.pi * 6) > 0.6)
    ach = np.stack([0.40 + ribs, 0.34 + ribs, 0.18 + ribs * 0.5], -1)
    beak = np.stack([0.86, 0.85, 0.78]) * np.ones((h, w, 1))
    t = np.clip((v - 0.28) / 0.04, 0, 1)[..., None]
    col = ach * (1 - t) + beak * t
    return np.dstack([col, np.ones((h, w))])


def tex_bract():
    h, w = 128, 32
    v = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    g = 0.9 + 0.1 * noise2((h, w), 8, 5)
    col = np.stack([0.20 * g, 0.34 * g, 0.12 * g], -1)
    tip = np.clip((v - 0.75) / 0.25, 0, 1)[..., None]
    col = col * (1 - tip) + np.array([0.28, 0.16, 0.10]) * tip
    return np.dstack([col, np.ones((h, w))])


def material(name, px=None, color=None, rough=0.6, sheen=0.0, trans=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    if px is not None:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = make_image(name + "_tex", px)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    if color is not None:
        bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    if sheen:
        bsdf.inputs["Sheen Weight"].default_value = sheen
    if trans:
        bsdf.inputs["Transmission Weight"].default_value = trans
    m.use_backface_culling = False
    return m


# ---------------------------------------------------------------- geometry
class Builder:
    def __init__(self):
        self.v, self.f, self.uv = [], [], []

    def add(self, verts, faces, uvs):
        o = len(self.v)
        self.v.extend(tuple(p) for p in verts)
        self.f.extend(tuple(i + o for i in fc) for fc in faces)
        self.uv.extend(uvs)

    def build(self, name, mat, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        uvl = me.uv_layers.new(name="UVMap")
        flat = [uv for fu in self.uv for uv in fu]
        uvl.data.foreach_set("uv", np.asarray(flat, np.float32).ravel())
        me.materials.append(mat)
        me.polygons.foreach_set("use_smooth", [smooth] * len(me.polygons))
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def frame(d):
    d = d / np.linalg.norm(d)
    a = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def tube(b, pts, radii, segs=8, v0=0.0, v1=1.0):
    pts = np.asarray(pts, float)
    n = len(pts)
    verts = []
    u_prev = None
    for i in range(n):
        d = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        u, w = frame(d)
        if u_prev is not None:  # parallel-transport to avoid twisting
            dn = d / np.linalg.norm(d)
            u = u_prev - dn * np.dot(u_prev, dn)
            u /= np.linalg.norm(u)
            w = np.cross(dn, u)
        u_prev = u
        for s in range(segs):
            a = 2 * math.pi * s / segs
            verts.append(pts[i] + radii[i] * (math.cos(a) * u + math.sin(a) * w))
    faces, uvs = [], []
    for i in range(n - 1):
        va, vb = v0 + (v1 - v0) * i / (n - 1), v0 + (v1 - v0) * (i + 1) / (n - 1)
        for s in range(segs):
            s2 = (s + 1) % segs
            faces.append((i * segs + s, i * segs + s2, (i + 1) * segs + s2, (i + 1) * segs + s))
            uvs.append([(s / segs, va), ((s + 1) / segs, va), ((s + 1) / segs, vb), (s / segs, vb)])
    b.add(verts, faces, uvs)


def ribbon(b, pts, widths, side):
    """Flat strip along pts; side = unit vector for the width direction."""
    verts, faces, uvs = [], [], []
    n = len(pts)
    for i in range(n):
        verts.append(pts[i] - side * widths[i] * 0.5)
        verts.append(pts[i] + side * widths[i] * 0.5)
    for i in range(n - 1):
        a = 2 * i
        faces.append((a, a + 1, a + 3, a + 2))
        uvs.append([(0, i / (n - 1)), (1, i / (n - 1)), (1, (i + 1) / (n - 1)), (0, (i + 1) / (n - 1))])
    b.add(verts, faces, uvs)


# ---------------------------------------------------------------- stem
H = 0.24
t = np.linspace(0, 1, 40)
stem_pts = np.stack([0.018 * np.sin(t * 2.0) + 0.004 * t ** 3, 0.006 * np.sin(t * 3.1), H * t], -1)
stem_r = 0.0021 - 0.0004 * t + 0.0006 * np.clip((t - 0.93) / 0.07, 0, 1) ** 2  # swells under head
axis = stem_pts[-1] - stem_pts[-3]
axis /= np.linalg.norm(axis)
REC_R = 0.0028
head = stem_pts[-1] + axis * REC_R * 0.6

stem = Builder()
tube(stem, stem_pts, stem_r, segs=12)
# receptacle: small squashed dome
lat, lon = 8, 16
ru, rw = frame(axis)
def rec_p(i, j):
    th, ph = math.pi * i / lat, 2 * math.pi * j / lon
    return head + REC_R * (math.sin(th) * (math.cos(ph) * ru + math.sin(ph) * rw) + 0.8 * math.cos(th) * axis)
for i in range(lat):
    for j in range(lon):
        stem.add([rec_p(i, j), rec_p(i, j + 1), rec_p(i + 1, j + 1), rec_p(i + 1, j)], [(0, 1, 2, 3)],
                 [[(0.5, 0.97)] * 4])
stem.build("Stem", material("Stem", tex_stem(), rough=0.45, sheen=0.2))

# involucral bracts: narrow, reflexed (curled back down the stem)
bracts = Builder()
for k in range(16):
    ph = 2 * math.pi * k / 16 + rng.uniform(-0.1, 0.1)
    radial = math.cos(ph) * ru + math.sin(ph) * rw
    L = rng.uniform(0.010, 0.014)
    pts = []
    for i in range(8):
        s = i / 7
        ang = math.pi * 0.55 + s * rng.uniform(0.5, 0.9)  # starts sideways, curls down/back
        p = head - axis * REC_R * 0.4 + radial * REC_R * 0.9
        pts.append(p + L * s * (math.cos(ang) * axis + math.sin(ang) * radial) * 0.9)
    side = np.cross(axis, radial)
    ribbon(bracts, np.array(pts), np.linspace(0.0014, 0.0003, 8), side)
bracts.build("Bracts", material("Bract", tex_bract(), rough=0.5))

# ---------------------------------------------------------------- seeds + pappus
seeds, hairs = Builder(), Builder()
N = 170
golden = math.pi * (3 - math.sqrt(5))
for i in range(N):
    z = 1 - 2 * (i + 0.5) / N
    if z < -0.82:  # leave room for the stem
        continue
    rr = math.sqrt(1 - z * z)
    # sphere in the head's own frame (axis = up)
    d = rr * math.cos(golden * i) * ru + rr * math.sin(golden * i) * rw + z * axis
    d = d + rng.normal(0, 0.035, 3)
    d /= np.linalg.norm(d)
    base = head + d * REC_R * 0.95
    ach_len = rng.uniform(0.0034, 0.0040)
    beak_len = rng.uniform(0.0085, 0.0100)
    ach_end = base + d * ach_len
    # beak bends very slightly
    bend = rng.normal(0, 0.03, 3)
    tip = ach_end + (d + bend) / np.linalg.norm(d + bend) * beak_len
    s_ = np.linspace(0, 1, 6)
    tube(seeds, base[None] + np.outer(s_, ach_end - base), 0.00045 * np.sin(math.pi * (0.15 + 0.7 * s_)) + 0.00008,
         segs=6, v0=0.0, v1=0.3)
    tube(seeds, [ach_end, (ach_end + tip) / 2, tip], [0.00011, 0.00009, 0.00008], segs=4, v0=0.3, v1=1.0)

    # pappus: ~60 very fine hairs spreading like an umbrella, tips curling up
    dd = (tip - ach_end) / np.linalg.norm(tip - ach_end)
    u, w = frame(dd)
    nh = 60
    for h in range(nh):
        a = 2 * math.pi * (h + rng.uniform(-0.3, 0.3)) / nh
        radial = math.cos(a) * u + math.sin(a) * w
        L = rng.uniform(0.0050, 0.0064)
        elev0 = math.radians(rng.uniform(8, 22))
        curl = math.radians(rng.uniform(15, 35))
        pts = [tip]
        p = tip.copy()
        seg = 4
        for k in range(1, seg + 1):
            e = elev0 + curl * (k / seg) ** 1.6
            step = (math.cos(e) * radial + math.sin(e) * dd)
            p = p + step * (L / seg)
            pts.append(p)
        pts = np.array(pts)
        tang = np.cross(radial, dd)
        tw = rng.uniform(0, math.pi)  # random twist so hairs read from every angle
        side = math.cos(tw) * tang + math.sin(tw) * dd
        side -= radial * np.dot(side, radial)
        side /= np.linalg.norm(side)
        ribbon(hairs, pts, np.linspace(0.000055, 0.00002, seg + 1), side)

seeds.build("Seeds", material("Seed", tex_seed(), rough=0.55))
hairs.build("Pappus", material("Pappus", color=(0.95, 0.95, 0.93), rough=0.45, sheen=0.6), smooth=False)

# ---------------------------------------------------------------- leaves (runcinate rosette)
def half_width(s, W, phase):
    """Backward-pointing lobes: steep rise to the tooth tip, long gentle return."""
    if s < 0.12:
        return W * (0.08 + 0.3 * s)
    if s < 0.80:
        nlobe = 4
        x = (s - 0.12) / 0.68 * nlobe + phase
        f = x % 1.0
        idx = min(int(x), nlobe - 1)
        A = W * (0.55 + 0.35 * idx / (nlobe - 1))
        sinus = W * 0.38
        if f < 0.18:
            return sinus + (A - sinus) * math.sin(f / 0.18 * math.pi / 2)
        return sinus + (A - sinus) * (1 - ((f - 0.18) / 0.82) ** 1.3)
    q = (s - 0.80) / 0.20
    if q < 0.06:
        return W * 0.38 + (W * 0.95 - W * 0.38) * q / 0.06
    return W * 0.95 * (1 - (q - 0.06) / 0.94) ** 1.1


leaves = Builder()
for k in range(7):
    ang = k * 2 * math.pi / 7 + rng.uniform(-0.25, 0.25)
    L = rng.uniform(0.13, 0.19)
    W = rng.uniform(0.020, 0.026)
    rise = rng.uniform(0.012, 0.035)
    pl, pr = rng.uniform(0, 0.2), rng.uniform(0, 0.2)
    nl, nw = 220, 9
    dirv = np.array([math.cos(ang), math.sin(ang), 0])
    sidev = np.array([-math.sin(ang), math.cos(ang), 0])
    verts, faces, uvs = [], [], []
    for i in range(nl):
        s = i / (nl - 1)
        wl, wr = half_width(s, W, pl), half_width(s, W, pr)
        lift = rise * math.sin(math.pi * min(s * 1.3, 1)) * (1 - 0.5 * s) + 0.004
        c = dirv * L * s + np.array([0, 0, lift])
        for j in range(nw):
            q = j / (nw - 1) * 2 - 1
            wq = wl if q < 0 else wr
            off = sidev * wq * q
            z = q * q * 0.0025 + abs(q) ** 2 * 0.0008 * math.sin(s * 40 + k)  # V-crease + ripple
            verts.append(c + off + np.array([0, 0, z]))
    for i in range(nl - 1):
        for j in range(nw - 1):
            a = i * nw + j
            faces.append((a, a + 1, a + nw + 1, a + nw))
            u0, u1 = j / (nw - 1), (j + 1) / (nw - 1)
            v0, v1 = i / (nl - 1), (i + 1) / (nl - 1)
            uvs.append([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    leaves.add(verts, faces, uvs)
leaves.build("Leaves", material("Leaf", tex_leaf(), rough=0.65, sheen=0.1))

# ---------------------------------------------------------------- export
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "dandelion.glb"), export_format="GLB",
                          export_draco_mesh_compression_enable=True)
if "--no-render" in sys.argv:
    raise SystemExit

# ---------------------------------------------------------------- preview scene (not exported)
sc = bpy.context.scene
world = bpy.data.worlds.new("W")
sc.world = world
world.use_nodes = True
wn = world.node_tree
sky = wn.nodes.new("ShaderNodeTexSky")
sky.sky_type = "NISHITA" if "NISHITA" in [e.identifier for e in sky.bl_rna.properties["sky_type"].enum_items] else sky.sky_type
sky.sun_elevation = math.radians(18)
sky.sun_rotation = math.radians(200)
wn.links.new(sky.outputs[0], wn.nodes["Background"].inputs[0])
wn.nodes["Background"].inputs[1].default_value = 0.35

sun = bpy.data.lights.new("Sun", "SUN")
sun.energy = 3.0
sun.angle = math.radians(1.5)
sun.color = (1.0, 0.93, 0.82)
so = bpy.data.objects.new("Sun", sun)
sc.collection.objects.link(so)
# low, behind-left: backlight makes the pappus glow
so.rotation_euler = mathutils.Vector((0.6, 0.9, -0.35)).to_track_quat("Z", "Y").to_euler()

# grassy ground with procedural noise
bpy.ops.mesh.primitive_plane_add(size=30)
gm = bpy.data.materials.new("Ground")
gm.use_nodes = True
gn = gm.node_tree
nz = gn.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 60
ramp = gn.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.05, 0.10, 0.03, 1)
ramp.color_ramp.elements[1].color = (0.22, 0.30, 0.08, 1)
gn.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
gn.links.new(ramp.outputs["Color"], gn.nodes["Principled BSDF"].inputs["Base Color"])
gn.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
bpy.context.object.data.materials.append(gm)

cam = bpy.data.cameras.new("Cam")
co = bpy.data.objects.new("Cam", cam)
sc.collection.objects.link(co)
sc.camera = co
cam.dof.use_dof = True


def aim(loc, target, lens, fstop):
    co.location = loc
    cam.lens = lens
    d = mathutils.Vector(target) - mathutils.Vector(loc)
    co.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.dof.focus_distance = d.length
    cam.dof.aperture_fstop = fstop


sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.use_denoising = True
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "AgX - Medium High Contrast"
sc.render.resolution_x, sc.render.resolution_y = 900, 1100

sc.cycles.samples = 128
aim((0.30, -0.36, 0.16), (0.004, 0, 0.13), 50, 5.6)
sc.render.filepath = os.path.join(OUT, "preview.png")
bpy.ops.render.render(write_still=True)

sc.render.resolution_x, sc.render.resolution_y = 1000, 1000
sc.cycles.samples = 256
hc = tuple(head)
aim((hc[0] + 0.09, hc[1] - 0.12, hc[2] - 0.01), (hc[0], hc[1], hc[2] - 0.006), 85, 16)
sc.render.filepath = os.path.join(OUT, "preview_closeup.png")
bpy.ops.render.render(write_still=True)
