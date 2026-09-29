"""Procedural dandelion seed head (with pappus) -> dandelion.glb + preview.png.
Run: python3 build_dandelion.py
"""
import math, os
import numpy as np
import bpy

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

bpy.ops.wm.read_factory_settings(use_empty=True)


# ---------- textures ----------
def make_image(name, px):
    h, w, _ = px.shape
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels = px.astype(np.float32).ravel()
    img.pack()
    return img


def tex_stem():
    h, w = 256, 64
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    streak = 0.08 * np.sin(x * 40 + np.sin(y * 7) * 2)
    g = 0.42 + streak + 0.1 * y
    rgb = np.stack([g * 0.45, g, g * 0.3], -1)
    rgb[..., 0] += (1 - y) * 0.12  # reddish base
    return np.dstack([rgb, np.ones((h, w))])


def tex_leaf():
    n = 256
    v, u = np.mgrid[0:n, 0:n] / n  # u across, v along
    mid = np.exp(-((u - 0.5) / 0.02) ** 2)
    veins = np.exp(-((np.abs(u - 0.5) * 2.2 - (v % 0.12) / 0.12 * 0.9 + 0.1) / 0.03) ** 2) * (np.abs(u - 0.5) > 0.02)
    base = 0.33 + 0.05 * np.sin(u * 30) * np.sin(v * 13)
    r = base * 0.4 + mid * 0.35 + veins * 0.08
    g = base + mid * 0.35 + veins * 0.12
    b = base * 0.2 + mid * 0.25
    return np.dstack([r, g, b, np.ones((n, n))])


def tex_pappus():
    n = 512
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n - 0.5
    r = np.hypot(x, y) * 2
    a = np.arctan2(y, x)
    k = 48
    fil = np.exp(-(((a * k / (2 * math.pi)) % 1 - 0.5) / 0.08) ** 2)
    fil = fil * (r < 0.98) * (r > 0.03)
    # fine barbs toward tips
    barbs = 0.35 * np.clip(np.sin(a * k * 6) * 0.5 + 0.5, 0, 1) * np.clip((r - 0.5) * 2, 0, 1)
    alpha = np.clip(fil + barbs * (r < 0.95), 0, 1) * np.clip(1.1 - r, 0, 1) * 1.6
    alpha = np.clip(alpha + (r < 0.06), 0, 1)
    c = 0.93 + 0.05 * r
    return np.dstack([c, c, c * 0.97, alpha])


def tex_seed():
    h, w = 128, 32
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    ridges = 0.1 * np.sin(x * 50)
    # v<0.3 achene (brown, ridged), above beak (pale)
    ach = y < 0.3
    r = np.where(ach, 0.42 + ridges, 0.85)
    g = np.where(ach, 0.30 + ridges, 0.83)
    b = np.where(ach, 0.18, 0.75)
    return np.dstack([r, g, b, np.ones((h, w))])


def material(name, px, rough=0.7, alpha=False, sss=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = make_image(name + "_tex", px)
    nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    if alpha:
        nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
        m.blend_method = "BLEND"
    m.use_backface_culling = False
    return m


# ---------- geometry helpers ----------
class Builder:
    def __init__(self):
        self.v, self.f, self.uv = [], [], []

    def add(self, verts, faces, uvs):
        o = len(self.v)
        self.v += [tuple(p) for p in verts]
        for fc, fu in zip(faces, uvs):
            self.f.append(tuple(i + o for i in fc))
            self.uv.append(fu)

    def build(self, name, mat, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        uvl = me.uv_layers.new(name="UVMap")
        k = 0
        for poly, fu in zip(me.polygons, self.uv):
            for li, uv in zip(poly.loop_indices, fu):
                uvl.data[li].uv = uv
        me.materials.append(mat)
        for p in me.polygons:
            p.use_smooth = smooth
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def frame(d):
    d = d / np.linalg.norm(d)
    a = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def tube(b, pts, radii, segs=8, v0=0.0, v1=1.0):
    pts = np.asarray(pts, float)
    n = len(pts)
    verts = []
    for i in range(n):
        d = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        u, w = frame(d)
        for s in range(segs):
            a = 2 * math.pi * s / segs
            verts.append(pts[i] + radii[i] * (math.cos(a) * u + math.sin(a) * w))
    faces, uvs = [], []
    for i in range(n - 1):
        for s in range(segs):
            s2 = (s + 1) % segs
            faces.append((i * segs + s, i * segs + s2, (i + 1) * segs + s2, (i + 1) * segs + s))
            va, vb = v0 + (v1 - v0) * i / (n - 1), v0 + (v1 - v0) * (i + 1) / (n - 1)
            ua, ub = s / segs, (s + 1) / segs
            uvs.append([(ua, va), (ub, va), (ub, vb), (ua, vb)])
    b.add(verts, faces, uvs)


# ---------- stem ----------
H = 0.32
t = np.linspace(0, 1, 24)
stem_pts = np.stack([0.025 * np.sin(t * 2.2) , 0.012 * t ** 2, H * t], -1)
head = stem_pts[-1] + np.array([0, 0, 0.004])
stem = Builder()
tube(stem, stem_pts, 0.0028 - 0.0008 * t, segs=10)
# receptacle (small sphere)
lat, lon = 8, 12
R = 0.0055
for i in range(lat):
    for j in range(lon):
        def p(ii, jj):
            th, ph = math.pi * ii / lat, 2 * math.pi * jj / lon
            return head + R * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        stem.add([p(i, j), p(i, j + 1), p(i + 1, j + 1), p(i + 1, j)], [(0, 1, 2, 3)],
                 [[(0.5, 0.95)] * 4])
stem_ob = stem.build("Stem", material("Stem", tex_stem(), 0.6))

# ---------- seeds + pappus ----------
seeds, papp = Builder(), Builder()
N = 220
g = math.pi * (3 - math.sqrt(5))
for i in range(N):
    z = 1 - 2 * (i + 0.5) / N
    if z < -0.55:
        continue
    rr = math.sqrt(1 - z * z)
    d = np.array([rr * math.cos(g * i), rr * math.sin(g * i), z])
    d += rng.normal(0, 0.05, 3); d /= np.linalg.norm(d)
    base = head + d * R
    ach_end = base + d * 0.006
    tip = base + d * (0.032 + rng.uniform(-0.002, 0.002))
    tube(seeds, [base, ach_end], [0.0008, 0.0006], segs=5, v0=0.0, v1=0.3)
    tube(seeds, [ach_end, tip], [0.00022, 0.00018], segs=4, v0=0.3, v1=1.0)
    # pappus: shallow cone (umbrella) with radial filament alpha texture
    u, w = frame(d)
    pr, depth, segs = 0.014 + rng.uniform(-0.001, 0.001), 0.005, 16
    rim = [tip + d * depth + pr * (math.cos(2 * math.pi * s / segs) * u + math.sin(2 * math.pi * s / segs) * w)
           for s in range(segs)]
    verts = [tip] + rim
    faces, uvs = [], []
    for s in range(segs):
        s2 = (s + 1) % segs
        faces.append((0, 1 + s, 1 + s2))
        a1, a2 = 2 * math.pi * s / segs, 2 * math.pi * (s + 1) / segs
        uvs.append([(0.5, 0.5), (0.5 + 0.5 * math.cos(a1), 0.5 + 0.5 * math.sin(a1)),
                    (0.5 + 0.5 * math.cos(a2), 0.5 + 0.5 * math.sin(a2))])
    papp.add(verts, faces, uvs)
seeds.build("Seeds", material("Seed", tex_seed(), 0.5))
papp.build("Pappus", material("Pappus", tex_pappus(), 0.9, alpha=True), smooth=False)

# ---------- leaves (rosette, toothed) ----------
leaves = Builder()
for k in range(6):
    ang = k * 2 * math.pi / 6 + rng.uniform(-0.2, 0.2)
    L = rng.uniform(0.13, 0.18)
    nl, nw = 40, 5
    dirv = np.array([math.cos(ang), math.sin(ang), 0])
    side = np.array([-math.sin(ang), math.cos(ang), 0])
    verts, faces, uvs = [], [], []
    for i in range(nl):
        s = i / (nl - 1)
        width = 0.022 * math.sin(math.pi * s ** 0.8) * (0.55 + 0.45 * ((s * 7) % 1))  # teeth
        lift = 0.05 * math.sin(math.pi * s * 0.9) * (1 - s * 0.6)
        c = dirv * L * s + np.array([0, 0, lift])
        for j in range(nw):
            q = j / (nw - 1) - 0.5
            verts.append(c + side * width * q * 2 + np.array([0, 0, -abs(q) * 0.004]))
    for i in range(nl - 1):
        for j in range(nw - 1):
            a = i * nw + j
            faces.append((a, a + 1, a + nw + 1, a + nw))
            uvs.append([(j / (nw - 1), i / (nl - 1)), ((j + 1) / (nw - 1), i / (nl - 1)),
                        ((j + 1) / (nw - 1), (i + 1) / (nl - 1)), (j / (nw - 1), (i + 1) / (nl - 1))])
    leaves.add(verts, faces, uvs)
leaves.build("Leaves", material("Leaf", tex_leaf(), 0.55))

# ---------- export ----------
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "dandelion.glb"), export_format="GLB")

# ---------- preview render ----------
sc = bpy.context.scene
world = bpy.data.worlds.new("W"); sc.world = world; world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.55, 0.72, 0.9, 1); bg.inputs[1].default_value = 0.9
sun = bpy.data.lights.new("Sun", "SUN"); sun.energy = 3.5; sun.angle = 0.1
so = bpy.data.objects.new("Sun", sun); so.rotation_euler = (0.9, 0.2, 0.8)
sc.collection.objects.link(so)
# ground
bpy.ops.mesh.primitive_plane_add(size=2)
gm = bpy.data.materials.new("Ground"); gm.use_nodes = True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.18, 0.28, 0.1, 1)
bpy.context.object.data.materials.append(gm)
cam = bpy.data.cameras.new("Cam"); cam.lens = 60
co = bpy.data.objects.new("Cam", cam); sc.collection.objects.link(co); sc.camera = co
co.location = (0.42, -0.5, 0.3)
tgt = np.array([0.0, 0, 0.17]); dvec = tgt - np.array(co.location)
import mathutils
co.rotation_euler = mathutils.Vector(dvec).to_track_quat("-Z", "Y").to_euler()
sc.render.engine = "CYCLES"; sc.cycles.samples = 96; sc.cycles.device = "CPU"
sc.render.resolution_x = sc.render.resolution_y = 900
sc.render.filepath = os.path.join(OUT, "preview.png")
bpy.ops.render.render(write_still=True)
# close-up of the seed head
cam.lens = 110
co.location = (0.12, -0.16, 0.36)
dvec = head - np.array(co.location)
co.rotation_euler = mathutils.Vector(dvec).to_track_quat("-Z", "Y").to_euler()
sc.render.filepath = os.path.join(OUT, "preview_closeup.png")
bpy.ops.render.render(write_still=True)
