"""Blender helpers: fast numpy mesh building, node-tree helpers, instancing, materials."""
import math

import bpy
import bmesh  # noqa: F401  (only importable after bpy)
import mathutils
import numpy as np

from world import fbm


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


# =====================================================================
# mesh building
# =====================================================================
class MB:
    """Accumulates numpy geometry (verts, faces of any arity, per-corner UVs,
    per-vertex colours, per-face material index) and builds a Blender mesh."""

    def __init__(self):
        self.V, self.F, self.UV, self.C, self.M = [], [], [], [], []
        self.n = 0

    def add(self, V, F, UV=None, C=None, mi=0):
        V = np.asarray(V, float).reshape(-1, 3)
        F = np.asarray(F, np.int64)
        if F.ndim == 1:
            F = F[None]
        if len(V) == 0 or len(F) == 0:
            return
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

    def extend(self, other, mi_offset=0):
        """Append another builder's geometry (its faces are relative to its own verts)."""
        base = self.n
        for V, F, UV, C, M in zip(other.V, other.F, other.UV, other.C, other.M):
            self.V.append(V)
            self.F.append(F + base)
            self.UV.append(UV)
            self.C.append(C)
            self.M.append(M + mi_offset)
        self.n += other.n

    def build(self, name, mats, smooth=True, link=True, collection=None):
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
            (collection or bpy.context.scene.collection).objects.link(ob)
        return ob


def grid_faces(nr, nc, wrap=False):
    cc = nc if wrap else nc - 1
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(cc), indexing="ij")
    j2 = (j + 1) % nc
    return np.stack([i * nc + j, i * nc + j2, (i + 1) * nc + j2, (i + 1) * nc + j], -1).reshape(-1, 4)


def perp_frame(t):
    t = t / np.linalg.norm(t)
    a = np.array([0, 0, 1.0]) if abs(t[2]) < 0.95 else np.array([1.0, 0, 0])
    u = np.cross(t, a)
    u /= np.linalg.norm(u)
    return u, np.cross(t, u)


def tube(pts, radii, segs=8, cols=None, uv_scale=1.0, rmod=None):
    """Tube along pts. UVs are in metres (u around, v along) so bark shaders tile correctly."""
    pts = np.asarray(pts, float)
    n = len(pts)
    radii = np.broadcast_to(np.asarray(radii, float), (n,))
    T = np.gradient(pts, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    u, _ = perp_frame(T[0])
    U, W = [], []
    for i in range(n):
        u = u - T[i] * np.dot(u, T[i])
        nu = np.linalg.norm(u)
        if nu < 1e-6:
            u, _ = perp_frame(T[i])
        else:
            u = u / nu
        U.append(u)
        W.append(np.cross(T[i], u))
    U, W = np.array(U), np.array(W)
    ang = 2 * np.pi * np.arange(segs) / segs
    R = np.repeat(radii[:, None], segs, 1)
    if rmod is not None:
        R = R * rmod(np.linspace(0, 1, n)[:, None], ang[None, :])
    V = pts[:, None] + R[:, :, None] * (np.cos(ang)[None, :, None] * U[:, None] + np.sin(ang)[None, :, None] * W[:, None])
    F = grid_faces(n, segs, wrap=True)
    # UVs (metres)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]) * uv_scale
    i, j = np.meshgrid(np.arange(n - 1), np.arange(segs), indexing="ij")
    circ = 2 * np.pi * np.maximum(radii, 0.01) * uv_scale
    u0 = (j / segs) * circ[i]
    u1 = ((j + 1) / segs) * circ[i]
    v0, v1 = L[i], L[i + 1]
    UV = np.stack([np.stack([u0, v0], -1), np.stack([u1, v0], -1), np.stack([u1, v1], -1), np.stack([u0, v1], -1)], -2)
    C = None if cols is None else np.repeat(np.asarray(cols, float).reshape(n, -1), segs, 0)
    return V.reshape(-1, 3), F, UV.reshape(-1, 4, 2), C


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


def noise3(P, scale, octaves=3, seed=0):
    """Cheap pseudo-3D fbm built from two 2D fbm projections."""
    x, y, z = P[:, 0] * scale, P[:, 1] * scale, P[:, 2] * scale
    return 0.5 * (fbm(x + 0.61 * z, y - 0.37 * z, octaves, seed) + fbm(y + 0.43 * z + 7.1, x - 0.71 * z, octaves, seed + 5))


# =====================================================================
# node helpers
# =====================================================================
class NB:
    """Tiny node-tree builder: n = nb.new("ShaderNodeMath", operation="MULTIPLY", inputs={0: x, 1: 2.0})."""

    def __init__(self, tree):
        self.t = tree
        self.nodes = tree.nodes
        self.links = tree.links

    def new(self, kind, inputs=None, **props):
        n = self.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        for k, v in (inputs or {}).items():
            self.set(n.inputs[k], v)
        return n

    def set(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.links.new(v, sock)
        elif isinstance(v, bpy.types.Node):
            self.links.new(v.outputs[0], sock)
        else:
            if isinstance(v, (tuple, list)) and len(v) == 3 and sock.type == "RGBA":
                v = (*v, 1.0)
            sock.default_value = v

    def link(self, a, b):
        self.links.new(a if isinstance(a, bpy.types.NodeSocket) else a.outputs[0], b)

    # math sugar -----------------------------------------------------
    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.new("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.set(n.inputs[0], a)
        if b is not None:
            self.set(n.inputs[1], b)
        if c is not None:
            self.set(n.inputs[2], c)
        return n.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        n = self.new("ShaderNodeVectorMath", operation=op)
        self.set(n.inputs[0], a)
        if b is not None:
            self.set(n.inputs[1], b)
        if scale is not None:
            self.set(n.inputs["Scale"], scale)
        return n.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs["Vector"]

    def mix(self, fac, a, b, blend="MIX", dtype="RGBA"):
        n = self.new("ShaderNodeMix", data_type=dtype, blend_type=blend)
        if dtype == "RGBA":
            self.set(n.inputs[0], fac)
            self.set(n.inputs[6], a)
            self.set(n.inputs[7], b)
            return n.outputs[2]
        if dtype == "FLOAT":
            self.set(n.inputs[0], fac)
            self.set(n.inputs[2], a)
            self.set(n.inputs[3], b)
            return n.outputs[0]
        self.set(n.inputs[0], fac)
        self.set(n.inputs[4], a)
        self.set(n.inputs[5], b)
        return n.outputs[1]

    def maprange(self, v, a, b, c=0.0, d=1.0, interp="SMOOTHSTEP", clamp=True):
        n = self.new("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        self.set(n.inputs[0], v)
        n.inputs[1].default_value = a
        n.inputs[2].default_value = b
        self.set(n.inputs[3], c)
        self.set(n.inputs[4], d)
        return n.outputs[0]

    def noise(self, vec, scale, detail=4.0, rough=0.5, dim="3D", distortion=0.0, out="Fac"):
        n = self.new("ShaderNodeTexNoise", noise_dimensions=dim)
        if vec is not None:
            self.set(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        return n.outputs[out]

    def voronoi(self, vec, scale, feature="F1", metric="EUCLIDEAN", out="Distance", rand=1.0):
        n = self.new("ShaderNodeTexVoronoi", feature=feature, distance=metric)
        if vec is not None:
            self.set(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Randomness"].default_value = rand
        return n.outputs[out]

    def sep(self, vec):
        n = self.new("ShaderNodeSeparateXYZ")
        self.set(n.inputs[0], vec)
        return n.outputs

    def comb(self, x, y, z):
        n = self.new("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.set(n.inputs[i], v)
        return n.outputs[0]

    def ramp(self, fac, stops, interp="LINEAR"):
        n = self.new("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) > len(stops):
            cr.elements.remove(cr.elements[-1])
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for el, (pos, col) in zip(cr.elements, stops):
            el.position = pos
            el.color = (*col, 1.0) if len(col) == 3 else col
        self.set(n.inputs[0], fac)
        return n.outputs[0]


def new_material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.bl_idname != "ShaderNodeOutputMaterial":
            nt.nodes.remove(n)
    return m, NB(nt), nt.nodes["Material Output"]


def principled(nb, **inputs):
    b = nb.new("ShaderNodeBsdfPrincipled")
    for k, v in inputs.items():
        nb.set(b.inputs[k.replace("_", " ")], v)
    return b


# =====================================================================
# instancing via geometry nodes
# =====================================================================
_INST_GROUPS = {}


def inst_group(src):
    if src.name in _INST_GROUPS:
        return _INST_GROUPS[src.name]
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
    sn = N.new("GeometryNodeInputNamedAttribute"); sn.data_type = "FLOAT_VECTOR"; sn.inputs["Name"].default_value = "scl"
    e2r = N.new("FunctionNodeEulerToRotation")
    L.new(gi.outputs[0], iop.inputs["Points"])
    L.new(oi.outputs["Geometry"], iop.inputs["Instance"])
    L.new(rn.outputs["Attribute"], e2r.inputs["Euler"])
    L.new(e2r.outputs["Rotation"], iop.inputs["Rotation"])
    L.new(sn.outputs["Attribute"], iop.inputs["Scale"])
    L.new(iop.outputs["Instances"], go.inputs[0])
    _INST_GROUPS[src.name] = ng
    return ng


def scatter(name, src, pos, rot=None, scale=None, tint=None, rng=None):
    """Instance `src` at pos (n,3). rot: (n,) yaw or (n,3) euler. scale: (n,) or (n,3)."""
    rng = rng or np.random.default_rng(0)
    pos = np.asarray(pos, float).reshape(-1, 3)
    n = len(pos)
    if n == 0:
        return None
    if rot is None:
        rot = rng.uniform(0, 2 * np.pi, n)
    rot = np.asarray(rot, float)
    if rot.ndim == 1:
        rot = np.column_stack([np.zeros(n), np.zeros(n), rot])
    if scale is None:
        scale = np.ones(n)
    scale = np.asarray(scale, float)
    if scale.ndim == 1:
        scale = np.repeat(scale[:, None], 3, 1)
    tint = rng.random(n) if tint is None else np.asarray(tint, float)
    me = bpy.data.meshes.new(name + "_pts")
    me.vertices.add(n)
    me.vertices.foreach_set("co", pos.astype(np.float32).ravel())
    for aname, data, kind, key in (("rot", rot, "FLOAT_VECTOR", "vector"), ("scl", scale, "FLOAT_VECTOR", "vector"),
                                   ("tint", tint, "FLOAT", "value")):
        a = me.attributes.new(aname, kind, "POINT")
        a.data.foreach_set(key, data.astype(np.float32).ravel())
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.modifiers.new("inst", "NODES").node_group = inst_group(src)
    return ob


def hide_library(objs, z=-10000.0):
    for o in objs:
        o.location = (0, 0, z)


def look_at(ob, target):
    d = mathutils.Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
