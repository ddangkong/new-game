"""Quick look-dev renders of individual assets: python3 asset_preview.py oak birch ..."""
import math
import os
import sys
import time

import bpy
import mathutils
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blx  # noqa: E402
import flora  # noqa: E402
import rocks  # noqa: E402
import shade  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders", "lookdev")
os.makedirs(OUT, exist_ok=True)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
RES = 480 if "--hi" not in sys.argv else 900
SAMPLES = 48 if "--hi" not in sys.argv else 128


def mats(kind):
    if kind in ("oak", "shrub"):
        return [shade.mat_bark("bark_oak", "oak"), shade.mat_leaf("leaf_oak"), shade.mat_petal("blossom")]
    if kind == "birch":
        return [shade.mat_bark("bark_birch", "birch"), shade.mat_leaf("leaf_birch", trans=0.45), shade.mat_petal("b")]
    if kind == "willow":
        return [shade.mat_bark("bark_willow", "willow"), shade.mat_leaf("leaf_willow", trans=0.45, spec=0.55), shade.mat_petal("b")]
    if kind == "spruce":
        return [shade.mat_bark("bark_pine", "pine"), shade.mat_leaf("needles", trans=0.2, rough=0.5), shade.mat_petal("b")]
    if kind == "sacred":
        return [shade.mat_bark("bark_sacred", "sacred"), shade.mat_leaf("leaf_sacred"), shade.mat_petal("blossom", emit=0.6)]
    if kind.startswith("grass"):
        return [shade.mat_grass()]
    if kind in ("poppy", "daisy", "cornflower", "buttercup", "lupine", "dandelion", "clock"):
        return [shade.mat_petal("petal"), shade.mat_leaf("green"), shade.mat_petal("centre", trans=0.1)]
    if kind in ("fern", "reeds"):
        return [shade.mat_leaf("x"), shade.mat_leaf("fernleaf", trans=0.5), shade.mat_bark("cattail", "pine")]
    if kind == "seed":
        return [shade.mat_pappus("pappus", emit=0.2), shade.mat_leaf("achene", trans=0.1)]
    if kind in ("boulder", "column"):
        return [shade.mat_rock("rock")]
    if kind in ("monolith", "arch"):
        return [shade.mat_stone(), shade.mat_rune(), shade.mat_leaf("ivy")]
    if kind in ("lily", "lotus"):
        return [shade.mat_leaf("pad", trans=0.2, rough=0.3, spec=0.6), shade.mat_petal("lotus")]
    raise KeyError(kind)


def build(kind, seed=1):
    t0 = time.time()
    if kind in flora.SPECIES:
        mb = flora.make_tree(kind, seed)
    elif kind.startswith("grass_"):
        mb = flora.make_grass(kind[6:], seed)
    elif kind in ("poppy", "daisy", "cornflower", "buttercup", "lupine", "dandelion", "clock"):
        mb = flora.make_flower(kind, seed)
    elif kind == "fern":
        mb = flora.make_fern(seed)
    elif kind == "reeds":
        mb = flora.make_reeds(seed)
    elif kind == "seed":
        mb = flora.make_player_seed()
    elif kind == "boulder":
        mb = rocks.make_boulder(seed)
    elif kind == "monolith":
        mb = rocks.make_monolith(seed)
    elif kind == "arch":
        mb = rocks.make_arch()
    elif kind == "column":
        mb = rocks.make_column(seed)
    elif kind in ("lily", "lotus"):
        mb = rocks.make_lily(seed, lotus=kind == "lotus")
    ob = mb.build(kind, mats(kind))
    print(f"{kind}: {len(ob.data.polygons)} faces, {time.time() - t0:.1f}s", flush=True)
    return ob


for kind in ARGS:
    sc = blx.reset_scene()
    ob = build(kind)
    bb = np.array([ob.matrix_world @ mathutils.Vector(c) for c in ob.bound_box])
    lo, hi = bb.min(0), bb.max(0)
    ctr = (lo + hi) / 2
    size = float(np.max(hi - lo))
    el, az = math.radians(14), math.radians(-35)
    shade.make_world(el, az, strength=0.35, clouds=0.4)
    shade.add_sun(el, az, energy=4.0)
    bpy.ops.mesh.primitive_plane_add(size=size * 8, location=(ctr[0], ctr[1], lo[2]))
    gm, nb, out = blx.new_material("ground")
    b = blx.principled(nb, Base_Color=(0.12, 0.13, 0.08), Roughness=0.9)
    nb.link(b.outputs[0], out.inputs["Surface"])
    bpy.context.object.data.materials.append(gm)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.lens = 50
    cam.data.clip_start = 0.001
    d = size * 1.9
    cam.location = ctr + np.array([-d * 0.75, -d * 0.62, d * 0.12])
    blx.look_at(cam, ctr)
    sc.render.engine = "CYCLES"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.render.resolution_x = sc.render.resolution_y = RES
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.cycles.transparent_max_bounces = 16
    sc.render.filepath = os.path.join(OUT, f"{kind}.png")
    bpy.ops.render.render(write_still=True)
