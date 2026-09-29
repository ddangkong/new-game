"""Procedural materials, sky and atmosphere for the hero renders (Cycles)."""
import math

import bpy

from blx import NB, new_material, principled

# ---------------------------------------------------------------------------
# shared bits
# ---------------------------------------------------------------------------


def _vc_tinted(nb, hue_var=0.03, val_var=0.25, sat_var=0.15):
    """Vertex colour 'Col' varied per instance by the scatter attribute 'tint'."""
    vc = nb.new("ShaderNodeVertexColor", layer_name="Col")
    at = nb.new("ShaderNodeAttribute", attribute_type="INSTANCER", attribute_name="tint")
    t = nb.math("SUBTRACT", at.outputs["Fac"], 0.5)
    hsv = nb.new("ShaderNodeHueSaturation")
    nb.set(hsv.inputs["Hue"], nb.math("MULTIPLY_ADD", t, hue_var * 2, 0.5))
    nb.set(hsv.inputs["Saturation"], nb.math("MULTIPLY_ADD", t, sat_var * 2, 1.0))
    nb.set(hsv.inputs["Value"], nb.math("MULTIPLY_ADD", t, val_var * 2, 1.0))
    nb.set(hsv.inputs["Color"], vc.outputs["Color"])
    return hsv.outputs[0], at.outputs["Fac"]


def _translucent_mix(nb, out, bsdf, color, amount, warm=(1.25, 1.15, 0.7)):
    warm_col = nb.mix(1.0, color, warm, blend="MULTIPLY")
    tr = nb.new("ShaderNodeBsdfTranslucent")
    nb.set(tr.inputs["Color"], warm_col)
    mix = nb.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = amount
    nb.link(bsdf.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs["Surface"])


# ---------------------------------------------------------------------------
# vegetation
# ---------------------------------------------------------------------------


def mat_leaf(name, trans=0.38, rough=0.45, spec=0.45, sheen=0.0, emit=0.0, hue_var=0.035, val_var=0.22):
    m, nb, out = new_material(name)
    col, _ = _vc_tinted(nb, hue_var, val_var)
    b = principled(nb, Base_Color=col, Roughness=rough, Specular_IOR_Level=spec)
    if sheen:
        b.inputs["Sheen Weight"].default_value = sheen
    if emit:
        nb.set(b.inputs["Emission Color"], col)
        b.inputs["Emission Strength"].default_value = emit
    _translucent_mix(nb, out, b, col, trans)
    return m


def mat_grass(name="grass"):
    return mat_leaf(name, trans=0.45, rough=0.38, spec=0.4, sheen=0.15, hue_var=0.03, val_var=0.2)


def mat_petal(name, trans=0.5, rough=0.35, spec=0.4, sheen=0.35, emit=0.0):
    return mat_leaf(name, trans=trans, rough=rough, spec=spec, sheen=sheen, emit=emit, hue_var=0.015, val_var=0.12)


def mat_pappus(name="pappus", emit=0.0):
    m, nb, out = new_material(name)
    b = principled(nb, Base_Color=(0.93, 0.93, 0.9), Roughness=0.4, Specular_IOR_Level=0.5)
    b.inputs["Sheen Weight"].default_value = 0.8
    if emit:
        b.inputs["Emission Color"].default_value = (1.0, 0.95, 0.85, 1)
        b.inputs["Emission Strength"].default_value = emit
    _translucent_mix(nb, out, b, (0.95, 0.95, 0.92), 0.45, warm=(1.0, 0.98, 0.9))
    return m


def mat_bark(name, style="oak"):
    m, nb, out = new_material(name)
    uv = nb.new("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    geo = nb.new("ShaderNodeNewGeometry")
    nz = nb.sep(geo.outputs["Normal"])[2]
    obj = nb.new("ShaderNodeTexCoord").outputs["Object"]
    if style in ("oak", "willow", "sacred"):
        su, sv = {"oak": (7.0, 1.3), "willow": (9.0, 0.9), "sacred": (3.0, 0.5)}[style]
        mp = nb.new("ShaderNodeMapping")
        nb.set(mp.inputs["Vector"], uv)
        mp.inputs["Scale"].default_value = (su, sv, 1)
        warp = nb.noise(mp.outputs[0], 3.0, 3, 0.5, out="Color")
        wv = nb.mix(0.18, mp.outputs[0], warp, dtype="VECTOR")
        edge = nb.voronoi(wv, 3.0, feature="DISTANCE_TO_EDGE")
        ridge = nb.maprange(edge, 0.02, 0.22)
        fine = nb.noise(uv, 18.0, 6, 0.6)
        if style == "sacred":
            dark, light = (0.16, 0.15, 0.15), (0.62, 0.6, 0.57)
        elif style == "willow":
            dark, light = (0.04, 0.035, 0.03), (0.24, 0.22, 0.19)
        else:
            dark, light = (0.03, 0.025, 0.02), (0.2, 0.18, 0.155)
        base = nb.mix(ridge, dark, light)
        base = nb.mix(nb.maprange(fine, 0.3, 0.7), nb.mix(1.0, base, (0.8, 0.8, 0.8), blend="MULTIPLY"), base)
        height = nb.math("ADD", ridge, nb.math("MULTIPLY", fine, 0.35))
    elif style == "birch":
        mp = nb.new("ShaderNodeMapping")
        nb.set(mp.inputs["Vector"], uv)
        mp.inputs["Scale"].default_value = (1.3, 12.0, 1)
        dash = nb.noise(mp.outputs[0], 1.0, 2, 0.5)
        lent = nb.maprange(dash, 0.62, 0.68)
        sv = nb.sep(uv)[1]
        basedark = nb.maprange(sv, 2.2, 0.4)  # dark fissured bark near the ground
        fiss = nb.voronoi(uv, 5.0, feature="DISTANCE_TO_EDGE")
        white = nb.mix(nb.noise(uv, 6.0, 4, 0.6), (0.72, 0.7, 0.65), (0.9, 0.88, 0.84))
        base = nb.mix(lent, white, (0.06, 0.05, 0.05))
        base = nb.mix(nb.math("MULTIPLY", basedark, nb.maprange(fiss, 0.0, 0.1, 1, 0.3)), base, (0.1, 0.09, 0.08))
        height = nb.math("SUBTRACT", 1.0, lent)
    else:  # pine: red-brown plates
        mp = nb.new("ShaderNodeMapping")
        nb.set(mp.inputs["Vector"], uv)
        mp.inputs["Scale"].default_value = (2.0, 0.8, 1)
        edge = nb.voronoi(mp.outputs[0], 2.5, feature="DISTANCE_TO_EDGE")
        plate = nb.maprange(edge, 0.0, 0.08)
        tone = nb.voronoi(mp.outputs[0], 2.5, out="Color")
        pc = nb.mix(0.35, (0.4, 0.22, 0.13), tone)
        base = nb.mix(plate, (0.05, 0.03, 0.025), pc)
        height = plate
    # moss on upward-facing bark
    moss_n = nb.noise(obj, 1.2, 5, 0.6)
    moss = nb.math("MULTIPLY", nb.maprange(nz, 0.15, 0.7), nb.maprange(moss_n, 0.45, 0.6))
    if style == "birch":
        moss = nb.math("MULTIPLY", moss, 0.4)
    mosscol = nb.mix(nb.noise(obj, 9.0, 3, 0.6), (0.07, 0.13, 0.03), (0.2, 0.28, 0.06))
    col = nb.mix(moss, base, mosscol)
    bump = nb.new("ShaderNodeBump", inputs={"Strength": 0.9, "Distance": 0.04, "Height": height})
    b = principled(nb, Base_Color=col, Roughness=0.85, Normal=bump.outputs[0])
    if style == "sacred":  # faint glowing lichen specks
        sp = nb.voronoi(obj, 14.0, out="Distance")
        glow = nb.maprange(sp, 0.06, 0.0)
        glow = nb.math("MULTIPLY", glow, nb.maprange(nb.noise(obj, 0.8, 2), 0.55, 0.62))
        b.inputs["Emission Color"].default_value = (0.45, 0.95, 1.0, 1)
        nb.set(b.inputs["Emission Strength"], nb.math("MULTIPLY", glow, 6.0))
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


# ---------------------------------------------------------------------------
# rocks, ruins, terrain
# ---------------------------------------------------------------------------


def _rock_color(nb, pos, nz, scale=1.0, moss_amount=1.0, warm=0.5):
    P = nb.vmath("SCALE", pos, scale=scale)
    n_big = nb.noise(P, 0.3, 6, 0.55)
    n_mid = nb.noise(P, 1.6, 7, 0.62)
    n_fine = nb.noise(P, 9.0, 4, 0.55)
    tone = nb.ramp(n_big, [(0.3, (0.12, 0.115, 0.11)), (0.5, (0.22, 0.21, 0.195)), (0.7, (0.33, 0.315, 0.29))])
    tone = nb.mix(nb.math("MULTIPLY", nb.maprange(nb.noise(P, 0.7, 3), 0.35, 0.65), warm), tone,
                  nb.mix(1.0, tone, (1.25, 1.08, 0.86), blend="MULTIPLY"))
    shadow = nb.maprange(n_mid, 0.3, 0.7, 0.7, 1.08, interp="LINEAR")
    tone = nb.mix(1.0, tone, nb.comb(shadow, shadow, shadow), blend="MULTIPLY")
    # thin hairline cracks, only here and there
    cr = nb.voronoi(nb.vmath("MULTIPLY", P, (1.0, 1.0, 1.6)), 2.2, feature="DISTANCE_TO_EDGE")
    crmask = nb.maprange(nb.noise(P, 0.9, 2), 0.5, 0.62)
    crack = nb.math("MULTIPLY", nb.maprange(cr, 0.012, 0.0), crmask)
    cf = nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", crack, 0.55))
    tone = nb.mix(1.0, tone, nb.comb(cf, cf, cf), blend="MULTIPLY")
    # lichen (sparse, pale and ochre)
    lv = nb.voronoi(P, 6.0, out="Distance")
    lich = nb.math("MULTIPLY", nb.maprange(lv, 0.14, 0.06), nb.maprange(nb.noise(P, 0.7, 2), 0.58, 0.66))
    lcol = nb.mix(nb.noise(P, 3.0, 2), (0.55, 0.53, 0.44), (0.5, 0.42, 0.2))
    tone = nb.mix(lich, tone, lcol)
    # moss on up-facing surfaces
    mn = nb.noise(pos, 1.5, 6, 0.65)
    moss = nb.math("MULTIPLY", nb.maprange(nz, 0.3, 0.75), nb.maprange(mn, 0.4, 0.56))
    moss = nb.math("MULTIPLY", moss, moss_amount, clamp=True)
    mosscol = nb.mix(nb.noise(pos, 12.0, 3), (0.05, 0.1, 0.02), (0.17, 0.25, 0.045))
    col = nb.mix(moss, tone, mosscol)
    height = nb.math("ADD", nb.math("MULTIPLY", n_mid, 0.7), nb.math("MULTIPLY", n_fine, 0.25))
    height = nb.math("SUBTRACT", height, nb.math("MULTIPLY", crack, 0.3))
    height = nb.math("ADD", height, nb.math("MULTIPLY", moss, 0.3))
    rough = nb.mix(moss, 0.78, 0.95, dtype="FLOAT")
    return col, height, rough, moss


def mat_rock(name="rock", moss_amount=1.0, warm=0.5):
    m, nb, out = new_material(name)
    pos = nb.new("ShaderNodeTexCoord").outputs["Object"]
    geo = nb.new("ShaderNodeNewGeometry")
    nz = nb.sep(geo.outputs["Normal"])[2]
    col, height, rough, _ = _rock_color(nb, pos, nz, 1.0, moss_amount, warm)
    bump = nb.new("ShaderNodeBump", inputs={"Strength": 0.7, "Distance": 0.06, "Height": height})
    b = principled(nb, Base_Color=col, Roughness=rough, Normal=bump.outputs[0])
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_stone(name="ruin_stone"):
    """Carved sandstone-granite with moss on top faces and dirt at the base."""
    m, nb, out = new_material(name)
    pos = nb.new("ShaderNodeTexCoord").outputs["Object"]
    wpos = nb.new("ShaderNodeNewGeometry")
    nz = nb.sep(wpos.outputs["Normal"])[2]
    col, height, rough, _ = _rock_color(nb, pos, nz, 1.6, 1.3, warm=0.9)
    bump = nb.new("ShaderNodeBump", inputs={"Strength": 0.55, "Distance": 0.04, "Height": height})
    b = principled(nb, Base_Color=col, Roughness=rough, Normal=bump.outputs[0])
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_rune(name="rune", color=(0.45, 0.95, 1.0), strength=8.0):
    m, nb, out = new_material(name)
    em = nb.new("ShaderNodeEmission", inputs={"Color": color, "Strength": strength})
    nb.link(em.outputs[0], out.inputs["Surface"])
    return m


def mat_glow(name, color, strength):
    return mat_rune(name, color, strength)


def mat_terrain(name="terrain"):
    """World-space terrain: grassy soil, stratified rock on steep ground, moss on ledges,
    wet mud at the waterline. Includes vector displacement (terraces + rock detail) for
    Cycles adaptive subdivision. Needs the float vertex attribute 'wl' (water level)."""
    m, nb, out = new_material(name)
    m.displacement_method = "BOTH"
    geo = nb.new("ShaderNodeNewGeometry")
    P = geo.outputs["Position"]
    px, py, pz = nb.sep(P)
    nz = nb.sep(geo.outputs["Normal"])[2]
    tnz = nb.sep(geo.outputs["True Normal"])[2]

    # ----- displacement (uses the undisplaced base normal) -----
    steep = nb.maprange(tnz, 0.86, 0.5)
    micro = nb.math("ADD", nb.math("MULTIPLY", nb.math("SUBTRACT", nb.noise(P, 0.18, 4, 0.5), 0.5), 0.45),
                    nb.math("MULTIPLY", nb.math("SUBTRACT", nb.noise(P, 1.1, 3, 0.5), 0.5), 0.1))
    warpn = nb.math("MULTIPLY", nb.math("SUBTRACT", nb.noise(P, 0.035, 3, 0.5), 0.5), 6.0)
    hn = nb.math("ADD", pz, warpn)
    layer = 3.2
    f = nb.math("DIVIDE", hn, layer)
    fr = nb.math("FRACT", f)
    terr = nb.math("MULTIPLY", nb.math("SUBTRACT", nb.maprange(fr, 0.5, 0.92), fr), layer)
    rockn = nb.math("MULTIPLY", nb.math("SUBTRACT", nb.noise(P, 0.22, 7, 0.6), 0.5), 2.6)
    cr = nb.voronoi(nb.vmath("MULTIPLY", P, (0.35, 0.35, 0.9)), 1.0, feature="DISTANCE_TO_EDGE")
    crack = nb.math("MULTIPLY", nb.maprange(cr, 0.0, 0.06, -0.6, 0.0), 1.0)
    steep_dz = nb.math("ADD", nb.math("ADD", nb.math("MULTIPLY", terr, 0.9), rockn), crack)
    dz = nb.mix(steep, micro, steep_dz, dtype="FLOAT")
    vd = nb.new("ShaderNodeVectorDisplacement", space="OBJECT")
    nb.set(vd.inputs["Vector"], nb.comb(0.0, 0.0, dz))
    vd.inputs["Scale"].default_value = 1.0
    nb.link(vd.outputs[0], out.inputs["Displacement"])

    # ----- surface -----
    rock = nb.maprange(nb.math("ADD", nz, nb.math("MULTIPLY", nb.math("SUBTRACT", nb.noise(P, 0.3, 3), 0.5), 0.25)), 0.8, 0.58)
    patch = nb.noise(P, 0.012, 4, 0.55)
    ground = nb.ramp(patch, [(0.3, (0.07, 0.15, 0.03)), (0.5, (0.16, 0.22, 0.05)), (0.7, (0.3, 0.29, 0.1))])
    ground = nb.mix(nb.maprange(nb.noise(P, 0.4, 3), 0.3, 0.7), nb.mix(1.0, ground, (0.75, 0.72, 0.6), blend="MULTIPLY"), ground)
    # stratified rock
    strata = nb.math("SINE", nb.math("ADD", nb.math("MULTIPLY", pz, 1.9), nb.math("MULTIPLY", nb.noise(P, 0.05, 3), 6.0)))
    rtone = nb.ramp(nb.maprange(strata, -1, 1, 0, 1, interp="LINEAR"),
                    [(0.0, (0.22, 0.2, 0.18)), (0.45, (0.38, 0.34, 0.29)), (0.8, (0.5, 0.45, 0.37)), (1.0, (0.3, 0.27, 0.24))])
    rcol, rheight, rrough, rmoss = _rock_color(nb, P, nz, 0.35, 0.9, 0.6)
    rock_col = nb.mix(0.55, rcol, rtone)
    col = nb.mix(rock, ground, rock_col)
    # wet band at the waterline
    wl = nb.new("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="wl").outputs["Fac"]
    above = nb.math("SUBTRACT", pz, wl)
    wet = nb.maprange(above, 0.9, 0.05)
    col = nb.mix(wet, col, nb.mix(1.0, col, (0.35, 0.33, 0.3), blend="MULTIPLY"))
    rough = nb.mix(rock, 0.95, rrough, dtype="FLOAT")
    rough = nb.mix(wet, rough, 0.25, dtype="FLOAT")
    bump = nb.new("ShaderNodeBump", inputs={"Strength": 0.35, "Distance": 0.05,
                                            "Height": nb.math("ADD", rheight, nb.noise(P, 2.5, 4))})
    b = principled(nb, Base_Color=col, Roughness=rough, Normal=bump.outputs[0])
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


# ---------------------------------------------------------------------------
# water
# ---------------------------------------------------------------------------


def mat_water(name="water", ripple=0.03, calm_patches=True, flow=False):
    m, nb, out = new_material(name)
    geo = nb.new("ShaderNodeNewGeometry")
    P = geo.outputs["Position"]
    if flow:
        Pw = nb.vmath("MULTIPLY", P, (0.6, 0.6, 1.0))
    else:
        Pw = P
    w1 = nb.noise(Pw, 0.9, 8, 0.55)
    w2 = nb.noise(nb.vmath("MULTIPLY", Pw, (1.0, 3.0, 1.0)), 2.5, 4, 0.5)
    h = nb.math("ADD", w1, nb.math("MULTIPLY", w2, 0.5))
    strength = ripple
    if calm_patches:
        patch = nb.maprange(nb.noise(P, 0.012, 3), 0.4, 0.65, 0.15, 1.0)
        strength = nb.math("MULTIPLY", patch, ripple)
    bump = nb.new("ShaderNodeBump", inputs={"Distance": 0.1, "Height": h})
    nb.set(bump.inputs["Strength"], strength)
    b = principled(nb, Base_Color=(0.012, 0.03, 0.026), Roughness=0.015, IOR=1.33, Specular_IOR_Level=0.55,
                   Normal=bump.outputs[0])
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_waterfall(name="waterfall"):
    m, nb, out = new_material(name)
    uv = nb.new("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    su, sv = nb.sep(uv)[0], nb.sep(uv)[1]
    streak = nb.noise(nb.comb(nb.math("MULTIPLY", su, 7.0), nb.math("MULTIPLY", sv, 0.35), 0.0), 3.0, 8, 0.65)
    streak2 = nb.noise(nb.comb(nb.math("MULTIPLY", su, 26.0), nb.math("MULTIPLY", sv, 1.2), 3.0), 2.0, 4, 0.5)
    a = nb.maprange(nb.math("ADD", streak, nb.math("MULTIPLY", streak2, 0.5)), 0.55, 0.95, 0.15, 1.0)
    col = nb.mix(nb.maprange(streak2, 0.3, 0.8), (0.72, 0.8, 0.82), (1.0, 1.0, 1.0))
    b = principled(nb, Base_Color=col, Roughness=0.25, Alpha=a, Transmission_Weight=0.2)
    b.inputs["Emission Color"].default_value = (1.0, 0.95, 0.85, 1)
    nb.set(b.inputs["Emission Strength"], nb.math("MULTIPLY", a, 0.25))
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_foam(name="foam"):
    m, nb, out = new_material(name)
    b = principled(nb, Base_Color=(0.95, 0.97, 1.0), Roughness=0.5, Transmission_Weight=0.3)
    b.inputs["Sheen Weight"].default_value = 0.5
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


# ---------------------------------------------------------------------------
# world: physical sky + procedural cloud layer + camera-only sun disc
# ---------------------------------------------------------------------------


def make_world(sun_el, sun_az, strength=0.3, clouds=0.55, cloud_scale=1.0, seed=0.0):
    """sun_az: radians from +X toward +Y. Blender sky rotation: 0 -> +Y, 90deg -> +X."""
    w = bpy.data.worlds.new("sky")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        if n.bl_idname != "ShaderNodeOutputWorld":
            nt.nodes.remove(n)
    nb = NB(nt)
    out = nt.nodes["World Output"]
    sky = nb.new("ShaderNodeTexSky", sky_type="MULTIPLE_SCATTERING")
    sky.sun_elevation = sun_el
    sky.sun_rotation = math.pi / 2 - sun_az
    sky.sun_disc = False
    sky.air_density = 1.2
    sky.aerosol_density = 1.6
    sun_dir = (math.cos(sun_el) * math.cos(sun_az), math.cos(sun_el) * math.sin(sun_az), math.sin(sun_el))

    D = nb.new("ShaderNodeTexCoord").outputs["Generated"]
    D = nb.vmath("NORMALIZE", D)
    dx, dy, dz = nb.sep(D)
    # cloud plane projection (clouds at "altitude")
    inv = nb.math("DIVIDE", 1.0, nb.math("ADD", nb.math("MAXIMUM", dz, 0.0), 0.06))
    cu = nb.math("MULTIPLY", dx, inv)
    cv = nb.math("MULTIPLY", dy, inv)
    cp = nb.comb(nb.math("MULTIPLY", cu, cloud_scale), nb.math("MULTIPLY", cv, cloud_scale), seed)
    warp = nb.noise(cp, 0.35, 3, 0.5, out="Color")
    cpw = nb.mix(0.35, cp, warp, dtype="VECTOR")
    big = nb.noise(cpw, 0.45, 6, 0.62)
    streaky = nb.noise(nb.vmath("MULTIPLY", cpw, (0.6, 2.4, 1.0)), 0.9, 8, 0.6)
    dens = nb.math("ADD", nb.math("MULTIPLY", big, 0.7), nb.math("MULTIPLY", streaky, 0.45))
    cov = nb.maprange(dens, 0.95 - clouds * 0.5, 1.12 - clouds * 0.3)
    horizon = nb.maprange(dz, 0.0, 0.18)
    cov = nb.math("MULTIPLY", cov, horizon)
    # cloud lighting: bright rims toward the sun, warm under-lighting, violet shadow
    cosang = nb.vmath("DOT_PRODUCT", D, sun_dir)
    fwd = nb.math("POWER", nb.maprange(cosang, 0.3, 1.0, 0.0, 1.0, interp="LINEAR"), 3.0)
    thick = nb.maprange(dens, 0.9, 1.3)
    shadow = (0.42, 0.36, 0.48)
    lit = (1.0, 0.72, 0.45)
    ccol = nb.mix(nb.math("SUBTRACT", 1.0, thick), shadow, lit)
    ccol = nb.mix(fwd, ccol, (1.6, 1.25, 0.85))
    ccol = nb.mix(1.0, ccol, (strength * 6.0,) * 3, blend="MULTIPLY")
    skycol = nb.mix(1.0, sky.outputs[0], (1.0, 1.0, 1.0), blend="MULTIPLY")
    col = nb.mix(nb.math("MULTIPLY", cov, 0.9), skycol, ccol)
    bg = nb.new("ShaderNodeBackground")
    nb.set(bg.inputs["Color"], col)
    bg.inputs["Strength"].default_value = strength
    # sun disc + glow for camera rays only (the sun lamp does the lighting)
    lp = nb.new("ShaderNodeLightPath")
    disc = nb.maprange(cosang, math.cos(math.radians(0.7)), math.cos(math.radians(0.45)), 0.0, 1.0)
    halo = nb.math("POWER", nb.maprange(cosang, 0.985, 1.0, 0.0, 1.0, interp="LINEAR"), 6.0)
    sunv = nb.math("ADD", nb.math("MULTIPLY", disc, 900.0), nb.math("MULTIPLY", halo, 6.0))
    sunv = nb.math("MULTIPLY", sunv, lp.outputs["Is Camera Ray"])
    sunv = nb.math("MULTIPLY", sunv, nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", cov, 0.85)))
    em = nb.new("ShaderNodeEmission", inputs={"Color": (1.0, 0.8, 0.55), "Strength": sunv})
    add = nb.new("ShaderNodeAddShader")
    nb.link(bg.outputs[0], add.inputs[0])
    nb.link(em.outputs[0], add.inputs[1])
    nb.link(add.outputs[0], out.inputs["Surface"])
    return w


def add_sun(sun_el, sun_az, energy=4.5, color=(1.0, 0.8, 0.58), angle_deg=0.6):
    import mathutils
    to_sun = mathutils.Vector((math.cos(sun_el) * math.cos(sun_az), math.cos(sun_el) * math.sin(sun_az), math.sin(sun_el)))
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = energy
    sun.color = color
    sun.angle = math.radians(angle_deg)
    so = bpy.data.objects.new("sun", sun)
    bpy.context.scene.collection.objects.link(so)
    so.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()
    return so


def fog_box(name, center, size, density, color=(1, 1, 1), aniso=0.6, height_falloff=None, noise_scale=None,
            noise_amount=0.6, base_z=None):
    """Volume box. height_falloff (m): density *= exp(-(z-base_z)/falloff)."""
    bpy.ops.mesh.primitive_cube_add(location=center)
    ob = bpy.context.object
    ob.name = name
    ob.scale = [s / 2 for s in size]
    m, nb, out = new_material(name)
    geo = nb.new("ShaderNodeNewGeometry")
    P = geo.outputs["Position"]
    dens = density
    if height_falloff:
        z0 = center[2] - size[2] / 2 if base_z is None else base_z
        dz = nb.math("SUBTRACT", nb.sep(P)[2], z0)
        dens = nb.math("MULTIPLY", nb.math("EXPONENT", nb.math("DIVIDE", nb.math("MULTIPLY", dz, -1.0), height_falloff)), density)
    if noise_scale:
        nn = nb.noise(P, noise_scale, 4, 0.55)
        nfac = nb.maprange(nn, 0.35, 0.75, 1.0 - noise_amount, 1.0 + noise_amount, interp="LINEAR")
        dens = nb.math("MULTIPLY", dens, nfac)
    pv = nb.new("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = (*color, 1)
    pv.inputs["Anisotropy"].default_value = aniso
    nb.set(pv.inputs["Density"], dens)
    nb.link(pv.outputs[0], out.inputs["Volume"])
    ob.data.materials.append(m)
    ob.visible_shadow = False
    return ob


# ---------------------------------------------------------------------------
# compositor: bloom, gentle dispersion, grade, vignette
# ---------------------------------------------------------------------------


def setup_compositor(bloom=0.35, bloom_threshold=1.0, vignette=0.28, dispersion=0.012,
                     lift=(1.0, 1.0, 1.02), gamma=(1.0, 1.0, 1.0), gain=(1.02, 1.0, 0.97)):
    sc = bpy.context.scene
    ng = bpy.data.node_groups.new("post", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    nb = NB(ng)
    rl = nb.new("CompositorNodeRLayers")
    gl = nb.new("CompositorNodeGlare")
    gl.inputs["Type"].default_value = "Bloom"
    gl.inputs["Quality"].default_value = "High"
    gl.inputs["Threshold"].default_value = bloom_threshold
    gl.inputs["Strength"].default_value = bloom
    gl.inputs["Size"].default_value = 0.8
    nb.link(rl.outputs["Image"], gl.inputs["Image"])
    ld = nb.new("CompositorNodeLensdist")
    ld.inputs["Dispersion"].default_value = dispersion
    ld.inputs["Distortion"].default_value = 0.0
    nb.link(gl.outputs["Image"], ld.inputs["Image"])
    cb = nb.new("CompositorNodeColorBalance")
    cb.inputs["Type"].default_value = "Lift/Gamma/Gain"
    cb.inputs["Factor"].default_value = 1.0
    socks = [s for s in cb.inputs if s.name in ("Lift", "Gamma", "Gain") and s.type == "RGBA"]
    for s, v in zip(socks, (lift, gamma, gain)):
        s.default_value = (*v, 1)
    nb.link(ld.outputs[0], cb.inputs["Image"])
    em = nb.new("CompositorNodeEllipseMask")
    em.inputs["Size"].default_value = (1.25, 1.1)
    bl = nb.new("CompositorNodeBlur")
    bl.inputs["Size"].default_value = (300, 300)
    nb.link(em.outputs["Mask"], bl.inputs["Image"])
    mix = nb.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    mix.inputs[0].default_value = 1.0
    nb.link(cb.outputs[0], mix.inputs[6])
    vg = nb.new("ShaderNodeMapRange", clamp=True)
    nb.link(bl.outputs[0], vg.inputs[0])
    vg.inputs[3].default_value = 1.0 - vignette
    vg.inputs[4].default_value = 1.0
    comb = nb.new("CompositorNodeCombineColor") if "CompositorNodeCombineColor" in dir(bpy.types) else None
    if comb is not None:
        for i in range(3):
            nb.link(vg.outputs[0], comb.inputs[i])
        nb.link(comb.outputs[0], mix.inputs[7])
    else:
        nb.link(vg.outputs[0], mix.inputs[7])
    go = nb.new("NodeGroupOutput")
    nb.link(mix.outputs[2], go.inputs[0])
    sc.compositing_node_group = ng
    return ng
