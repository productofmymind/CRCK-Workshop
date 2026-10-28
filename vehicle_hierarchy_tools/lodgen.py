"""Automatic LOD generation for the parts of a vehicle, into Sollumz's LOD slots (no loose copies).
measurements behind each choice.

Levels are Sollumz's: L0 very high (the `_hi` file), L1 high, L2 medium, L3 low, L4 very low. Generation starts from L0 or from L1 and fills the levels
below it. A slot that already has a mesh is left alone unless asked, objects are never created, and the source meshes are never changed.

Reduction (`reduce_levels`) uses CRCK Workshop' own simplifier (simplify.py): the mesh is welded by position so the pieces a game mesh is split into stay
joined; UV seams, material borders and open edges are kept as lines; and every level of a part comes from one run. Vertices keep their original
positions, UVs, colours and bone weights. Blender's Decimate is available as an alternative.
Shading: the original corner normals are transferred onto the new mesh (nearest point on the source surface, interpolated; the new face normal wins where the
two disagree strongly, which keeps hard edges), so the LODs shade like the source.
Parts that must keep their look (wheels, glass, lights, mirrors) keep more triangles by a factor.
"""
import math
import re

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from . import simplify
from . import sollumz_bridge as sz

# slot name in Object.sz_lods, label used in the mesh name
LEVELS = (("very_high", "very high"), ("high", "high"), ("medium", "medium"), ("low", "low"), ("very_low", "very low"))
SLOT_INDEX = {s: i for i, (s, _) in enumerate(LEVELS)}
LABEL = dict(LEVELS)
MIN_TRIANGLES = 4
WELD = 1e-5                     # positions closer than this (metres) are one point while reducing
# Triangles kept relative to the start mesh, from 11 vanilla vehicles (pooled over the parts that have both levels)
VANILLA = {"very_high": {"high": 0.30, "medium": 0.16, "low": 0.08, "very_low": 0.03},
           "high": {"medium": 0.50, "low": 0.27, "very_low": 0.09}}

CRITICAL = (("wheel", re.compile(r"wheel|tyre|tire")), ("glass", re.compile(r"window|windscreen|glass")),
            ("light", re.compile(r"light|indicator|siren|neon|emissive|lamp")), ("mirror", re.compile(r"mirror")))


def part_class(model, bones):
    """'wheel', 'glass', 'light', 'mirror' or None, from the names of the model and of the bone it is bound to."""
    from . import analysis
    names = [analysis.name_stem(model.name).lower()] + [b.lower() for b in analysis.bone_links(model, bones)]
    for cls, rx in CRITICAL:
        if any(rx.search(n) for n in names):
            return cls
    return None


def triangles(mesh):
    return sum(len(p.vertices) - 2 for p in mesh.polygons)


def slot_counts(models):
    """{slot: number of models that have a mesh in it}."""
    return {slot: sum(1 for o in models if getattr(o.sz_lods, slot).mesh is not None) for slot, _ in LEVELS}


# --- extraction and rebuilding ---------------------------------------------------

def corner_normals(mesh, dtype=np.float32):
    """Flat array of per-corner normals. Mesh.corner_normals is Blender 4.1+; 4.0 computes split normals into the loops."""
    out = np.empty(len(mesh.loops) * 3, dtype)
    if bpy.app.version >= (4, 1, 0):  # 4.0 has the attribute too, but empty
        mesh.corner_normals.foreach_get("vector", out)
    else:
        mesh.calc_normals_split()
        mesh.loops.foreach_get("normal", out)
    return out


def set_corner_normals(mesh, normals):
    """Custom per-corner normals. Blender 4.0 uses them only with Auto Smooth on (4.1 removed the option)."""
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    mesh.normals_split_custom_set(normals)


def extract(mesh):
    """Numpy view of a mesh: vertex positions, loops, per-corner normals, every UV layer and colour attribute per corner, material per triangle."""
    mesh.calc_loop_triangles()
    t = len(mesh.loop_triangles)
    loops = np.empty(t * 3, np.int32)
    mesh.loop_triangles.foreach_get("loops", loops)
    mat = np.empty(t, np.int32)
    mesh.loop_triangles.foreach_get("material_index", mat)
    nl = len(mesh.loops)
    vidx = np.empty(nl, np.int32)
    mesh.loops.foreach_get("vertex_index", vidx)
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    nrm = corner_normals(mesh)
    uvs = []
    for layer in mesh.uv_layers:
        a = np.empty(nl * 2, np.float32)
        layer.uv.foreach_get("vector", a)
        uvs.append(a.reshape(-1, 2))
    cols = []
    for attr in mesh.color_attributes:
        a = np.empty(len(attr.data) * 4, np.float32)
        attr.data.foreach_get("color", a)
        a = a.reshape(-1, 4)
        cols.append(a if attr.domain == "CORNER" else a[vidx])
    return {"co": co.reshape(-1, 3), "vidx": vidx, "loops": loops.reshape(-1, 3), "mat": mat, "nrm": nrm.reshape(-1, 3), "uvs": uvs, "cols": cols}


def skin_of(mesh):
    """(group names, dense (vertices, groups) deform weights or None when no vertex has a weight). The names live on the mesh in current Blender
    versions and are read through a temporary object."""
    tmp = bpy.data.objects.new("vht_lod_tmp", mesh)
    try:
        names = [g.name for g in tmp.vertex_groups]
    finally:
        bpy.data.objects.remove(tmp)
    vi, gi, wi = [], [], []
    for v in mesh.vertices:
        for g in v.groups:
            vi.append(v.index); gi.append(g.group); wi.append(g.weight)
    if not vi:
        return names, None
    width = max(len(names), max(gi) + 1)
    w = np.zeros((len(mesh.vertices), width), np.float32)
    w[vi, gi] = wi
    return names, w


def _set_weights(mesh, names, weights):
    """Give `mesh` the vertex groups `names` and the dense deform weights (in 1/255 steps, which is what a file can hold anyway)."""
    tmp = bpy.data.objects.new("vht_lod_tmp", mesh)
    try:
        for g in range(weights.shape[1]):
            tmp.vertex_groups.new(name=names[g] if g < len(names) else f"group_{g}")
        w = np.round(weights * 255.0) / 255.0
        for g in range(w.shape[1]):
            col = w[:, g]
            for val in np.unique(col[col > 0]):
                tmp.vertex_groups[g].add(np.flatnonzero(col == val).tolist(), float(val), "REPLACE")
    finally:
        bpy.data.objects.remove(tmp)


COLOUR_SCALE = 0.05   # a full colour change at a seam costs like a jump of 5% of the part's size
NORMAL_SCALE = 0.05   # a normal turned by 60 degrees (distance 1) at a hard edge: 5% of the part's size
WEIGHT_SCALE = 0.25   # a full bone-weight change: 25% of the part's size


def _prepare(src, skin):
    """Everything the simplifier needs: welded positions, triangles as position ids and as wedge ids (one wedge per distinct set of corner
    attributes at a position), every wedge's attributes with metres per unit for the seam cost, and the locks."""
    d = extract(src)
    loops = d["loops"]
    if not len(loops):
        return None
    _, first_v, pid_of_v = np.unique(np.round(d["co"] / WELD).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    cpid = pid_of_v.ravel()[d["vidx"]]
    pos = d["co"][first_v].astype(np.float64)
    names, dense = skin
    size = float(np.linalg.norm(pos.max(0) - pos.min(0))) or 1.0
    blocks, scale, layout = [], [], {}

    def add(name, block, metres):
        layout[name] = (sum(b.shape[1] for b in blocks), block.shape[1])
        blocks.append(block.astype(np.float64))
        scale.extend([metres] * block.shape[1])

    ea = np.concatenate([loops[:, 0], loops[:, 1], loops[:, 2]])
    eb = np.concatenate([loops[:, 1], loops[:, 2], loops[:, 0]])
    xyz = np.linalg.norm(d["co"][d["vidx"][ea]] - d["co"][d["vidx"][eb]], axis=1)
    for i, uv in enumerate(d["uvs"]):  # metres per UV unit: the median over the triangle edges
        duv = np.linalg.norm(uv[ea] - uv[eb], axis=1)
        ok = duv > 1e-9
        add(("uv", i), uv, float(np.median(xyz[ok] / duv[ok])) if ok.any() else size)
    for i, col in enumerate(d["cols"]):
        add(("col", i), col, COLOUR_SCALE * size)
    if dense is not None:
        add(("w",), dense[d["vidx"]], WEIGHT_SCALE * size)
    add(("n",), d["nrm"], NORMAL_SCALE * size)
    corner = np.concatenate(blocks, axis=1)
    no, _ = layout[("n",)]
    key = np.concatenate([cpid[:, None].astype(np.float64), np.round(corner[:, :no] * 1e5), np.round(corner[:, no:] * 1e3)], axis=1)
    _, wfirst, wid = np.unique(key, axis=0, return_index=True, return_inverse=True)
    tris = cpid[loops]
    lock = np.zeros(len(pos), bool)  # no pinned points: the simplifier keeps every material's last triangle instead (pins distort around them)
    return {"d": d, "pos": pos, "tris": tris, "wtri": wid.ravel()[loops], "cpid": cpid, "attrs": corner[wfirst], "scale": np.array(scale),
            "layout": layout, "lock": lock, "skin": (names, dense)}


def _build(src, ctx, snap, rows_all, name, keep_shading=True):
    """A new mesh from a simplifier snapshot: one vertex per position; UVs, corner colours and (keep_shading) normals from each corner's wedge;
    bone weights per vertex."""
    wtri, ptri, tmat, P = snap
    lay = ctx["layout"]
    used, inv = np.unique(ptri.ravel(), return_inverse=True)
    inv = inv.ravel()
    rows = rows_all[wtri.ravel()]                 # attributes of every new corner
    vert_corner = np.empty(len(used), np.int64)   # a corner of every new vertex (point-domain attributes, weights)
    vert_corner[inv] = np.arange(len(inv))
    layers = [l.name for l in src.uv_layers]
    cols = [(a.name, a.data_type, a.domain) for a in src.color_attributes]
    mesh = src.copy()
    mesh.clear_geometry()
    mesh.from_pydata(P[used].tolist(), [], inv.reshape(-1, 3).tolist())
    mesh.polygons.foreach_set("material_index", np.asarray(tmat, np.int32))
    for i, nm in enumerate(layers):
        o, w = lay[("uv", i)]
        mesh.uv_layers.new(name=nm).uv.foreach_set("vector", rows[:, o:o + w].ravel().astype(np.float32))
    for i, (nm, dtype, domain) in enumerate(cols):
        o, w = lay[("col", i)]
        data = rows[:, o:o + w] if domain == "CORNER" else rows[vert_corner, o:o + w]
        mesh.color_attributes.new(name=nm, type=dtype, domain=domain).data.foreach_set("color", data.ravel().astype(np.float32))
    mesh.update()
    mesh.name = name
    if keep_shading:
        o, _ = lay[("n",)]
        set_corner_normals(mesh, _pick_normals(ctx, ptri.ravel(), rows[:, o:o + 3], mesh).tolist())
    names, dense = ctx["skin"]
    if dense is not None:
        o, w = lay[("w",)]
        _set_weights(mesh, names, rows[vert_corner, o:o + w])
    return mesh


def _pick_normals(ctx, pid, own, mesh):
    """Per new corner: its wedge's normal, or another original normal at the same position (one per side at a hard edge) that is closer to the
    new triangle's own normal. A corner that crossed a hard edge while reducing so still gets the normal of its own side."""
    d = ctx["d"]
    cpid, nrm = ctx["cpid"], d["nrm"]
    pairs = np.unique(np.concatenate([cpid[:, None].astype(np.float64), np.round(nrm * 1e3) / 1e3], axis=1), axis=0)
    ppid = pairs[:, 0].astype(np.int64)
    cand = pairs[:, 1:]
    first = np.searchsorted(ppid, np.arange(len(ctx["pos"])))
    count = np.bincount(ppid, minlength=len(ctx["pos"]))
    fn = np.empty(len(mesh.polygons) * 3, np.float64)
    mesh.polygons.foreach_get("normal", fn)
    face = np.repeat(fn.reshape(-1, 3), 3, axis=0)
    best = own / np.maximum(np.linalg.norm(own, axis=1, keepdims=True), 1e-12)
    best_dot = (best * face).sum(1)
    for k in range(int(count.max()) if len(count) else 0):
        c = cand[np.minimum(first[pid] + k, len(cand) - 1)]
        dot = (c * face).sum(1)
        take = (count[pid] > k) & (dot > best_dot + 1e-6)
        best[take], best_dot[take] = c[take], dot[take]
    return best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)


def reduce_levels(src, targets, skin=None, keep_shading=True):
    """{triangle target: new mesh} for every target, from one run of the simplifier on `src`."""
    ctx = _prepare(src, skin or skin_of(src))
    if ctx is None:
        return {}
    s = simplify.Simplifier(ctx["pos"], ctx["tris"], ctx["wtri"], ctx["d"]["mat"], ctx["lock"], ctx["attrs"], ctx["scale"])
    snaps = s.run(targets)
    rows = s.attributes()
    return {t: _build(src, ctx, snap, rows, src.name, keep_shading) for t, snap in snaps.items() if len(snap[0])}


def reduce_blender(src, ratio):
    """Reduce with Blender's Decimate (collapse), keeping vertex groups, UVs and colours (the modifier does)."""
    faces = len(src.polygons)
    ratio = max(min(ratio, 1.0), min(1.0, MIN_TRIANGLES / max(faces, 1)))
    tmp = bpy.data.objects.new("vht_lod_tmp", src)
    bpy.context.scene.collection.objects.link(tmp)
    try:
        mod = tmp.modifiers.new("Decimate", "DECIMATE")
        mod.decimate_type, mod.ratio, mod.use_collapse_triangulate = "COLLAPSE", ratio, True
        depsgraph = bpy.context.evaluated_depsgraph_get()
        out = bpy.data.meshes.new_from_object(tmp.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
    finally:
        bpy.data.objects.remove(tmp)
    return out


def transfer_shading(new, original, strong_disagreement=0.5):
    """Give `new` the corner normals of `original`: the nearest point of the original surface, with its corner normals interpolated. Where that normal
    and the new face's own normal disagree strongly (dot below `strong_disagreement`) the new face normal is kept, which preserves hard edges."""
    if not len(new.polygons) or not len(original.polygons):
        return
    original.calc_loop_triangles()
    ot = len(original.loop_triangles)
    ol = np.empty(ot * 3, np.int32)
    original.loop_triangles.foreach_get("loops", ol)
    ol = ol.reshape(-1, 3)
    ov = np.empty(ot * 3, np.int32)
    original.loop_triangles.foreach_get("vertices", ov)
    ov = ov.reshape(-1, 3)
    oco = np.empty(len(original.vertices) * 3, np.float64)
    original.vertices.foreach_get("co", oco)
    oco = oco.reshape(-1, 3)
    onrm = corner_normals(original, np.float64)
    onrm = onrm.reshape(-1, 3)
    bvh = BVHTree.FromPolygons([tuple(v) for v in oco], [tuple(t) for t in ov])
    new.calc_loop_triangles()
    nl = len(new.loops)
    nvidx = np.empty(nl, np.int32)
    new.loops.foreach_get("vertex_index", nvidx)
    nco = np.empty(len(new.vertices) * 3, np.float64)
    new.vertices.foreach_get("co", nco)
    nco = nco.reshape(-1, 3)
    fn = np.empty(len(new.polygons) * 3, np.float64)
    new.polygons.foreach_get("normal", fn)
    fn = fn.reshape(-1, 3)
    starts = np.empty(len(new.polygons), np.int32)
    new.polygons.foreach_get("loop_start", starts)
    sizes = np.diff(np.append(starts, nl))
    poly_of_loop = np.repeat(np.arange(len(starts)), sizes)
    # one nearest-point query per vertex (it depends on the position only)
    near_loc = np.empty((len(nco), 3))
    near_tri = np.empty(len(nco), np.int32)
    for i, p in enumerate(nco):
        loc, _, idx, _ = bvh.find_nearest(tuple(p))
        near_loc[i], near_tri[i] = loc, idx
    a, b, c = oco[ov[near_tri, 0]], oco[ov[near_tri, 1]], oco[ov[near_tri, 2]]
    v0, v1, v2 = b - a, c - a, near_loc - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-30, 1.0, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1 - v - w
    n_at = u[:, None] * onrm[ol[near_tri, 0]] + v[:, None] * onrm[ol[near_tri, 1]] + w[:, None] * onrm[ol[near_tri, 2]]
    n_at /= np.maximum(np.linalg.norm(n_at, axis=1, keepdims=True), 1e-20)
    cand = n_at[nvidx]
    face = fn[poly_of_loop]
    ok = (cand * face).sum(1) >= strong_disagreement
    normals = np.where(ok[:, None], cand, face)
    set_corner_normals(new, normals.tolist())


def reduce(src, ratio, method="AUTO", keep_shading=True, shading_from=None):
    """Reduce `src` to `ratio` of its triangles. Returns (mesh, which method produced it)."""
    target = max(int(triangles(src) * ratio), MIN_TRIANGLES)
    out, used = None, "blender"
    if method != "BLENDER" and (shading_from is None or shading_from is src):
        out, used = reduce_levels(src, [target], keep_shading=keep_shading).get(target), "crckworkshop"
        if out is not None:
            return out, used
    if method != "BLENDER":  # chained from another level: the shading comes from the start mesh
        out, used = reduce_levels(src, [target], keep_shading=False).get(target), "crckworkshop"
    if out is None:
        out, used = reduce_blender(src, ratio), "blender"
    if keep_shading:
        transfer_shading(out, shading_from if shading_from is not None else src)
    return out, used


# --- generation ---------------------------------------------------------------------

def generate(models, start="high", ratios=None, make_very_high=False, overwrite=False, method="AUTO", chain=False, protect=1.5,
             keep_shading=True, bones=None):
    """Fill the LOD slots below `start` ('very_high' = L0, `_hi`; or 'high' = L1) for each model. `ratios` maps slot -> share of the *start* mesh's
    triangles (0 or missing skips that slot; defaults are the vanilla ratios). With `chain` every level is reduced from the next higher level that has a
    mesh instead of from the start mesh. `make_very_high` (start 'high' only) copies the high mesh into an empty L0. `protect` multiplies the share kept
    by parts that must keep their look (1 turns it off). Returns {'done': [(model name, [slots], methods)], 'skipped': [(model name, why)], 'triangles': {slot: n}, 'methods': {method: meshes made}}."""
    ratios = ratios or VANILLA[start]
    s = SLOT_INDEX[start]
    below = [slot for slot, _ in LEVELS[s + 1:]]
    report = {"done": [], "skipped": [], "triangles": {slot: 0 for slot, _ in LEVELS}, "methods": {}}
    for o in models:
        lods = o.sz_lods
        src = getattr(lods, start).mesh
        if src is None:
            report["skipped"].append((o.name, f"no {LABEL[start]} (L{s}) mesh to start from"))
            continue
        cls = part_class(o, bones) if bones is not None and protect > 1.0 else None
        factor = protect if cls else 1.0
        base_tris = triangles(src)
        filled, used_methods = [], set()
        source = src
        if start == "high" and make_very_high and getattr(lods, "very_high").mesh is None:
            mesh = src.copy()
            mesh.name = f"{o.name}.{LABEL['very_high']}"
            lods.very_high.mesh = mesh
            filled.append("very_high")
        batch = {}  # every level reduced from the start mesh comes from one simplifier run
        if not chain and method != "BLENDER":
            wants = {slot: max(MIN_TRIANGLES, int(base_tris * min(1.0, ratios.get(slot, 0.0) * factor))) for slot in below
                     if ratios.get(slot, 0.0) > 0 and (overwrite or getattr(lods, slot).mesh is None)}
            wants = {k: n for k, n in wants.items() if n < base_tris}
            if wants:
                made = reduce_levels(src, list(wants.values()), keep_shading=keep_shading)
                batch = {slot: made.get(n) for slot, n in wants.items()}
                if len(set(wants.values())) < len(wants):  # two levels asked the same count: give each its own mesh
                    seen = set()
                    for slot, m in batch.items():
                        if m is not None and m.as_pointer() in seen:
                            batch[slot] = m.copy()
                        elif m is not None:
                            seen.add(m.as_pointer())
        for slot in below:
            ratio = ratios.get(slot, 0.0)
            lod = getattr(lods, slot)
            if lod.mesh is not None and not overwrite:
                if chain:
                    source = lod.mesh
                continue
            if ratio <= 0:
                continue
            want = max(MIN_TRIANGLES, int(base_tris * min(1.0, ratio * factor)))
            have = triangles(source)
            if slot in batch and source is src:
                made = batch.pop(slot)
                mesh, how = (made, "crckworkshop") if made is not None else (source.copy(), "copy")
            elif want >= have:
                mesh, how = source.copy(), "copy"
            else:
                mesh, how = reduce(source, want / have, method, keep_shading, shading_from=src)
            mesh.name = f"{o.name}.{LABEL[slot]}"
            old = lod.mesh
            lod.mesh = mesh
            if old is not None and old.users == 0:
                bpy.data.meshes.remove(old)
            filled.append(slot)
            used_methods.add(how)
            report["methods"][how] = report["methods"].get(how, 0) + 1
            report["triangles"][slot] += triangles(mesh)
            if chain:
                source = mesh
        for left in batch.values():  # not used (cannot happen, but never leave orphan meshes)
            if left is not None:
                bpy.data.meshes.remove(left)
        report["triangles"][start] += base_tris
        if filled:
            report["done"].append((o.name, filled, "+".join(sorted(used_methods)) or "copy"))
    return report
