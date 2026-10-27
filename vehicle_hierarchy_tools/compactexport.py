"""CRCK Workshop Export (compact export): Sollumz's own exporter, with one change, a vertex merge that tolerates float noise.

Why: Sollumz writes one vertex per Blender face corner and merges two only when every attribute (position, normal, tangent, UVs, colours, weights) is equal
after rounding to 6 decimals. Blender computes the normal of a corner from its surrounding faces, and corners of one vertex end up with normals that differ
in the 4th to 6th decimal; they stay separate vertices. On the mammoth, 14% of the vertices (161,211 written, 138,260 distinct once normals are compared to 3
decimals) were such copies. A file of the same vehicle written by another tool has 139,419. Vertex buffers are most of a yft, so the file grows about as much.

What this does: runs `sollumz.export_assets` with the user's own Sollumz export settings (format, version, selection, mesh domain), and for the duration of the
call swaps Sollumz's `dedupe_and_get_indices` (ydr/ydrexport.py) for `dedupe_and_get_indices` below, which rounds normals and tangents to 3 decimals (0.06
degrees) when deciding what is the same vertex. The surviving vertex keeps its own un-rounded values. Everything else (materials, bones, bounds, physics,
textures, file writing) is Sollumz's. The swap is undone in a finally block.

Third change: triangles and vertices are reordered for the GPU's vertex cache (Tipsify, then vertices in order of first use). The cache order puts
the vertices of neighbouring triangles next to each other, so the buffers hold runs of similar values and compress better, and the game transforms
fewer vertices. A mesh another tool already ordered keeps its order when that compresses smaller. The set of triangles is unchanged.

Second change, the larger one: before merging, normals, tangents, positions and UVs are snapped to a fine power-of-two grid (STEPS below). A float from
Blender carries noise in its lowest bits, which makes the vertex buffer compress badly; snapped values compress far better, and snapping first also lets
more corners merge. On the mammoth (3.73 MB with Sollumz alone, 3.43 MB with the merge alone) the result is 2.40 MB, with the same triangles
and a largest error of 0.0005 per normal component (0.05 degrees), 0.06 mm in position and 0.00012 in UV. Colours and bone weights are not touched.
"""
import contextlib
import os
import sys
import time
import zlib

import bpy
import numpy as np

from . import sollumz_bridge as sz

DEFAULT_DECIMALS = 6
DECIMALS = {}                             # after snapping, equal values are exactly equal; the default 6 decimals only guards against float noise
STEPS = {"Normal": 1 / 1024, "Tangent": 1 / 1024, "Position": 1 / 8192,
         "TexCoord0": 1 / 4096, "TexCoord1": 1 / 4096, "TexCoord2": 1 / 4096, "TexCoord3": 1 / 4096}  # powers of two, so the low mantissa bits become zeros
STATS = {"stock": 0, "compact": 0}
_ORIGINAL = [None]  # Sollumz's own dedupe_and_get_indices while patched() is active
_ROUND_POSITIONS = [True]  # False while patched() could not confirm that the cloth guard below still matches Sollumz


def _unique(vertex_arr, decimals):
    names = vertex_arr.dtype.names
    cols = []
    for name in names:
        a = vertex_arr[name].astype(np.float64).reshape(len(vertex_arr), -1)
        cols.append(np.round(a, decimals.get(name, DEFAULT_DECIMALS)) + 0.0)  # + 0.0 turns -0.0 into 0.0
    flat = np.concatenate(cols, axis=1)
    return np.unique(flat, axis=0, return_index=True, return_inverse=True)


def snap(vertex_arr):
    """A copy of the vertex array with the attributes in STEPS rounded to their grid."""
    out = vertex_arr.copy()
    for name, step in STEPS.items():
        if name == "Position" and not _ROUND_POSITIONS[0]:
            continue
        if name in out.dtype.names:
            a = out[name]
            out[name] = (np.round(a.astype(np.float64) / step) * step).astype(a.dtype)
    return out


CACHE_SIZE = 16  # vertex cache size Tipsify plans for


def tipsify(indices, n, k=CACHE_SIZE):
    """Triangle order for the vertex cache: Tipsify (Sander, Nehab and Barczak, "Fast Triangle Reordering for Vertex Locality and Reduced
    Overdraw", 2007). Fans out from a vertex, then continues from the recent vertex that is still in the cache and has the fewest triangles left.
    Returns the triangles' new order (indices into the triangle list)."""
    tri = np.asarray(indices, np.int64).reshape(-1, 3)
    nt = len(tri)
    flat = tri.ravel()
    order = np.argsort(flat, kind="stable")
    adj = (order // 3).tolist()                     # triangles of each vertex, grouped by vertex
    start = np.searchsorted(flat[order], np.arange(n + 1)).tolist()
    live = np.bincount(flat, minlength=n).tolist()  # triangles not yet emitted, per vertex
    stamp = [0] * n
    tris = tri.tolist()
    emitted = bytearray(nt)
    out = []
    dead = []
    s = k + 1
    cursor = 0
    f = 0 if n else -1
    while f >= 0:
        cand = []
        for j in range(start[f], start[f + 1]):
            t = adj[j]
            if emitted[t]:
                continue
            emitted[t] = 1
            out.append(t)
            for v in tris[t]:
                dead.append(v)
                cand.append(v)
                live[v] -= 1
                if s - stamp[v] > k:
                    stamp[v] = s
                    s += 1
        best, best_p = -1, -1
        for v in cand:
            if live[v] > 0:
                p = s - stamp[v] if s - stamp[v] + 2 * live[v] <= k else 0
                if p > best_p:
                    best, best_p = v, p
        if best < 0:
            while dead:
                d = dead.pop()
                if live[d] > 0:
                    best = d
                    break
        if best < 0:
            while cursor < n and live[cursor] <= 0:
                cursor += 1
            best = cursor if cursor < n else -1
        f = best
    return np.array(out, np.int64)


def fetch_order(vertices, indices):
    """Vertices in order of first use by the triangles; (vertices, indices)."""
    idx = np.asarray(indices, np.int64)
    _, first = np.unique(idx, return_index=True)
    used = idx[np.sort(first)]                       # vertices in order of first use
    remap = np.empty(len(vertices), np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[idx].astype(np.uint32)


def cache_order(vertices, indices):
    """Triangles reordered for the vertex cache, then the vertices in order of first use. Same triangles; (vertices, indices)."""
    tri = np.asarray(indices, np.int64).reshape(-1, 3)
    kept = fetch_order(vertices, indices)
    new = fetch_order(vertices, tri[tipsify(indices, len(vertices))].ravel())
    # a mesh that another tool already ordered can lose from reordering: keep the order whose buffers compress smaller
    packed = lambda vi: len(zlib.compress(np.ascontiguousarray(vi[0]).tobytes() + vi[1].tobytes(), 1))
    return new if packed(new) < packed(kept) else kept


def _is_triangle_geometry():
    """True when the caller is Sollumz's create_geometries for an ordinary mesh. Cable meshes call the same function with vertices that are not
    triangles, so their order must stay as it is."""
    caller = sys._getframe(2)
    return caller.f_code.co_name == "create_geometries" and caller.f_locals.get("is_cable") is False


CLOTH_EXPORT_FUNCTIONS = ("_cloth_env_export", "cloth_env_export")  # ydr/cloth_env.py


def cloth_guard_matches_sollumz():
    """True when the functions the cloth guard looks for exist in Sollumz. If a Sollumz update renames them the guard would silently stop
    working, so positions are then left exactly as they are (a little bigger file, but a cloth vehicle still exports)."""
    try:
        mod = sz._mod("ydr.cloth_env")
    except Exception:
        return False
    return all(hasattr(mod, n) for n in CLOTH_EXPORT_FUNCTIONS)


def _in_cloth_export():
    """True while Sollumz is building a cloth mesh. It matches the geometry's vertices to the cloth's by position (to 0.00001), so those
    positions must stay exactly what Blender has; rounding them would make the export fail."""
    f = sys._getframe(2)
    for _ in range(12):
        if f is None:
            return False
        if f.f_code.co_name in CLOTH_EXPORT_FUNCTIONS:
            return True
        f = f.f_back
    return False


def dedupe_and_get_indices(vertex_arr):
    """Same contract as Sollumz's function of that name: (vertices without duplicates, indices in triangle order)."""
    if _ORIGINAL[0] is not None and _in_cloth_export():
        return _ORIGINAL[0](vertex_arr)
    STATS["stock"] += len(_unique(vertex_arr, {})[1])
    vertex_arr = snap(vertex_arr)
    _, first, inverse = _unique(vertex_arr, DECIMALS)
    STATS["compact"] += len(first)
    out, indices = vertex_arr[first], np.asarray(inverse, dtype=np.uint32).reshape(-1)
    if len(indices) % 3 == 0 and _is_triangle_geometry():
        out, indices = cache_order(out, indices)
    return out, indices


@contextlib.contextmanager
def patched():
    """Sollumz's exporter uses the tolerant merge inside this block."""
    mod = sz.ydrexport()
    original = mod.dedupe_and_get_indices
    mod.dedupe_and_get_indices = dedupe_and_get_indices
    _ORIGINAL[0] = original
    _ROUND_POSITIONS[0] = cloth_guard_matches_sollumz()
    try:
        yield
    finally:
        mod.dedupe_and_get_indices = original
        _ORIGINAL[0] = None
        _ROUND_POSITIONS[0] = True


FORMATS = {"NATIVE": {"NATIVE"}, "CWXML": {"CWXML"}, "BOTH": {"NATIVE", "CWXML"}}


def export_kwargs(fmt):
    """Arguments for sollumz.export_assets: the user's own Sollumz export settings, with the output format(s) replaced by `fmt`."""
    prefs = sz._mod("sollumz_preferences").get_export_settings()
    op_props = bpy.ops.sollumz.export_assets.get_rna_type().properties
    kw = {}
    for p in prefs.bl_rna.properties:
        i = p.identifier
        if i in ("rna_type", "name") or i not in op_props or op_props[i].is_readonly:
            continue
        kw[i] = getattr(prefs, i)
    kw.update(target_formats=FORMATS[fmt], direct_export=True, use_custom_settings=True)
    return kw


EXTENSIONS = {"NATIVE": (".yft",), "CWXML": (".yft.xml",), "BOTH": (".yft", ".yft.xml")}


def check_written(files, fmt):
    """Raise when Sollumz did not write what `fmt` asks for. Sollumz reports its own problems in the Info log and still returns normally,
    so an export that wrote nothing would otherwise look like a success."""
    names = [n.lower() for n, _ in files]
    missing = [e for e in EXTENSIONS[fmt] if not any(n.endswith(e) for n in names)]
    if missing:
        raise RuntimeError(f"Sollumz wrote no {' or '.join(missing)} file. Check the Sollumz export settings (formats, 'Limit to Selected') "
                           "and the Info log for what it reported.")


def export(context, frag, directory, fmt="NATIVE"):
    """Export `frag` with Sollumz's exporter and the user's Sollumz export settings, writing `fmt` ("NATIVE", "CWXML" or "BOTH").
    Returns ([(file, bytes)], stock vertices, written vertices)."""
    view = context.view_layer
    prev_active, prev_sel = view.objects.active, [o for o in view.objects if o.select_get()]
    STATS.update(stock=0, compact=0)
    started = time.time()
    try:
        for o in prev_sel:
            o.select_set(False)
        frag.select_set(True)
        view.objects.active = frag
        with patched():
            bpy.ops.sollumz.export_assets(directory=directory, **export_kwargs(fmt))
    finally:
        frag.select_set(False)
        for o in prev_sel:
            o.select_set(True)
        view.objects.active = prev_active
    files = []
    for root, _, names in os.walk(directory):
        for n in names:
            p = os.path.join(root, n)
            if os.path.getmtime(p) >= started - 1:
                files.append((os.path.relpath(p, directory), os.path.getsize(p)))
    check_written(files, fmt)
    return sorted(files), STATS["stock"], STATS["compact"]


def describe(files, stock, compact):
    mb = lambda b: f"{b / 1048576:.2f} MiB"
    lines = [f"Wrote {len(files)} file(s), {mb(sum(b for _, b in files))} in total."]
    lines += [f"  {n}: {mb(b)}" for n, b in files[:6]]
    if stock:
        lines.append(f"Vertices: {compact} written, Sollumz's own merge would have written {stock} ({100 * (stock - compact) / stock:.1f}% fewer).")
    return "\n".join(lines)
