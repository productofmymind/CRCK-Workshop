"""Regression tests for bugs found in review. Run inside Blender with Sollumz enabled:

    blender -b --python tests/test_regressions.py

Exit code 1 on any failure. Each test builds the synthetic vehicle from tests/synthetic.py (no game assets needed).
"""
import os
import sys
import traceback

import addon_utils
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

addon_utils.enable("bl_ext.user_default.sollumz", default_set=False)
for _mod in list(bpy.context.preferences.addons.keys()):  # an installed copy would register the same operator ids
    if _mod.endswith("vehicle_hierarchy_tools"):
        addon_utils.disable(_mod, default_set=False)
import vehicle_hierarchy_tools as vht  # noqa: E402
from vehicle_hierarchy_tools import analysis, arrange  # noqa: E402
from synthetic import build_vehicle, cube, sollumz  # noqa: E402

vht.register()
results = []


def test(fn):
    try:
        fn()
        results.append((fn.__name__, True))
        print(f"PASS {fn.__name__}", flush=True)
    except Exception:
        results.append((fn.__name__, False))
        print(f"FAIL {fn.__name__}\n{traceback.format_exc()}", flush=True)
    return fn


def codes(frag):
    return [(f.code, f.target) for f in analysis.validate(analysis.load(frag))]


@test
def t_damaged_parts_are_not_nested_or_stray():
    bh, sp = sollumz("tools.blenderhelper"), sollumz("sollumz_properties")
    frag = build_vehicle()
    drawable = bpy.data.objects["testcar_drawable"]
    # the damaged version of the vehicle: its own Drawable with a model, its own Bound Composite with a collision
    dmg = bh.create_empty_object(sp.SollumType.DRAWABLE, "testcar_drawable.damaged")
    dmg.parent = frag
    me = cube("body_m.damaged")
    for m in bpy.data.objects["body_m"].data.materials:
        me.materials.append(m)
    model = bh.create_blender_object(sp.SollumType.DRAWABLE_MODEL, "body_m.damaged", me)
    model.sz_lods.get_lod(sp.LODLevel.HIGH).mesh = me
    model.parent = dmg
    dcomp = bh.create_empty_object(sp.SollumType.BOUND_COMPOSITE, "testcar_composite.damaged")
    dcomp.parent = frag
    col = bh.create_blender_object(sp.SollumType.BOUND_BOX, "col_chassis.damaged", cube("col_chassis.damaged"))
    col.parent = dcomp
    v = analysis.load(frag)
    assert model in v.damaged_models and model not in v.models and model not in v.nested_models
    assert col in v.damaged_bounds and col not in v.bounds
    bad = [c for c in codes(frag) if c[1] in ("body_m.damaged", "col_chassis.damaged")]
    assert not bad, bad  # no NESTED_MODEL, MODEL_UNBOUND, BOUND_UNBOUND ... for the damaged version
    planned, _ = arrange.plan(v)
    assert not [f for f in planned if "body_m.damaged" in str(f) or "col_chassis.damaged" in str(f)], planned  # Auto Arrange leaves them where they are
    assert model.parent is dmg and col.parent is dcomp
    # a genuinely nested model is still reported
    nested = bh.create_blender_object(sp.SollumType.DRAWABLE_MODEL, "nested_m", cube("nested_m"))
    nested.parent = bpy.data.objects["body_m"]
    assert ("NESTED_MODEL", "nested_m") in codes(frag)
    assert drawable is analysis.load(frag).drawable


@test
def t_crckworkshop_export_leaves_cloth_vertices_alone():
    import numpy as np
    from vehicle_hierarchy_tools import compactexport as ce
    dt = np.dtype([("Position", "f4", 3), ("Normal", "f4", 3)])
    v = np.zeros(6, dtype=dt)
    v["Position"] = np.random.RandomState(3).rand(6, 3) * 3 + 0.00003  # off the rounding grid
    v["Normal"] = (0, 0, 1)
    seen = []

    def stock(arr):  # stands in for Sollumz's own function
        seen.append(arr)
        return arr, np.arange(len(arr), dtype=np.uint32)

    def _cloth_env_export():  # what Sollumz calls while it builds a cloth mesh
        return ce.dedupe_and_get_indices(v)

    def ordinary_export():
        return ce.dedupe_and_get_indices(v)

    ce._ORIGINAL[0] = stock
    try:
        out, _ = _cloth_env_export()
        assert seen and np.array_equal(out["Position"], v["Position"]), "cloth: Sollumz's own merge, positions untouched"
        seen.clear()
        out, _ = ordinary_export()
        assert not seen and not np.array_equal(out["Position"], v["Position"]), "everything else: rounded"
    finally:
        ce._ORIGINAL[0] = None
    with ce.patched():  # and the hook is set and cleared by patched()
        assert ce._ORIGINAL[0] is not None
    assert ce._ORIGINAL[0] is None


@test
def t_crckworkshop_export_reports_an_export_that_wrote_nothing():
    import tempfile
    from vehicle_hierarchy_tools import compactexport as ce
    for files, fmt in (([], "NATIVE"), ([("v\blank.dds", 144)], "NATIVE"), ([("v.yft", 9)], "CWXML"), ([("v.yft", 9)], "BOTH"), ([("v.yft.xml", 9)], "BOTH")):
        try:
            ce.check_written(files, fmt)
        except RuntimeError as e:
            assert "Sollumz wrote no" in str(e)
        else:
            raise AssertionError(f"{files} {fmt} should be reported")
    ce.check_written([("v.yft", 9), ("v\blank.dds", 144)], "NATIVE")
    ce.check_written([("v.yft.xml", 9)], "CWXML")
    ce.check_written([("v.yft", 9), ("v.yft.xml", 9)], "BOTH")
    frag = build_vehicle()
    files, _, _ = ce.export(bpy.context, frag, tempfile.mkdtemp(prefix="vht_ex_"), "BOTH")  # a real export still passes the check
    assert any(n.endswith(".yft") for n, _ in files) and any(n.endswith(".yft.xml") for n, _ in files)


@test
def t_apply_that_fails_keeps_its_undo_step():
    import json
    import types
    frag = build_vehicle()
    bpy.context.view_layer.objects.active = frag
    st = bpy.context.window_manager.vht
    st.plan.clear()
    for fx in ({"op": "enable_physics", "bone": "bonnet"}, {"op": "enable_physics", "bone": "no_such_bone"}):
        it = st.plan.add()
        it.text, it.kind, it.fix, it.enabled = "x", "physics", json.dumps(fx), True
    said = []
    fake = types.SimpleNamespace(report=lambda kind, msg: said.append((kind, msg)))
    result = vht.operators.VHT_OT_apply.execute(fake, bpy.context)
    assert result == {"FINISHED"}, result  # a cancelled operator gets no undo step, so Ctrl+Z would undo something earlier instead
    assert said and "ERROR" in said[-1][0] and "Ctrl+Z" in said[-1][1], said
    assert len(st.plan) == 0, "the list is cleared, so the done part is not applied twice"
    assert frag.data.bones["bonnet"].sollumz_use_physics, "the part before the failure is in; that is what the undo step is for"


@test
def t_estimate_size_works_for_a_duplicated_name():
    import tempfile
    from vehicle_hierarchy_tools import sizing
    frag = build_vehicle()
    frag.name = "testcar.001"  # Blender's suffix for a duplicate; Sollumz writes testcar.yft.xml
    m = sizing.estimate(bpy.context, frag)
    assert m["total"] > 0 and m["vertex"] > 0, m
    out = tempfile.mkdtemp(prefix="vht_xml_")
    for name in ("a_hi.yft.xml", "a.yft.xml"):
        open(os.path.join(out, name), "w").close()
    assert sizing.find_xml(out) == "a.yft.xml", "the _hi file is not the one measured"
    try:
        sizing.find_xml(tempfile.mkdtemp(prefix="vht_empty_"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("an empty folder is reported")


@test
def t_sollumz_version_is_read_once():
    import builtins
    from vehicle_hierarchy_tools import sollumz_bridge as sz
    sz._VERSION.clear()
    reads = []
    real = builtins.open

    def counting(path, *a, **k):
        if str(path).endswith("blender_manifest.toml"):
            reads.append(path)
        return real(path, *a, **k)

    builtins.open = counting
    try:
        first = [sz.version() for _ in range(50)]
        for _ in range(50):
            sz.status()
    finally:
        builtins.open = real
    assert len(reads) <= 1 and len(set(first)) == 1 and first[0], (len(reads), first[:2])  # a legacy add-on (4.0/4.1) has no manifest to read
    assert sz.status()[0], "status still works from the cache"


@test
def t_crckworkshop_export_keeps_positions_when_the_cloth_guard_cannot_work():
    import numpy as np
    from vehicle_hierarchy_tools import compactexport as ce
    assert ce.cloth_guard_matches_sollumz(), "the function names the guard looks for exist in this Sollumz"
    v = np.zeros(6, dtype=np.dtype([("Position", "f4", 3), ("Normal", "f4", 3)]))
    v["Position"] = np.random.RandomState(5).rand(6, 3) * 3 + 0.00003
    v["Normal"] = (0, 0, 1)
    real = ce.CLOTH_EXPORT_FUNCTIONS
    ce.CLOTH_EXPORT_FUNCTIONS = ("renamed_by_a_sollumz_update",)
    try:
        with ce.patched():
            assert not ce.cloth_guard_matches_sollumz() and ce._ROUND_POSITIONS[0] is False
            out = ce.snap(v)
            assert np.array_equal(out["Position"], v["Position"]), "positions stay exact when the guard cannot be trusted"
    finally:
        ce.CLOTH_EXPORT_FUNCTIONS = real
    assert ce._ROUND_POSITIONS[0] is True
    with ce.patched():
        assert ce._ROUND_POSITIONS[0] is True and not np.array_equal(ce.snap(v)["Position"], v["Position"]), "normally they are rounded"
    assert ce._ROUND_POSITIONS[0] is True


@test
def t_paint_variant_is_not_borrowed_from_outside_the_vehicle():
    from vehicle_hierarchy_tools import paint
    shaders = sollumz("ydr.shader_materials")
    frag = build_vehicle()
    base = shaders.create_shader("vehicle_paint4.sps")
    base.sz_paint_layer = "PRIMARY"
    body = bpy.data.objects["body_m"]
    body.data.materials.append(base)
    foreign = base.copy()  # a material of some other vehicle in the same .blend, or a leftover: no part of this vehicle uses it
    foreign.sz_paint_layer = "SECONDARY"
    v = analysis.load(frag)
    new, created = paint.make_variant(v, base, "SECONDARY")
    assert created and new is not foreign, "a variant of its own is made, not the foreign material"
    body.data.materials.append(new)  # now the vehicle uses it: asking again finds it
    new2, created2 = paint.make_variant(analysis.load(frag), base, "SECONDARY")
    assert new2 is new and not created2


@test
def t_create_material_with_nothing_selected_makes_nothing():
    import types
    frag = build_vehicle()
    for o in bpy.data.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = frag  # the fragment itself is selected, no model of it
    bpy.context.window_manager.vht.vehicle_shader = "vehicle_shuts.sps"
    before = len(bpy.data.materials)
    said = []
    result = vht.operators.VHT_OT_shader_add.execute(types.SimpleNamespace(report=lambda kind, msg: said.append((kind, msg))), bpy.context)
    assert result == {"CANCELLED"} and said and "ERROR" in said[0][0], (result, said)
    assert len(bpy.data.materials) == before, "no material was left behind"
    body = bpy.data.objects["body_m"]
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    result = vht.operators.VHT_OT_shader_add.execute(types.SimpleNamespace(report=lambda kind, msg: said.append((kind, msg))), bpy.context)
    assert result == {"FINISHED"} and len(bpy.data.materials) == before + 1, "with a model selected it still works"


@test
def t_part_collections_survive_renaming_the_fragment():
    from vehicle_hierarchy_tools import fixes
    frag = build_vehicle()
    fixes.build_part_collections(analysis.load(frag))
    mine = lambda: [c for c in bpy.data.collections if c.get("vht_frag")]
    first = len(mine())
    assert first == len(frag.data.bones) + 1
    frag.name = "renamed_car"
    created, _ = fixes.build_part_collections(analysis.load(frag))
    assert created == 0 and len(mine()) == first, "the same collections are found again, not a second set"
    assert bpy.data.collections["renamed_car parts"], "the root follows the new name"
    restored = fixes.remove_part_collections(frag)
    assert restored > 0 and not mine(), "Remove takes every one of them away"
    # a collection made before ids existed (found by the fragment's name) is adopted
    frag2 = build_vehicle()
    fixes.build_part_collections(analysis.load(frag2))
    for c in mine():
        del c["vht_frag_id"]
    del frag2["vht_id"]
    created, _ = fixes.build_part_collections(analysis.load(frag2))
    assert created == 0 and len(mine()) == len(frag2.data.bones) + 1 and all(c.get("vht_frag_id") for c in mine())
    # a duplicated fragment does not share the collections of the original
    copy = frag2.copy()
    assert fixes._frag_id(copy) != frag2["vht_id"]


@test
def t_rename_with_nothing_typed_is_refused_and_zero_mass_collision_is_still_reported():
    import types
    frag = build_vehicle()
    st = bpy.context.window_manager.vht
    st.rename_scope, st.rename_find, st.rename_replace = "ACTIVE", "", ""
    said = []
    result = vht.operators.VHT_OT_rename_part.execute(types.SimpleNamespace(report=lambda kind, msg: said.append((kind, msg))), bpy.context)
    assert result == {"CANCELLED"} and said and said[0][0] == {"ERROR"}, (result, said)
    assert not hasattr(analysis, "shader_name")
    bpy.data.objects["col_door"].child_properties.mass = 0.0  # the condition that was tidied
    assert ("COLLISION_MASS_ZERO", "col_door") in codes(frag)


@test
def t_bl_info_matches_the_manifest():
    import re
    manifest = open(os.path.join(os.path.dirname(HERE), "vehicle_hierarchy_tools", "blender_manifest.toml"), encoding="utf-8").read()
    want = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.M).group(1)
    assert ".".join(map(str, vht.bl_info["version"])) == want, (vht.bl_info["version"], want)
    assert vht.bl_info["blender"] == (4, 0, 0)


@test
def t_sollumz_2_8_fallbacks():
    """Sollumz 2.8 names things differently; hide the 2.9 names and check the bridge uses the 2.8 ones."""
    from vehicle_hierarchy_tools import sollumz_bridge as sz
    ex, bh = sz._mod("ydr.ydrexport"), sz._mod("tools.blenderhelper")
    sp = sz._mod("sollumz_properties")
    # joint limits: 2.8 has get_limit_rot_constraint / get_limit_pos_constraint
    frag = build_vehicle()
    calls = []
    saved = {n: getattr(ex, n) for n in ("find_bone_constraint_rotation_limit", "find_bone_constraint_translation_limit") if hasattr(ex, n)}  # real 2.8 has none
    try:
        for n in saved:
            delattr(ex, n)
        ex.get_limit_rot_constraint = lambda pb: calls.append("rot")
        ex.get_limit_pos_constraint = lambda pb: calls.append("pos")
        assert sz.has_joint_limit(frag, frag.data.bones[0].name) is False and calls == ["rot", "pos"], calls
    finally:
        for n, f in saved.items():
            setattr(ex, n, f)
        del ex.get_limit_rot_constraint, ex.get_limit_pos_constraint
    # new objects: 2.8's create_blender_object always links to the active collection
    original = bh.create_blender_object
    target = bpy.data.collections.new("vht_target")
    bpy.context.scene.collection.children.link(target)
    try:
        bh.create_blender_object = lambda sollum_type, name=None, object_data=None: original(sollum_type, name=name, object_data=object_data)
        obj = sz.new_bound_geometry("vht_bound", bpy.data.meshes.new("vht_bound"), target)
        assert list(obj.users_collection) == [target], [c.name for c in obj.users_collection]
        assert obj.sollum_type == sp.SollumType.BOUND_GEOMETRY
    finally:
        bh.create_blender_object = original
    # shader parameters: 2.8 keeps a texture name, x/y/z/w or a list of vectors instead of .value
    class Tex: texture_name = "vehicle_generic_mud"
    class Vec: x, y, z, w = 1.0, 0.5, 0.0, 0.25
    class Arr: values = [(1, 2, 3, 4), (5, 6, 7, 8)]
    assert sz._param_value(Tex()) == "vehicle_generic_mud"
    assert sz._param_value(Vec()) == (1.0, 0.5, 0.0, 0.25)
    assert sz._param_value(Arr()) == ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0))


@test
def t_sollumz_version_without_tomllib():
    """Blender 4.0's Python 3.10 has no tomllib; the version must still be read."""
    from vehicle_hierarchy_tools import sollumz_bridge as sz
    sz._VERSION.clear()
    saved = sys.modules.get("tomllib")
    sys.modules["tomllib"] = None  # makes `import tomllib` fail like on Python 3.10
    try:
        ver = sz.version()
    finally:
        if saved is None:
            del sys.modules["tomllib"]
        else:
            sys.modules["tomllib"] = saved
    assert ver and ver[0].isdigit(), ver


@test
def t_lods_keep_seams_closed_and_materials():
    """The simplifier on a game-style mesh (split at UV seams, two materials): targets reached, no cracks, no needles, every material kept."""
    import bmesh
    import numpy as np
    from vehicle_hierarchy_tools import lodgen
    me = bpy.data.meshes.new("vht_lod_sphere")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0, calc_uvs=True)
    for f in bm.faces:
        f.material_index = 1 if f.calc_center_median().z > 0.6 else 0
    # split like a game mesh: separate vertices along the UV seam (one meridian) and along the material border
    seam = [e for e in bm.edges if all(abs(v.co.y) < 1e-6 and v.co.x > 1e-6 for v in e.verts)]
    border = [e for e in bm.edges if len(e.link_faces) == 2 and e.link_faces[0].material_index != e.link_faces[1].material_index]
    bmesh.ops.split_edges(bm, edges=seam + border)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(bpy.data.materials.new("vht_a"))
    me.materials.append(bpy.data.materials.new("vht_b"))
    assert len(me.vertices) > 48 * 23 + 2, "the seams were split"

    def open_length(mesh):
        mesh.calc_loop_triangles()
        co = np.array([v.co for v in mesh.vertices])
        _, w = np.unique(np.round(co / 1e-5), axis=0, return_inverse=True)
        w = w.ravel()
        e = {}
        for t in mesh.loop_triangles:
            a, b, c = (int(w[i]) for i in t.vertices)
            for x, y in ((a, b), (b, c), (c, a)):
                k = (min(x, y), max(x, y))
                e[k] = e.get(k, 0) + 1
        return sum(1 for n in e.values() if n == 1)

    assert open_length(me) == 0, "the test sphere is closed once welded"
    tris = lodgen.triangles(me)
    out = lodgen.reduce_levels(me, [tris // 3, tris // 10])
    for target, lod in out.items():
        n = lodgen.triangles(lod)
        assert n <= target * 1.05 + 4, (target, n)
        assert open_length(lod) == 0, "a crack opened at a seam"
        assert {p.material_index for p in lod.polygons} == {0, 1}, "a material vanished"
        assert len(lod.uv_layers) == len(me.uv_layers)
        lod.calc_loop_triangles()
        co = np.array([v.co for v in lod.vertices])
        tv = np.array([t.vertices for t in lod.loop_triangles])
        a, b, c = co[tv[:, 0]], co[tv[:, 1]], co[tv[:, 2]]
        area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        s = ((b - a) ** 2).sum(1) + ((c - b) ** 2).sum(1) + ((a - c) ** 2).sum(1)
        quality = 2 * np.sqrt(3) * area2 / s
        assert (quality < 0.1).mean() < 0.05, f"{(quality < 0.1).mean():.0%} needle triangles"
        assert abs(np.linalg.norm(co, axis=1) - 1.0).max() < 0.15, "the surface moved too far"


@test
def t_lone_model_gets_lods_and_material_tools():
    """A Drawable Model outside any fragment: Generate LODs and Create Material work on it; the vehicle-only panels stay hidden."""
    import bmesh
    from vehicle_hierarchy_tools import ui
    build_vehicle()  # a vehicle in the scene too: the lone model must not be taken for one of its parts
    bh, sp = sollumz("tools.blenderhelper"), sollumz("sollumz_properties")
    me = bpy.data.meshes.new("lone_m")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.5, calc_uvs=True)
    bm.to_mesh(me)
    bm.free()
    obj = bh.create_blender_object(sp.SollumType.DRAWABLE_MODEL, "lone_m", me)
    if obj.sz_lods.high.mesh is None:
        obj.sz_lods.high.mesh = me
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    assert not ui.has_vehicle(bpy.context) and ui.has_model(bpy.context)
    assert not ui.VHT_PT_materials.poll(bpy.context) and ui.VHT_PT_mat_shader.poll(bpy.context) and ui.VHT_PT_lods.poll(bpy.context)
    st = bpy.context.window_manager.vht
    st.lod_start, st.lod_scope, st.lod_overwrite = "high", "ALL", True
    assert bpy.ops.vht.lods_generate() == {"FINISHED"}
    assert obj.sz_lods.medium.mesh is not None and len(obj.sz_lods.medium.mesh.polygons) < len(me.polygons)
    st.vehicle_shader = "vehicle_paint1.sps"
    before = len(obj.data.materials)
    assert bpy.ops.vht.shader_add() == {"FINISHED"}
    assert len(obj.data.materials) == before + 1


@test
def t_transfer_materials_adds_slots_and_moves_no_face():
    frag = build_vehicle()
    src, dst = bpy.data.objects["body_m"], bpy.data.objects["door_m"]
    extra = bpy.data.materials.new("vht_transfer_src")
    src.data.materials.append(extra)
    before_slots = list(dst.data.materials)
    before_faces = [p.material_index for p in dst.data.polygons]
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.mode != "OBJECT" else None
    for o in bpy.context.selected_objects:
        o.select_set(False)
    dst.select_set(True)
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    assert bpy.ops.vht.mat_transfer(mode="ALL") == {"FINISHED"}
    assert list(dst.data.materials)[:len(before_slots)] == before_slots, "existing slots unchanged"
    assert extra in list(dst.data.materials), "the source's materials were added"
    assert [p.material_index for p in dst.data.polygons] == before_faces, "no face changed material"
    n = len(dst.data.materials)
    assert bpy.ops.vht.mat_transfer(mode="ALL") == {"FINISHED"} and len(dst.data.materials) == n, "nothing added twice"


fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} passed")
sys.exit(1 if fails else 0)
