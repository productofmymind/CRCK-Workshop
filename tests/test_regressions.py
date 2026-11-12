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


@test
def t_rear_rotor_aligns_local_x_and_origin_moves_to_spin_centre():
    import math
    from mathutils import Matrix, Vector
    from vehicle_hierarchy_tools import axes
    frag = build_vehicle()
    bh, sp = sollumz("tools.blenderhelper"), sollumz("sollumz_properties")
    head = Vector((0.5, -3.0, 1.0))
    bpy.ops.object.mode_set(mode="EDIT")
    eb = frag.data.edit_bones.new("rotor_rear")
    eb.head, eb.tail = head, head + Vector((0, 0, 0.1))
    eb.parent = frag.data.edit_bones["chassis"]
    eb.matrix = Matrix.Translation(head)  # identity rotation, as vanilla rear rotors have
    bpy.ops.object.mode_set(mode="OBJECT")
    # a disc whose normal is 3 degrees off the bone's X and whose centre is 30 mm off its origin
    tilt = Matrix.Rotation(math.radians(3.0), 3, "Z")
    centre = Vector((0.0, 0.03, 0.0))
    verts = [centre + tilt @ Vector((0.0, r * math.cos(a), r * math.sin(a)))
             for r, n in ((1.0, 24), (0.2, 12)) for a in (2 * math.pi * i / n for i in range(n))]
    me = bpy.data.meshes.new("rotor_m")
    me.from_pydata(verts, [], [])
    o = bh.create_blender_object(sp.SollumType.DRAWABLE_MODEL, "rotor_m", me)
    o.sz_lods.get_lod(sp.LODLevel.HIGH).mesh = me
    o.parent = bpy.data.objects["testcar_drawable"]
    o.location = head
    bh.add_child_of_bone_constraint(o, frag, "rotor_rear")
    o.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    assert abs(axes.spin_wobble(frag, "rotor_rear") - 3.0) < 0.05, "the rear rotor's wobble is measured from local X"
    assert abs(axes.spin_centre(frag, "rotor_rear")[1] - 0.03) < 0.002
    found = codes(frag)
    assert ("HV090_ROTOR_AXIS", "rotor_rear") in found and ("HV091_ROTOR_ORIGIN", "rotor_rear") in found
    for mode in ("ALIGN", "SPIN"):
        lines, ok = axes.set_axes(bpy.context, frag, "rotor_rear", mode)
        assert ok, lines
    assert axes.spin_wobble(frag, "rotor_rear") < 0.05 and axes.spin_centre(frag, "rotor_rear")[1] < 0.001
    x = frag.data.bones["rotor_rear"].matrix_local.to_3x3().col[0]
    assert x.angle(Vector((1, 0, 0))) < math.radians(4), "local X stays the spin axis: the rotor is not turned sideways"
    found = codes(frag)
    assert ("HV090_ROTOR_AXIS", "rotor_rear") not in found and ("HV091_ROTOR_ORIGIN", "rotor_rear") not in found


@test
def t_glass_collision_faces_outwards_on_a_mirrored_model_and_takes_only_its_bone():
    from mathutils import Vector
    from vehicle_hierarchy_tools import fixes, sollumz_bridge as sz

    def outward(obj):  # share of triangles whose world-space winding faces away from the object's centre
        pts = [obj.matrix_world @ x.co for x in obj.data.vertices]
        mid = sum(pts, Vector()) / len(pts)
        good = 0
        for p in obj.data.polygons:
            a, b, c = (pts[i] for i in p.vertices[:3])
            good += (b - a).cross(c - a).dot((a + b + c) / 3 - mid) > 0
        return good / len(obj.data.polygons)

    for mirrored in (False, True):
        frag = build_vehicle()
        model = bpy.data.objects["window_m"]
        if mirrored:
            model.scale = (-1.0, 1.0, 1.0)
        bpy.context.view_layer.update()
        assert (model.matrix_world.determinant() < 0) == mirrored
        v = analysis.load(frag)
        fixes.make_glass_collision(v, "window_lf", "window_m")
        bpy.context.view_layer.update()
        col = bpy.data.objects["window_lf.col"]
        assert col.matrix_world.determinant() > 0, "a mirrored bound is exported inside out"
        assert outward(col) == 1.0, f"mirrored={mirrored}: {outward(col)} of the triangles face outwards"
        want = sorted(tuple(round(c, 4) for c in model.matrix_world @ x.co) for x in model.data.vertices)
        got = sorted(tuple(round(c, 4) for c in col.matrix_world @ x.co) for x in col.data.vertices)
        assert got == want, "the collision sits on the pane"
        assert sz.child_of_bone(col).name == "window_lf" and col.parent == v.composite

    # one mesh skinned to two bones: each window gets its own glass only
    build_vehicle()
    me = cube("two_windows")
    me.materials.append(bpy.data.materials["mat2"])
    far = cube("far")
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.from_mesh(far)
    for x in bm.verts[8:]:
        x.co.x += 2.0
    bm.to_mesh(me)
    bm.free()
    bh, sp = sollumz("tools.blenderhelper"), sollumz("sollumz_properties")
    o = bh.create_blender_object(sp.SollumType.DRAWABLE_MODEL, "two_windows", me)
    o.sz_lods.get_lod(sp.LODLevel.HIGH).mesh = me
    o.vertex_groups.new(name="window_lf").add(list(range(8)), 1.0, "REPLACE")
    o.vertex_groups.new(name="bonnet").add(list(range(8, 16)), 1.0, "REPLACE")
    assert len(fixes.glass_pane(o)[0]) == 16
    verts, faces = fixes.glass_pane(o, "window_lf")
    assert len(verts) == 8 and len(faces) == 12 and all(x.x < 1.0 for x in verts)


@test
def t_fill_textures_names_shared_textures_and_embeds_only_generated_ones():
    from vehicle_hierarchy_tools import sollumz_bridge as sz, textures
    build_vehicle()
    mat = sz.create_shader_material("vehicle_vehglass.sps")
    nodes = [n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"]
    for n in nodes:
        n.texture_properties.embedded = True  # as a material copied from an embedded one has it
    done = dict(textures.apply_defaults(mat))
    assert done.get("DiffuseSampler") and done.get("SpecSampler") == "vehicle_generic_smallspecmap", done
    for n in nodes:
        if n.name in done:
            assert n.sollumz_texture_name == done[n.name]
            assert n.texture_properties.embedded == (done[n.name] in textures.GENERATED), n.name
    assert textures.apply_defaults(mat) == [], "filled samplers are left alone"


@test
def t_lock_material_order_makes_indices_unique_and_keeps_the_order():
    from vehicle_hierarchy_tools import fixes
    frag = build_vehicle()
    for m in bpy.data.materials:
        if m.name.startswith("mat"):
            m.shader_properties.index = 0
    v = analysis.load(frag)
    before = [m.name for m in analysis.export_order(v)[0]]
    assert "MAT_INDEX_TIES" in [c for c, _ in codes(frag)]
    assert fixes.describe(v, {"op": "lock_material_order"})[0].endswith("indices made unique")
    fixes.apply(bpy.context, v, [{"op": "lock_material_order"}])
    after = analysis.export_order(analysis.load(frag))[0]
    assert [m.name for m in after] == before
    assert [m.shader_properties.index for m in after] == list(range(len(after)))
    assert "MAT_INDEX_TIES" not in [c for c, _ in codes(frag)]


@test
def t_empty_material_slot_is_reported_and_does_not_stop_the_checks():
    from vehicle_hierarchy_tools import fixes
    frag = build_vehicle()
    kept = bpy.data.objects["door_m"].data.materials[0]
    bpy.data.objects["body_m"].data.materials[0] = None  # what deleting a material in Blender leaves behind
    found = codes(frag)  # Sollumz's own material and model walks raise on the empty slot
    assert ("EMPTY_MATERIAL_SLOT", "body_m") in found, found
    v = analysis.load(frag)
    assert kept in analysis.export_order(v)[0] and None not in analysis.export_order(v)[0]
    assert analysis.order_report(v)["models"], "the model order is still reported"
    # merging into a material that no longer exists must not empty the slots
    line = fixes.merge_materials(v, "no_such_material", [kept.name])
    assert line.startswith("Nothing merged") and bpy.data.objects["door_m"].data.materials[0] is kept
    # a model with no material at all: Sollumz drops its geometry with only a log warning
    bpy.data.objects["boot_m"].data.materials.clear()
    found = codes(frag)
    assert ("MODEL_NO_MATERIAL", "boot_m") in found and ("MODEL_NO_MATERIAL", "door_m") not in found, found


@test
def t_middle_window_is_glass_and_may_sit_under_its_sides_door():
    from vehicle_hierarchy_tools import rules
    r = rules.load()
    for name, door, other in (("window_lm", "door_dside_f", "door_pside_f"), ("window_rm", "door_pside_r", "door_dside_r")):
        rule = r.parent_rule(name)
        assert rule and rule.get("glass"), name
        assert r.parent_ok("chassis", rule["parents"]) and r.parent_ok(door, rule["parents"])
        assert not r.parent_ok(other, rule["parents"]), f"{name} under the other side's door is still reported"
    # in a door: no parent finding, and the missing collision is offered as a glass fix
    frag = build_vehicle()
    bpy.ops.object.mode_set(mode="EDIT")
    eb = frag.data.edit_bones.new("window_lm")
    eb.head, eb.tail = (0.9, 0.3, 0.9), (0.9, 0.3, 1.0)
    eb.parent = frag.data.edit_bones["door_dside_f"]
    bpy.ops.object.mode_set(mode="OBJECT")
    found = [f for f in analysis.validate(analysis.load(frag)) if f.target == "window_lm"]
    assert not [f for f in found if f.code in ("HV020_PARENT", "HV021_WINDOW_NOT_IN_DOOR")], [f.code for f in found]
    assert "HV040_GLASS_NO_COLLISION" in [f.code for f in found]


# --- every quick fix, on a vehicle broken in just that way -----------------------------------------------------------------------
# quick_fix() applies the fix a finding offers and checks what all of them promise: the finding goes away, no object moves, no bone's
# rest pose changes, and nothing new is reported.

def add_model(frag, name, bone=None, at=(0.0, 0.0, 0.0), mat="mat0"):
    bh, sp = sollumz("tools.blenderhelper"), sollumz("sollumz_properties")
    me = cube(name)
    me.materials.append(bpy.data.materials[mat])
    o = bh.create_blender_object(sp.SollumType.DRAWABLE_MODEL, name, me)
    o.sz_lods.get_lod(sp.LODLevel.HIGH).mesh = me
    o.parent = bpy.data.objects["testcar_drawable"]
    o.location = at
    if bone:
        o.location = frag.data.bones[bone].head_local
        bh.add_child_of_bone_constraint(o, frag, bone)
        o.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    return o


def add_bone(frag, name, parent, head):
    from mathutils import Vector
    bpy.context.view_layer.objects.active = frag
    bpy.ops.object.mode_set(mode="EDIT")
    eb = frag.data.edit_bones.new(name)
    eb.head, eb.tail = Vector(head), Vector(head) + Vector((0, 0, 0.1))
    eb.parent = frag.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")


def _differs(a, b, tol=1e-4):
    return any(abs(x - y) > tol for r1, r2 in zip(a, b) for x, y in zip(r1, r2))


def quick_fix(frag, code, target=None, allow_new=(), renamed=()):
    from vehicle_hierarchy_tools import fixes
    bpy.context.view_layer.update()
    v = analysis.load(frag)
    before = analysis.validate(v)
    hits = [f for f in before if f.code == code and (target is None or f.target == target)]
    assert hits, f"{code} not reported for {target}: {sorted({(f.code, f.target) for f in before})}"
    assert hits[0].fix, f"{code} offers no quick fix"
    text = fixes.describe(v, hits[0].fix)[0]
    assert text and not text.endswith("no change"), text
    objs = {o.name: o.matrix_world.copy() for o in bpy.data.objects}
    rest = {b.name: b.matrix_local.copy() for b in frag.data.bones}
    report = fixes.apply(bpy.context, v, [hits[0].fix])
    bpy.context.view_layer.update()
    after = analysis.validate(analysis.load(frag))
    assert not [f for f in after if f.code == code and f.target == hits[0].target], f"{code} is still reported: {report}"
    moved = [n for n, m in objs.items() if n in bpy.data.objects and _differs(m, bpy.data.objects[n].matrix_world)]
    assert not moved, f"{code}: objects moved: {moved}"
    turned = [b.name for b in frag.data.bones if b.name in rest and _differs(rest[b.name], b.matrix_local)]
    assert not turned, f"{code}: bone rest pose changed: {turned}"
    gone = sorted(set(objs) - set(bpy.data.objects.keys()) - set(renamed))
    assert not gone, f"{code}: objects removed: {gone}"
    new = {(f.code, f.target) for f in after} - {(f.code, f.target) for f in before}
    new = {x for x in new if x[0] not in allow_new}
    assert not new, f"{code}: the fix caused new findings: {sorted(new)}"
    return report


@test
def t_fix_nested_model_is_moved_under_the_drawable():
    frag = build_vehicle()
    door, body = bpy.data.objects["door_m"], bpy.data.objects["body_m"]
    bpy.context.view_layer.update()
    door.parent = body
    bpy.context.view_layer.update()
    world = door.matrix_world.copy()
    quick_fix(frag, "NESTED_MODEL", "door_m")
    assert door.parent == bpy.data.objects["testcar_drawable"], door.parent
    assert not _differs(world, door.matrix_world), (world.to_translation(), door.matrix_world.to_translation())
    assert sollumz("tools.blenderhelper").get_child_of_bone(door).name == "door_dside_f"


@test
def t_fix_unbound_model_and_collision_are_bound_to_the_bone_they_are_named_after():
    frag = build_vehicle()
    sz = vht.sollumz_bridge
    model = add_model(frag, "bonnet", at=(0.3, 1.2, 0.9))
    quick_fix(frag, "MODEL_UNBOUND", "bonnet")
    assert sz.child_of_bone(model).name == "bonnet"
    col = bpy.data.objects["col_door"]
    col.name = "boot.col"  # bound to the door, named after the boot
    # the door loses its only collision, which is reported next; the boot gets Use Physics so its new collision is exported
    assert not frag.data.bones["boot"].sollumz_use_physics
    quick_fix(frag, "BOUND_ELSEWHERE", "boot.col", allow_new=("PHYSICS_NO_COLLISION",))
    assert frag.data.bones["boot"].sollumz_use_physics
    frag.data.bones["boot"].sollumz_use_physics = False
    assert sz.child_of_bone(col).name == "boot"
    sz.unbind(col)
    quick_fix(frag, "BOUND_UNBOUND", "boot.col")
    assert sz.child_of_bone(col).name == "boot" and frag.data.bones["boot"].sollumz_use_physics


@test
def t_fix_collision_outside_the_composite_is_adopted():
    frag = build_vehicle()
    col = bpy.data.objects["col_window"]
    bpy.context.view_layer.update()
    world = col.matrix_world.copy()
    col.parent = bpy.data.objects["testcar_drawable"]
    bpy.context.view_layer.update()
    quick_fix(frag, "BOUND_OUTSIDE_COMPOSITE", "col_window")
    assert col.parent == bpy.data.objects["testcar_composite"] and not _differs(world, col.matrix_world)
    assert vht.sollumz_bridge.child_of_bone(col).name == "window_lf"


@test
def t_fix_use_physics_is_switched_on_for_a_bone_with_a_collision():
    frag = build_vehicle()
    frag.data.bones["door_dside_f"].sollumz_use_physics = False
    quick_fix(frag, "BOUND_BONE_NO_PHYSICS", "col_door")
    assert frag.data.bones["door_dside_f"].sollumz_use_physics


@test
def t_fix_bone_name_suffix_and_typo_are_renamed_and_bindings_follow():
    sz = vht.sollumz_bridge
    for wrong, code in (("boot.001", "BONE_NAME_SUFFIX"), ("bonnett", "HV003_UNKNOWN_BONE")):
        frag = build_vehicle()
        right = "boot" if wrong.startswith("boot") else "bonnet"
        model = bpy.data.objects[right + "_m"]
        frag.data.bones[right].name = wrong
        assert sz.child_of_bone(model).name == wrong
        quick_fix(frag, code, wrong, allow_new=("HV030_NO_COLLISION",))
        assert right in frag.data.bones and wrong not in frag.data.bones
        assert sz.child_of_bone(model).name == right
        assert frag.data.bones[right].bone_properties.tag == sz.tag_hash(right)


@test
def t_fix_manual_tag_follows_the_name_again():
    sz = vht.sollumz_bridge
    frag = build_vehicle()
    props = frag.data.bones["bonnet"].bone_properties
    props.use_manual_tag, props.manual_tag = True, 1234
    quick_fix(frag, "TAG_MISMATCH", "bonnet")
    assert props.tag == sz.tag_hash("bonnet")
    root = frag.data.bones["chassis"].bone_properties
    root.use_manual_tag, root.manual_tag = True, 77
    quick_fix(frag, "HV004_ROOT_TAG", "chassis")
    assert root.tag == 0


@test
def t_fix_window_goes_back_under_its_door():
    frag = build_vehicle()
    from vehicle_hierarchy_tools import fixes
    order = [b.name for b in frag.data.bones]
    fixes.apply(bpy.context, analysis.load(frag), [{"op": "reparent_bone", "bone": "window_lf", "parent": "bonnet"}])
    assert frag.data.bones["window_lf"].parent.name == "bonnet"
    tags = {b.name: b.bone_properties.tag for b in frag.data.bones}
    limits = {pb.name: [c.type for c in pb.constraints] for pb in frag.pose.bones}
    quick_fix(frag, "HV021_WINDOW_NOT_IN_DOOR", "window_lf")
    assert frag.data.bones["window_lf"].parent.name == "door_dside_f"
    index = {b.name: i for i, b in enumerate(frag.data.bones)}
    assert sorted(index) == sorted(order) and all(index[b.parent.name] < index[b.name] for b in frag.data.bones if b.parent)
    assert {b.name: b.bone_properties.tag for b in frag.data.bones} == tags
    assert {pb.name: [c.type for c in pb.constraints] for pb in frag.pose.bones} == limits, "joint limits survive the edit"
    for name, bone in (("window_m", "window_lf"), ("col_window", "window_lf"), ("door_m", "door_dside_f")):
        assert vht.sollumz_bridge.child_of_bone(bpy.data.objects[name]).name == bone


@test
def t_fix_wheel_gets_a_disc_and_the_other_wheel_a_copy_of_it():
    from mathutils import Vector
    sz = vht.sollumz_bridge
    frag = build_vehicle()
    add_bone(frag, "wheel_lf", "chassis", (-0.9, 1.2, 0.3))
    add_bone(frag, "wheel_rf", "chassis", (0.9, 1.2, 0.3))
    add_model(frag, "wheel_lf", "wheel_lf")
    quick_fix(frag, "WHEEL_NO_COLLISION", "wheel_lf")
    disc = bpy.data.objects["wheel_lf.col"]
    assert sz.child_of_bone(disc).name == "wheel_lf" and frag.data.bones["wheel_lf"].sollumz_use_physics
    assert (disc.matrix_world.to_translation() - Vector((-0.9, 1.2, 0.3))).length < 1e-4 and disc.child_properties.mass > 0
    quick_fix(frag, "WHEEL_NO_COLLISION", "wheel_rf")
    copy = bpy.data.objects["wheel_rf.col"]
    assert sz.child_of_bone(copy).name == "wheel_rf" and frag.data.bones["wheel_rf"].sollumz_use_physics
    assert (copy.matrix_world.to_translation() - Vector((0.9, 1.2, 0.3))).length < 1e-4
    assert copy.data is not disc.data and copy.parent == disc.parent and copy.child_properties.mass == disc.child_properties.mass
    assert sz.sollum_type(copy) == sz.sollum_type(disc) and len(copy.constraints) == 1


@test
def t_fix_duplicate_materials_are_merged_without_touching_faces():
    frag = build_vehicle()
    keep = bpy.data.objects["body_m"].data.materials[0]
    twin = keep.copy()
    twin.shader_properties.index = 9
    boot = bpy.data.objects["boot_m"].data
    boot.materials.append(twin)
    for p in boot.polygons[:3]:
        p.material_index = 1
    faces = [p.material_index for p in boot.polygons]
    quick_fix(frag, "DUPLICATE_MATERIALS", allow_new=("MAT_INDEX_TIES",))
    assert twin not in list(boot.materials) and boot.materials[1] is keep
    assert [p.material_index for p in boot.polygons] == faces


@test
def t_fix_window_gets_a_glass_copy_of_its_material_and_nothing_else_changes():
    frag = build_vehicle()
    window, body = bpy.data.objects["window_m"].data, bpy.data.objects["body_m"].data
    plain = body.materials[0]
    window.materials[0] = plain
    quick_fix(frag, "HV040_GLASS_NO_SHADER", "window_lf", allow_new=("EMPTY_TEXTURE_SAMPLER", "HV062_GLASS_ORDER", "HV063_SHADER_ORDER"))
    glass = window.materials[0]
    assert glass is not plain and glass.shader_properties.name == "vehicle_vehglass"
    assert body.materials[0] is plain and plain.shader_properties.name != "vehicle_vehglass", "the shared material is left alone"
    order = analysis.export_order(analysis.load(frag))[0]
    assert len({m.shader_properties.index for m in order}) == len(order), "the new material has an index of its own"


@test
def t_fix_inner_glass_is_ordered_before_outer_glass():
    sz = vht.sollumz_bridge
    frag = build_vehicle()
    window = bpy.data.objects["window_m"]
    inner = sz.create_shader_material("vehicle_vehglass_inner.sps")
    for n in inner.node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.name != "DamageSampler":
            n.image = bpy.data.images.new("tex_inner", 1, 1)
            n.image.filepath = "//tex_inner.dds"
    inner.shader_properties.index = 7
    window.data.materials.append(inner)
    window.data.polygons[0].material_index = 1
    sz.prepare_mesh_for_material(window, inner)
    before = [m.name for m in analysis.export_order(analysis.load(frag))[0]]
    quick_fix(frag, "HV062_GLASS_ORDER", allow_new=("HV063_SHADER_ORDER",))
    after = analysis.export_order(analysis.load(frag))[0]
    names = [m.shader_properties.name for m in after]
    assert names.index("vehicle_vehglass_inner") < names.index("vehicle_vehglass")
    assert sorted(m.name for m in after) == sorted(before) and len({m.shader_properties.index for m in after}) == len(after)


@test
def t_fix_all_at_once_leaves_nothing_fixable_and_moves_nothing():
    """Several problems at once, fixed in one Apply as Fix All does: the fixes must not trip over each other."""
    from vehicle_hierarchy_tools import fixes
    sz = vht.sollumz_bridge
    frag = build_vehicle()
    add_model(frag, "bonnet", at=(0.3, 1.2, 0.9))                         # unbound, named after a bone
    bpy.data.objects["door_m"].parent = bpy.data.objects["body_m"]         # nested model
    bpy.data.objects["col_window"].parent = bpy.data.objects["testcar_drawable"]  # collision outside the composite
    frag.data.bones["door_dside_f"].sollumz_use_physics = False            # collision on a bone without physics
    props = frag.data.bones["bonnet"].bone_properties
    props.use_manual_tag, props.manual_tag = True, 1234                    # tag that does not match the name
    frag.data.bones["boot"].name = "boot.001"                             # Blender suffix
    fixes.apply(bpy.context, analysis.load(frag), [{"op": "reparent_bone", "bone": "window_lf", "parent": "bonnet"}])  # window outside its door
    for m in bpy.data.materials:
        if m.name.startswith("mat"):
            m.shader_properties.index = 0                                  # index ties
    bpy.context.view_layer.update()
    v = analysis.load(frag)
    before = analysis.validate(v)
    plan = fixes.dedupe([f.fix for f in before if f.fix and not fixes.describe(v, f.fix)[2]])
    assert len({fx["op"] for fx in plan}) >= 7, [fx["op"] for fx in plan]
    objs = {o.name: o.matrix_world.copy() for o in bpy.data.objects}
    rest = {b.name: b.matrix_local.copy() for b in frag.data.bones}
    fixes.apply(bpy.context, v, plan)
    bpy.context.view_layer.update()
    after = analysis.validate(analysis.load(frag))
    assert not [n for n, m in objs.items() if _differs(m, bpy.data.objects[n].matrix_world)], "nothing moves"
    assert not [b.name for b in frag.data.bones if _differs(rest.get(b.name, rest.get("boot.001")), b.matrix_local)]
    left = sorted({(f.code, f.target) for f in after if f.fix and not fixes.describe(analysis.load(frag), f.fix)[2]})
    assert not left, f"still fixable after one Apply: {left}"
    assert "boot" in frag.data.bones and sz.child_of_bone(bpy.data.objects["boot_m"]).name == "boot"
    assert frag.data.bones["window_lf"].parent.name == "door_dside_f"


@test
def t_move_bone_rename_part_and_every_axes_mode_move_nothing():
    from vehicle_hierarchy_tools import axes, fixes
    sz = vht.sollumz_bridge
    frag = build_vehicle()

    def still(label, objs, rest=None):
        bpy.context.view_layer.update()
        moved = [n for n, m in objs.items() if n in bpy.data.objects and _differs(m, bpy.data.objects[n].matrix_world)]
        assert not moved, f"{label}: objects moved: {moved}"
        if rest is not None:
            turned = [b.name for b in frag.data.bones if b.name in rest and _differs(rest[b.name], b.matrix_local)]
            assert not turned, f"{label}: bone rest pose changed: {turned}"

    snap = lambda: ({o.name: o.matrix_world.copy() for o in bpy.data.objects}, {b.name: b.matrix_local.copy() for b in frag.data.bones})
    bound = lambda: {o.name: sz.child_of_bone(o).name for o in bpy.data.objects if sz.child_of_bone(o) is not None}

    # move a bone among its siblings: order changes, everything else stays
    objs, rest = snap()
    links, order = bound(), [b.name for b in frag.data.bones]
    tags = {b.name: b.bone_properties.tag for b in frag.data.bones}
    limits = {pb.name: [(c.type, round(c.max_x, 4)) for c in pb.constraints] for pb in frag.pose.bones}
    physics = {b.name: b.sollumz_use_physics for b in frag.data.bones}
    report = fixes.apply(bpy.context, analysis.load(frag), [{"op": "move_bone", "bone": "bonnet", "delta": 1}])
    still("move_bone", objs, rest)
    now = [b.name for b in frag.data.bones]
    assert now != order and sorted(now) == sorted(order), report
    assert now.index("boot") < now.index("bonnet"), now
    assert bound() == links and {b.name: b.bone_properties.tag for b in frag.data.bones} == tags
    assert {pb.name: [(c.type, round(c.max_x, 4)) for c in pb.constraints] for pb in frag.pose.bones} == limits
    assert {b.name: b.sollumz_use_physics for b in frag.data.bones} == physics

    # rename a part: the bone, the objects named after it and their meshes follow, bindings stay
    bpy.data.objects["bonnet_m"].name = "bonnet"
    extra = add_model(frag, "bonnet.001", "bonnet")
    objs, rest = snap()
    lines = fixes.apply(bpy.context, analysis.load(frag), [{"op": "rename_part", "bone": "bonnet", "to": "boot_r"}])
    assert "boot_r" in frag.data.bones and "bonnet" not in frag.data.bones, lines
    assert "boot_r" in bpy.data.objects and extra.name == "boot_r.001", [o.name for o in bpy.data.objects]
    assert sz.child_of_bone(bpy.data.objects["boot_r"]).name == "boot_r" and sz.child_of_bone(extra).name == "boot_r"
    assert frag.data.bones["boot_r"].bone_properties.tag == sz.tag_hash("boot_r")
    still("rename_part", {("boot_r" + n[6:] if n.startswith("bonnet") else n): m for n, m in objs.items()})

    # every Local Axes mode: the axes change, nothing you can see moves
    bpy.context.scene.cursor.location = (0.2, 0.4, 0.6)
    for mode, source in (("CENTER", None), ("PARENT", None), ("WORLD", None), ("CURSOR", None), ("COPY", "boot")):
        for position in (False, True):
            objs, rest = snap()
            lines, ok = axes.set_axes(bpy.context, frag, "door_dside_f", mode, source, (), True, position)
            assert ok, (mode, position, lines)
            still(f"axes {mode}", objs, {n: m for n, m in rest.items() if n != "door_dside_f"})


@test
def t_every_panel_draws_with_hints_on_and_off():
    """The panels cannot be shown without a window, so each one is drawn onto a stand-in layout: a mistake in a draw method fails here."""
    import types
    from unittest import mock
    from vehicle_hierarchy_tools import ui
    frag = build_vehicle()
    bpy.context.view_layer.objects.active = frag
    st = bpy.context.window_manager.vht
    panels = [c for c in ui.classes if issubclass(c, bpy.types.Panel)]
    parents = {c.__name__ for c in panels}
    assert all(getattr(c, "bl_parent_id", "") in parents | {""} for c in panels), "every sub-panel names a registered parent"
    top = [c.bl_label for c in panels if not getattr(c, "bl_parent_id", "")]
    assert top == ["Check", "Issues", "Vehicle", "Generate LODs", "Materials", "Export Info"], top
    closed = [c.bl_label for c in panels if getattr(c, "bl_parent_id", "") and "DEFAULT_CLOSED" not in getattr(c, "bl_options", set())]
    assert not closed, f"sub-panels start closed: {closed}"
    assert bpy.ops.vht.validate() == {"FINISHED"} and bpy.ops.vht.inspect() == {"FINISHED"} and bpy.ops.vht.suggest() == {"FINISHED"}
    drawn = 0
    for hints in (False, True):
        st.show_hints = hints
        for c in panels:
            if hasattr(c, "poll") and not c.poll(bpy.context):
                continue
            self = types.SimpleNamespace(layout=mock.MagicMock(), icon=getattr(c, "icon", "NONE"), icon_name=getattr(c, "icon_name", None),
                                         bl_parent_id=getattr(c, "bl_parent_id", ""))
            c.draw(self, bpy.context)
            for extra in ("draw_header", "draw_header_preset"):
                if hasattr(c, extra):
                    getattr(c, extra)(self, bpy.context)
            drawn += 1
    assert drawn >= 2 * 15, drawn
    st.show_hints = False


@test
def t_grab_and_drop_follow_the_part_selected_in_the_viewport():
    frag = build_vehicle()
    st = bpy.context.window_manager.vht

    def pick(name):  # as clicking the object in the viewport does
        for o in bpy.context.selected_objects:
            o.select_set(False)
        bpy.data.objects[name].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[name]

    bpy.context.view_layer.objects.active = frag
    assert bpy.ops.vht.inspect() == {"FINISHED"}
    st.tree_index = next(i for i, r in enumerate(st.tree) if r.kind == "node" and r.bone == "bonnet")  # a row clicked earlier
    pick("boot_m")
    assert bpy.ops.vht.tree_grab() == {"FINISHED"}
    assert st.tree_grab == "node:boot", f"the selected part is grabbed, not the row clicked before: {st.tree_grab}"
    pick("door_m")
    assert bpy.ops.vht.tree_drop(mode="INTO") == {"FINISHED"}
    assert frag.data.bones["boot"].parent.name == "door_dside_f" and st.tree_grab == ""
    # with no new selection, the row clicked in the list is what gets grabbed (a [COL] row grabs that collision)
    st.tree_index = next(i for i, r in enumerate(st.tree) if r.kind == "col")
    assert bpy.ops.vht.tree_grab() == {"FINISHED"} and st.tree_grab.startswith("col:"), st.tree_grab
    assert bpy.ops.vht.tree_grab(cancel=True) == {"FINISHED"} and st.tree_grab == ""


fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} passed")
sys.exit(1 if fails else 0)
