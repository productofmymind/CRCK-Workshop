"""All access to Sollumz lives here.

Sollumz has no public Python API, so everything below touches its internals. Each accessor
names what it depends on; when a Sollumz update breaks one, fix it here and nowhere else.
Written against Sollumz 2.9.0; Sollumz 2.8 differences are handled where they occur (marked "2.8").
Nothing is imported at module load, so this add-on still registers when Sollumz is missing.
"""
import importlib
import inspect
import os
import re
import sys

import bpy

TESTED_VERSIONS = ("2.9",)  # major.minor prefixes verified by the test suite

FRAGMENT = "sollumz_fragment"
DRAWABLE = "sollumz_drawable"
DRAWABLE_MODEL = "sollumz_drawable_model"
BOUND_COMPOSITE = "sollumz_bound_composite"
SHADER_MATERIAL = "sollumz_material_shader"


def package():
    """Module name Sollumz is registered under, or None. An extension is `bl_ext.<repo>.sollumz`; a legacy add-on (Blender 4.0/4.1)
    is named after its folder, which depends on how it was unzipped, so it is recognised by its bl_info name."""
    for key in bpy.context.preferences.addons.keys():
        mod = sys.modules.get(key)
        if mod is None:
            continue
        if key.rsplit(".", 1)[-1].lower() == "sollumz" or getattr(mod, "bl_info", {}).get("name") == "Sollumz":
            return key
    return None


_VERSION = {}  # package name -> version string; it does not change while Blender runs, and every panel asks on every redraw
_MANIFEST_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)


def version():
    pkg = package()
    if pkg is None:
        return None
    if pkg not in _VERSION:
        mod = sys.modules[pkg]
        info = getattr(mod, "bl_info", None)  # legacy add-on; Blender deletes bl_info from extensions
        if info and info.get("version"):
            _VERSION[pkg] = ".".join(map(str, info["version"]))
        else:
            # Read with a regex: tomllib is not in Blender 4.0's Python 3.10.
            try:
                with open(os.path.join(os.path.dirname(mod.__file__), "blender_manifest.toml"), encoding="utf-8") as f:
                    m = _MANIFEST_VERSION.search(f.read())
            except OSError:
                return "unknown"  # not cached: the file may be readable next time
            _VERSION[pkg] = m.group(1) if m else "unknown"
    return _VERSION[pkg]


def status():
    """(usable, message) for the UI."""
    pkg = package()
    if pkg is None:
        return False, "Sollumz is not installed or not enabled."
    ver = version()
    try:
        _mod("sollumz_helper").get_sollumz_materials
        _mod("tools.blenderhelper").get_child_of_bone
        _mod("ydr.properties").BoneProperties.calc_tag_hash
    except (ImportError, AttributeError) as e:
        return False, f"Sollumz {ver} is incompatible: {e}"
    if not any(str(ver).startswith(v) for v in TESTED_VERSIONS):
        return True, f"Sollumz {ver} (untested; tested with {', '.join(TESTED_VERSIONS)})"
    return True, f"Sollumz {ver}"


def _mod(name):
    return importlib.import_module(f"{package()}.{name}")


def ydrexport():
    """Sollumz's drawable exporter module (ydr/ydrexport.py): where `dedupe_and_get_indices` lives, which compactexport swaps while it exports."""
    return _mod("ydr.ydrexport")


def sollum_type(id_data):
    return getattr(id_data, "sollum_type", "")


def is_shader_material(mat):
    return mat is not None and sollum_type(mat) == SHADER_MATERIAL


def find_fragment(obj):
    """Walk up from obj to its Sollumz fragment armature."""
    while obj is not None:
        if sollum_type(obj) == FRAGMENT:
            return obj
        obj = obj.parent
    return None


def export_materials(frag, hi=False):
    """Materials in the exact order Sollumz will write the shader group.

    Depends on sollumz_helper.get_sollumz_materials(obj, mode): first-encounter order over
    frag.children_recursive and LOD meshes, then a stable sort by shader_properties.index.
    """
    helper = _mod("sollumz_helper")
    mode = helper.GetSollumzMaterialsMode.HI if hi else helper.GetSollumzMaterialsMode.BASE
    return helper.get_sollumz_materials(frag, mode)


def export_model_objs(drawable, frag):
    """Drawable Models in the order Sollumz's exporter will process them (ydr.ydrexport.get_model_objs, then
    sort_skinned_models_by_bone). Before Sollumz merges skinned parts and splits by vertex count."""
    ex = _mod("ydr.ydrexport")
    return ex.sort_skinned_models_by_bone(ex.get_model_objs(drawable), frag.data.bones)


def child_of_bone(obj):
    """Bone an object is bound to with Sollumz's bone constraint (Copy Transforms), or None."""
    return _mod("tools.blenderhelper").get_child_of_bone(obj)


def bind(obj, frag, bone_name):
    """Add Sollumz's own bone constraint (Copy Transforms, tools.blenderhelper.add_child_of_bone_constraint)."""
    return _mod("tools.blenderhelper").add_child_of_bone_constraint(obj, frag, bone_name)


def unbind(obj):
    """Remove the constraint get_child_of_bone() reads, if any (tools.blenderhelper.get_child_of_constraint)."""
    c = _mod("tools.blenderhelper").get_child_of_constraint(obj)
    if c is not None:
        obj.constraints.remove(c)


def create_shader_material(filename, in_place=None):
    """New Sollumz shader material (ydr.shader_materials.create_shader), or reset `in_place` to that shader."""
    return _mod("ydr.shader_materials").create_shader(filename, in_place_material=in_place)


def switch_shader(mat, filename):
    """Change a material's shader keeping the parameter values and textures both shaders share
    (what Sollumz's Change Shader operator does: ydr.gta5.presets.shader capture/apply; 2.8: ydr.operators.materials
    shader_preset_from_material / shader_preset_apply_to_material)."""
    images = {n.name: n.image for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image is not None}
    try:
        presets = _mod("ydr.gta5.presets.shader")
        capture, apply = presets.shader_preset_capture_dict, presets.shader_preset_apply_dict
    except ImportError:  # 2.8
        ops = _mod("ydr.operators.materials")
        capture, apply = ops.shader_preset_from_material, ops.shader_preset_apply_to_material
    tmp = capture(mat)
    create_shader_material(filename, in_place=mat)
    apply(mat, tmp, apply_textures=True)
    for n in mat.node_tree.nodes:  # Sollumz restores textures by image name; keep the very same images for shared samplers
        if n.type == "TEX_IMAGE" and n.name in images:
            n.image = images[n.name]
    _mod("ydr.operators.materials").post_create_shader_add_default_images(mat)


def _new_sollumz_object(sollum_type, name, mesh, collection):
    """tools.blenderhelper.create_blender_object, linked only to `collection`. 2.8 has no link_to_context_collection
    and always links to the active collection, so the object is moved afterwards."""
    create = _mod("tools.blenderhelper").create_blender_object
    if "link_to_context_collection" in inspect.signature(create).parameters:
        obj = create(sollum_type, name=name, object_data=mesh, link_to_context_collection=False)
    else:
        obj = create(sollum_type, name=name, object_data=mesh)
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def new_bound_geometry(name, mesh, collection):
    """A Bound Geometry object (what Sollumz's own bound creation makes) linked to `collection`."""
    return _new_sollumz_object(_mod("sollumz_properties").SollumType.BOUND_GEOMETRY, name, mesh, collection)


def new_bound_disc(name, radius, length, collection):
    """A Bound Disc (axle along X, as Sollumz makes and exports it) with a DEFAULT collision material, linked to `collection`."""
    me = _mod("tools.meshhelper")
    mesh = _bpy_data_meshes_new(name)
    me.create_disc(mesh, radius, length)
    mesh.materials.append(collision_material("DEFAULT"))
    return _new_sollumz_object(_mod("sollumz_properties").SollumType.BOUND_DISC, name, mesh, collection)


def _bpy_data_meshes_new(name):
    import bpy
    return bpy.data.meshes.new(name)


def collision_material(name):
    """New Sollumz collision material by its name (ybn.collision_materials), e.g. CAR_GLASS_STRONG."""
    mats = _mod("ybn.collision_materials")
    return mats.create_collision_material_from_index(next(i for i, m in enumerate(mats.collisionmats) if m.name == name))


def prepare_mesh_for_material(obj, mat):
    """UV maps and colour attributes the material's shader needs (ydr.operators.materials.post_create_shader_update_object)."""
    _mod("ydr.operators.materials").post_create_shader_update_object(obj, mat)


def selected_bones(frag):
    """Names of the bones selected in Pose mode. Blender 5.x keeps selection on PoseBone, 4.2 on Bone."""
    return [b.name for b in frag.data.bones
            if (frag.pose.bones[b.name].select if hasattr(frag.pose.bones[b.name], "select") else b.select)]


def tag_hash(bone_name):
    """Sollumz bone tag: ElfHash(upper(name)) % 0xFE8F + 0x170."""
    return _mod("ydr.properties").BoneProperties.calc_tag_hash(bone_name)


LOD_ATTRS = ("very_high", "high", "medium", "low", "very_low")  # Object.sz_lods.<attr>.mesh


def lod_meshes(obj):
    """(lod_attr, mesh) for every LOD mesh of a Drawable Model, highest first."""
    lods = getattr(obj, "sz_lods", None)
    if lods is None:
        return []
    out = []
    for attr in LOD_ATTRS:
        mesh = getattr(lods, attr).mesh
        if mesh is not None:
            out.append((attr, mesh))
    return out


def _norm(v):
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    return tuple(round(float(x), 6) for x in v)  # mathutils.Vector


def _param_value(p):
    """A shader parameter's value: 2.9 ShaderInst parameters have .value; 2.8 Shader parameters keep a texture name,
    x/y/z/w or a list of vectors."""
    if hasattr(p, "value"):
        return _norm(p.value)
    if hasattr(p, "texture_name"):
        return p.texture_name
    if hasattr(p, "values"):
        return tuple(_norm(v) for v in p.values)
    return _norm(tuple(getattr(p, c, 0.0) for c in "xyzw"))


def shader_signature(mat):
    """What Sollumz would write for this material (ydr.ydrexport.create_shader; 2.8: get_shaders_from_blender): shader,
    bucket, every parameter value and texture name. Two materials with the same signature export identical shaders."""
    ex = _mod("ydr.ydrexport")
    if hasattr(ex, "create_shader"):
        inst = ex.create_shader(mat)
        filename = inst.preset_filename
    else:  # 2.8
        inst = ex.get_shaders_from_blender([mat])[0]
        filename = inst.filename
    return (inst.name, filename, str(inst.render_bucket),
            tuple((p.name, _param_value(p)) for p in inst.parameters))


def mesh_missing_attrs(mesh):
    """(missing UV map names, missing/incorrect colour attribute names) for a mesh, using the same helpers as the
    Sollumz Data-tab warnings (tools.meshhelper)."""
    mh = _mod("tools.meshhelper")
    uvs = [mh.get_uv_map_name(i) for i in sorted(mh.get_mesh_used_texcoords_indices(mesh))]
    cols = [mh.get_color_attr_name(i) for i in sorted(mh.get_mesh_used_colors_indices(mesh))]
    miss_uv = [n for n in uvs if n not in mesh.uv_layers]
    miss_col = [n for n in cols if n not in mesh.color_attributes
                or mesh.color_attributes[n].domain != "CORNER" or mesh.color_attributes[n].data_type != "BYTE_COLOR"]
    return miss_uv, miss_col


def has_joint_limit(frag, bone_name):
    """True when Sollumz exports the bone with a rotation or translation limit (an articulated link):
    ydr.ydrexport.find_bone_constraint_rotation_limit / find_bone_constraint_translation_limit
    (2.8: get_limit_rot_constraint / get_limit_pos_constraint)."""
    pb = frag.pose.bones.get(bone_name)
    if pb is None:
        return False
    ex = _mod("ydr.ydrexport")
    rot = getattr(ex, "find_bone_constraint_rotation_limit", None) or ex.get_limit_rot_constraint
    pos = getattr(ex, "find_bone_constraint_translation_limit", None) or ex.get_limit_pos_constraint
    return rot(pb) is not None or pos(pb) is not None
