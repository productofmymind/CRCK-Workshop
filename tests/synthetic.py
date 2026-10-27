"""Synthetic GTA V vehicle for tests: a Sollumz fragment with bones, bound models and collisions.
Built with Sollumz's own helpers, so no game assets are needed. Import after Sollumz is enabled."""
import importlib

import bpy
from mathutils import Vector

from vehicle_hierarchy_tools import analysis, sollumz_bridge as sz


def sollumz(name):
    return importlib.import_module(f"{sz.package()}.{name}")


BONES = {  # name: (parent, head)
    "chassis": (None, (0, 0, 0.5)),
    "door_dside_f": ("chassis", (0.9, 0.5, 0.5)),
    "window_lf": ("door_dside_f", (0.9, 0.5, 0.9)),
    "handle_dside_f": ("door_dside_f", (0.9, 0.6, 0.6)),
    "bonnet": ("chassis", (0, 1.5, 0.7)),
    "boot": ("chassis", (0, -1.5, 0.7)),
}
MODELS = {"body_m": "chassis", "door_m": "door_dside_f", "window_m": "window_lf", "handle_m": "door_dside_f",
          "bonnet_m": "bonnet", "boot_m": "boot"}
BOUNDS = {"col_chassis": "chassis", "col_door": "door_dside_f", "col_window": "window_lf"}
JOINT_BONES = ("door_dside_f", "bonnet", "boot")  # opening parts get a rotation limit, as Sollumz imports them


def cube(name):
    me = bpy.data.meshes.new(name)
    s = 0.2
    v = [(x * s, y * s, z * s) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    me.from_pydata(v, [], f)
    me.update()
    mh = sollumz("tools.meshhelper")
    for i in (0, 1, 2):  # whatever the default shader uses; extra maps are harmless
        me.uv_layers.new(name=mh.get_uv_map_name(i))
        me.color_attributes.new(name=mh.get_color_attr_name(i), type="BYTE_COLOR", domain="CORNER")
    return me


def build_vehicle(reset=True):
    if reset:
        bpy.ops.wm.read_homefile(use_empty=True)
    bh, sp = sollumz("tools.blenderhelper"), sollumz("sollumz_properties")
    shaders = sollumz("ydr.shader_materials")
    frag = bh.create_blender_object(sp.SollumType.FRAGMENT, "testcar", bpy.data.armatures.new("skel"))
    drawable = bh.create_empty_object(sp.SollumType.DRAWABLE, "testcar_drawable")
    composite = bh.create_empty_object(sp.SollumType.BOUND_COMPOSITE, "testcar_composite")
    drawable.parent = composite.parent = frag
    bpy.context.view_layer.objects.active = frag
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (parent, head) in BONES.items():
        eb = frag.data.edit_bones.new(name)
        eb.head, eb.tail = Vector(head), Vector(head) + Vector((0, 0, 0.1))
        if parent:
            eb.parent = frag.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    mats = [shaders.create_shader(n) for n in ("default.sps", "default.sps", "vehicle_vehglass.sps")]
    for i, m in enumerate(mats):
        m.name = f"mat{i}"
        m.shader_properties.index = i
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.name.endswith("Sampler") and n.name != "DamageSampler":
                n.image = bpy.data.images.new("tex", 1, 1)
                n.image.filepath = f"//tex{i}.dds"  # name comes from the file path; distinct per material
    for name, bone in MODELS.items():
        me = cube(name)
        me.materials.append(mats[list(MODELS).index(name) % 3])
        o = bh.create_blender_object(sp.SollumType.DRAWABLE_MODEL, name, me)
        o.sz_lods.get_lod(sp.LODLevel.HIGH).mesh = me
        o.parent = drawable
        o.location = frag.data.bones[bone].head_local
        bh.add_child_of_bone_constraint(o, frag, bone)
        o.location = (0.0, 0.0, 0.0)
    for name, bone in BOUNDS.items():
        o = bh.create_blender_object(sp.SollumType.BOUND_BOX, name, cube(name))
        o.parent = composite
        bh.add_child_of_bone_constraint(o, frag, bone)
        frag.data.bones[bone].sollumz_use_physics = True
        o.child_properties.mass = 10.0
    for name in JOINT_BONES:
        c = frag.pose.bones[name].constraints.new("LIMIT_ROTATION")
        c.owner_space = "LOCAL"
        c.use_limit_x = True
        c.min_x, c.max_x = 0.0, 1.2
    bpy.context.view_layer.update()
    bpy.context.view_layer.objects.active = frag
    return frag
