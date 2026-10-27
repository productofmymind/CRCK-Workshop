# CRCK Workshop

**CRCK Workshop** is a Blender add-on for GTA V vehicle modding. It sits **on top of [Sollumz](https://github.com/Sollumz/Sollumz)** and adds what a vehicle workflow is missing: a tree view of the vehicle's parts, one-click checks with fixes, whole-vehicle LOD generation, paint and material tools, and a smaller-file export. It replaces nothing Sollumz does.

**Version 0.17.1** · Blender 4.0+ (tested on 5.2.3) · Sollumz 2.8+ (tested with 2.9) · Windows, macOS and Linux · GPL-3.0-or-later

## What it does, in short

| You want to... | CRCK Workshop gives you |
|---|---|
| See how a vehicle is built | A tree of parts: bones, the models bound to them, collisions and LOD slots, in one view |
| Find out what is wrong | **Check Vehicle**: each problem explained in plain words, with a fix you can preview and apply |
| Make LODs | **Generate LODs** for the whole vehicle at once (or a model on its own), starting from L0 (`_hi`) or L1 |
| Paint and materials | Paint layers and liveries, vehicle shaders, default textures, material usage, material order |
| Ship a smaller file | **CRCK Workshop Export**: Sollumz's own exporter with fine rounding, a vertex merge and cache-friendly ordering that make files about half the size |
| Fix a wobbling rotor, rename a part, rearrange bones | Local-axes tools, Rename Part, Auto Arrange |

New here? Read the step-by-step [tutorial](TUTORIAL.md).

## Install

1. Download the latest `CRCK-Workshop-<version>.zip` from the [Releases](../../releases) page. (Or zip the `vehicle_hierarchy_tools` folder itself, so the zip holds that one folder.)
2. **Blender 4.2 and newer:** Edit > Preferences > Get Extensions > the arrow menu > **Install from Disk**, pick the zip, and enable it.
   **Blender 4.0 and 4.1:** Edit > Preferences > Add-ons > **Install...**, pick the same zip, and tick CRCK Workshop.
   **Sollumz must be enabled too.**
3. Select the vehicle's Fragment (or any object inside it), or a Drawable Model on its own. Everything is in one tab of the 3D Viewport sidebar (press **N**): **CRCK Workshop**.
4. Restart Blender after updating the add-on.

CRCK Workshop is pure Python: nothing to compile and no bundled binaries, so it works the same on every platform Blender runs on.

## Why it exists

A GTA V vehicle has no object tree. Its hierarchy is a **bone tree** plus the **objects bound to each bone**. Sollumz shows flat objects and a separate armature. CRCK Workshop works on the real structure and shows it as a tree of nested parts, so you can see and change how the vehicle is put together.

## Features

### See and fix the structure

| Tool | What it does |
|---|---|
| **Hierarchy** | One tree: fragment root, parts nested in bone order, `[COL]` rows, the LOD slots each part has. Select, expand, hide/isolate, show L0-L4 or COL, Move Up/Down, attach objects or bones to a part. **Grab and drop** moves a part or collision into another part, before/after a sibling, or unlinks it. Search filters rows. |
| **Auto Arrange** | One preview of the moves that put parts where vanilla vehicles have them: bones under their usual parents, models and collisions on the bone they are named after, nested models back under the Drawable, collisions into the Bound Composite. Nothing moves in space. |
| **Local Axes** | Center to Object, Reset to Parent/World, Copy From Grabbed, Origin to 3D Cursor (rotation and position switches) and **Align Spin Axis to Geometry**, all without moving geometry, children or collisions. Fixes rotor wobble. |
| **Parts and Bone Order** | Bind several objects to one bone, or several bones under one (preview, then Apply, one undo step). Move bones up/down among siblings. **Rename Part** renames bone, models, collisions, mesh data and vertex groups together, with find/replace. Part collections mirror the bone tree in the Outliner. |
| **Problems** | **Check Vehicle** lists each problem in plain words: what is wrong (with your names), why it matters, how to fix it by hand, and a previewable **quick fix** where one exists. 51 kinds explained, including **wheel bones without a collision** (the quick fix copies another wheel's collision, or builds a disc from the wheel model). Rules are data (`data/vehicle_rules.json`), graded by confidence, and silent on the vanilla vehicles tested. Some checks may still be wrong, so read each one before you apply its fix. If you find one that is wrong, please report it in the Issues tab. |
| **Export Info** | The order Sollumz will actually write, and a size estimate from a trial export (warns at 256 MiB). **CRCK Workshop Export** (also under File > Export) runs Sollumz's exporter with your settings but rounds normals, tangents, positions and UVs to a very fine grid (0.05 degrees, 0.06 mm), merges equal vertices and reorders triangles for the GPU's vertex cache (Tipsify; a mesh another tool already ordered keeps its order when that compresses smaller): a test vehicle re-exported 51% smaller than with Sollumz alone (3.73 MB to 1.83 MB, measured with the earlier meshoptimizer ordering, which the new one matched within 0.1% on the vehicle it was compared on), with the same triangles. |

### Materials

| Tool | What it does |
|---|---|
| **Paint layers and liveries** | A colour variant is a material with another paint layer (Primary, Secondary, Pearlescent, Wheel, ...), as vanilla does it. Add paint materials from any paint shader, make livery materials. |
| **Default textures** | New materials get the texture names vanilla uses per shader and sampler; *Fill Empty Textures* does it for existing ones. |
| **Vehicle Shaders** | Every `vehicle_*` shader in one dropdown. *Create Material* adds a new material slot to the selected models (nothing is replaced) with the shared textures vanilla vehicles use for that shader already attached; samplers meant for your own textures stay empty. |
| **Material Usage** | Which materials the vehicle uses, by how many models, polygons and LODs; Select Users; Replace Everywhere; texture report. |
| **Material Order** | Reorder the shader group. Writes Sollumz's own shader index. **Sort Like Vanilla** orders it the way vanilla vehicles do (paint first, inner glass, shutters, lights ... outer glass, interior, tyres last); a Problems entry offers it as a quick fix. A tendency learned from 11 vehicles, not a proven game rule. |

### Generate LODs

Fills the LOD slots below a start level you choose, **L0 (`_hi`) or L1**, for every part, the selected models, or a Drawable Model that is not in a vehicle yet.

- Works in Sollumz's own LOD slots: no loose objects, start meshes never touched, existing slots left alone unless *Replace existing* is on. Models with no start mesh are skipped with a reason.
- CRCK Workshop' own simplifier (`simplify.py`, quadric error edge collapse after Garland and Heckbert). Game meshes are split into pieces at every UV seam and hard edge; the simplifier works on them joined, so **no cracks open at seams**. Open edges and material borders stay as lines, UV, colour and normal seams are kept unless crossing one is cheap, **no material disappears**, and moves that would fold a triangle or make a needle are refused, so the triangles stay even.
- Keeps every UV map and colour attribute, bone weights and the original shading (each corner keeps an original normal of its own side, so hard edges stay hard).
- Wheels, glass, lights and mirrors keep extra triangles (default 1.5x). Defaults are the triangle shares 11 vanilla vehicles use. Blender's Decimate is still there as an option.
- Measured on a modded vehicle (104,675 triangles at L0) against the previous version: no open seams at any level (the previous one had five times the source's open edge length), worst-case surface error at L3 94 mm instead of 154 mm, average error at L1 0.9 mm instead of 1.0 mm. About 20 seconds for the whole vehicle. LOD switch distances are not changed.

## How it works with Sollumz

- **Sollumz stays the foundation.** Import, export, shaders, collisions, physics and the data model are Sollumz's. CRCK Workshop never edits its files or replaces its importer or exporter.
- **It edits only data Sollumz already reads:** Sollumz's own bone constraint, `Material.shader_properties.index`, material slots, bone parents and order, LOD slots.
- **Checks use Sollumz's own functions**, so what is reported matches what the exporter does.
- **One file touches Sollumz internals:** `vehicle_hierarchy_tools/sollumz_bridge.py`. If a Sollumz update moves something, that is the only file to fix. A missing function shows "incompatible" in the panel instead of failing; other Sollumz versions show an "untested" note.
- **Nothing is changed silently.** Consequential edits are previewed, applied in one operator, and are one undo step. Read-only checks are tested not to modify anything.

## What Sollumz does not do (so CRCK Workshop does)

Verified against Sollumz 2.9.0's source and by experiment: show a vehicle as nested parts (models are flat siblings, and parenting one under an object makes the exporter drop it); bind objects to a chosen bone in bulk; reorder sibling bones; warn about name-ordered rigid models, `.001` bone names, tag mismatches, duplicate materials, empty samplers or missing UV maps and colour attributes across a vehicle; estimate export size; generate LODs for a whole vehicle at once.
Import/export, shader conversion, collision creation, mass, light IDs, cloth and texture dictionaries stay Sollumz's.

## Limits

- Generated LODs are judged by surface distance, normals and UVs against the start mesh, not rendered or run in game. Set LOD distances yourself.
- The size estimate is not checked against the resource header.
- Blender 4.0/4.1 and Sollumz 2.8 are supported from their source and by tests that stand in for their differences, but have not been run in those versions yet. Tested on Blender 5.2.3 with Sollumz 2.9.
- At the lowest levels (L3, L4) a part with a lot of fine detail in a small area (a spoiler lip on a boot lid) can lose that detail, because every part gets the same share of its triangles. Look over L3 and L4 before shipping.
- Performance on very large vehicles is untested.

## Credits

This project builds on the work of others, and would not exist without it:

- **[Sollumz](https://github.com/Sollumz/Sollumz)** by the Sollumz team (GPL-3.0): the import and export of every GTA V asset type, the shader and collision data model, and LOD slots that
  CRCK Workshop works through. It reads Sollumz's source for compatibility and never modifies it. Sollumz in turn relies on its own file libraries for reading and writing the native files.
- **[meshoptimizer](https://github.com/zeux/meshoptimizer)** by Arseny Kapoulkine (MIT): versions up to 0.16.1 used it for LODs and vertex-cache ordering; it set the bar the own simplifier and ordering were measured against.
- **[Blender](https://www.blender.org)** (GPL): the platform, the Decimate modifier used as the fallback reducer, and the BVH used to measure and transfer surfaces.
- **Published research** on mesh simplification: M. Garland and P. Heckbert, *Surface simplification using quadric error metrics* (SIGGRAPH 1997) and *Simplifying surfaces with color
  and texture using quadric error metrics* (1998); H. Hoppe, *New quadric metric for simplifying meshes with appearance attributes* (1999); P. Cignoni et al., *A general method for
  preserving attribute values on simplified meshes* (1998); P. Sander, D. Nehab and J. Barczak, *Fast triangle reordering for vertex locality and reduced
  overdraw* (SIGGRAPH 2007), the Tipsify ordering CRCK Workshop Export uses.
- **The GTA V modding community's** vehicle hierarchy references (bone names, parents, vehicle types), which the rule data started from and which were then checked against real vehicles.
- **Rockstar Games' vanilla vehicle files** were read only to measure what a vehicle normally looks like (parents, binding, shader order, textures, LOD shares). None are included in this repository.

GTA V is a trademark of Take-Two Interactive. This project is not affiliated with Rockstar Games, Take-Two or the Sollumz team.

## Thanks to Sollumz

CRCK Workshop only exists because of [Sollumz](https://github.com/Sollumz/Sollumz). It does the importing, exporting and the whole data model this add-on stands on, and without it GTA V modding in Blender would still be a pain. Thanks to the Sollumz team and its code contributors.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

## AI disclaimer

AI tools were used to help build this add-on, including most of the code, the tests and the documentation. I directed the work and tested it on my own vehicles, in Blender and in game. It is provided as is: test it on copies of your files, and if you find a problem or have a suggestion, open an issue.
