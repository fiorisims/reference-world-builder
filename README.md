# Reference World Builder

Turns a concept painting, photo or Photoshop file into a small, walkable
Unreal Engine 5.8 environment: it recognises the objects in the picture, builds
3D models of them, and lays out terrain, water, streets and buildings so that
the world lines up with the picture from the start position.

The full guide (panel, settings, how each step works) is in
[`ReferenceWorldBuilder/README.md`](ReferenceWorldBuilder/README.md).

## Layout

Keep these two parts side by side: the Unreal project finds the helper
scripts one folder above itself.

- `ReferenceWorldBuilder/` - the Unreal project (`.uproject`, the panel and
  `Content/Python/` tool code).
- `reference_*.py`, `blender_refine.py`, ... - helpers run by system Python
  (image analysis, vision models, mesh building) and by Blender.
- `ModelLibrary/` - drop your own model packs here (see its README).

## Setup

1. Unreal Engine 5.8 (no C++ or Visual Studio needed).
2. Python 3 on PATH, then: `python -m pip install -r requirements.txt`
3. Optional: Blender (refines the generated models) and Photoshop (only for
   `.psd` / `.psb` pictures).
4. Open `ReferenceWorldBuilder/ReferenceWorldBuilder.uproject`. The panel opens
   by itself; Browse for a picture and press Analyze.

The first Analyze downloads the vision models from Hugging Face.

## Tests

```
python test_reference_world.py          # offline checks, no artwork needed
python test_reference_world.py --live   # also the real vision models and Blender
```
