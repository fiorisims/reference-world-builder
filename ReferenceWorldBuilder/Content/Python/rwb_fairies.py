"""Fairies: small coloured lights with wings that float around the world, instead of solid models.

A fairy is a glowing core, two crossed cards of soft wings (so the wings show from every side) and a
coloured point light, all moving together along a slow closed loop (InterpToMovementComponent) when
you press Play.  Painted fairies keep their spot and colour; a few more float around the rest of the
world in the painting's fairy colours.

The pure helpers (wing texture, colours, float path) don't need Unreal and are tested offline.
"""
import math
import struct
import zlib
from pathlib import Path

try:
    import unreal
except ImportError:            # offline tests
    unreal = None

FAIRY_CATEGORIES = {"fairy"}
FAIRY_MATERIAL = "M_RWB_FairyGlow_v1"
CORE_CM = 9.0                  # the glowing body
WING_CM = 42.0                 # wingspan
CORE_GLOW = 14.0
WING_GLOW = 3.5
LIGHT_CANDELA = 8.0
LIGHT_RADIUS = 500.0
EXTRA_FAIRIES = 8              # more fairies floating around the world, outside the painted ones
FLOAT_REACH = (70.0, 180.0)    # cm: how far a fairy wanders from its spot
FLOAT_RISE = (25.0, 70.0)      # cm: how much it bobs up and down
FLOAT_SECONDS = (8.0, 15.0)    # one loop
HOVER = (110.0, 320.0)         # cm above the ground for the extra fairies
DEFAULT_COLOURS = ["#C77DFF", "#FF8AD8", "#F4F1FF", "#7FD6FF"]
WING_TEXTURE = "fairy_wings.png"


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def wing_pixels(size=128):
    """RGBA bytes of a pair of soft white wings (upper and lower lobes each side), size x size."""
    lobes = []
    for side in (-1.0, 1.0):
        lobes.append((0.5 + side * 0.24, 0.38, 0.22, 0.16, side * 0.5))    # upper wing: big, tilted up
        lobes.append((0.5 + side * 0.17, 0.62, 0.13, 0.10, -side * 0.6))   # lower wing: small
    out = bytearray(size * size * 4)
    for py in range(size):
        y = (py + 0.5) / size
        for px in range(size):
            x = (px + 0.5) / size
            best = 0.0
            for cx, cy, rx, ry, tilt in lobes:
                dx, dy = x - cx, y - cy
                c, s = math.cos(tilt), math.sin(tilt)
                u, v = (dx * c + dy * s) / rx, (-dx * s + dy * c) / ry
                d = math.sqrt(u * u + v * v)
                if d < 1.0:
                    # soft inside, a brighter rim, like light caught in the wing's edge
                    best = max(best, 0.35 + 0.45 * d ** 3 + 0.2 * (1.0 - d))
            core = math.hypot(x - 0.5, y - 0.5) / 0.09
            if core < 1.0:
                best = max(best, 1.0 - 0.3 * core)
            a = int(round(255 * min(1.0, best)))
            i = 4 * (py * size + px)
            out[i:i + 4] = bytes((255, 255, 255, a))
    return bytes(out)


def write_png(path, width, height, rgba):
    """A plain RGBA PNG (no numpy needed)."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + rgba[y * width * 4:(y + 1) * width * 4] for y in range(height))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))
    return str(path)


def _hex_rgb(value):
    value = (value or "").lstrip("#")
    if len(value) != 6:
        return None
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _saturation(rgb):
    hi, lo = max(rgb), min(rgb)
    return 0.0 if hi == 0 else (hi - lo) / float(hi)


def glow_colour(colors):
    """The fairy's light colour from its painted colours (the brightest saturated one), as linear 0..1 RGB."""
    picks = [_hex_rgb(v) for v in (colors or {}).values() if isinstance(v, str)]
    picks = [p for p in picks if p]
    if not picks:
        picks = [_hex_rgb(DEFAULT_COLOURS[0])]
    best = max(picks, key=lambda p: _saturation(p) * 0.7 + max(p) / 255.0 * 0.3)
    # push it toward a light: full brightness, keep the hue
    top = max(1, max(best))
    return tuple(min(1.0, (c / float(top)) ** 2.2) for c in best)


def colours_for_extras(painted, count=EXTRA_FAIRIES):
    """Light colours for the extra fairies: the painted fairies' colours in turn, else the defaults."""
    pool = list(painted) or [glow_colour({"c": h}) for h in DEFAULT_COLOURS]
    return [pool[i % len(pool)] for i in range(count)]


def float_path(rng, points=5):
    """A closed loop of relative positions (cm) a fairy drifts along: (x, y, z) tuples, starting and
    ending at (0, 0, 0)."""
    reach = rng.uniform(*FLOAT_REACH)
    rise = rng.uniform(*FLOAT_RISE)
    turn = rng.uniform(0.0, 2.0 * math.pi)
    path = [(0.0, 0.0, 0.0)]
    for k in range(1, points):
        a = turn + 2.0 * math.pi * k / points + rng.uniform(-0.4, 0.4)
        r = reach * rng.uniform(0.55, 1.0)
        path.append((r * math.cos(a), r * math.sin(a), rise * math.sin(2.3 * k + turn)))
    path.append((0.0, 0.0, 0.0))
    return path


# --------------------------------------------------------------------------- #
# Unreal
# --------------------------------------------------------------------------- #
def _build_fairy_glow(material):
    """Unlit, additive: Colour x Glow x the wing texture.  Adds light, so it reads as a glow, not paint."""
    import rwb_layers
    x, link = rwb_layers._x, rwb_layers._link
    mel = unreal.MaterialEditingLibrary
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_ADDITIVE)
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    material.set_editor_property("two_sided", True)
    tex = x(material, unreal.MaterialExpressionTextureSampleParameter2D, -800, 0)
    tex.set_editor_property("parameter_name", "Wings")
    default = (unreal.load_asset("/Engine/EngineResources/WhiteSquareTexture")
               or unreal.load_asset("/Engine/EngineResources/DefaultTexture"))
    if default:
        tex.set_editor_property("texture", default)
    shape = x(material, unreal.MaterialExpressionMultiply, -550, 0)
    link(tex, "RGB", shape, "A")
    link(tex, "A", shape, "B")
    colour = rwb_layers._mask(material, rwb_layers._vector_param(material, "Colour", (1, 0.6, 1, 1), -800, 260), -600, 260)
    tinted = x(material, unreal.MaterialExpressionMultiply, -350, 100)
    link(shape, "", tinted, "A")
    link(colour, "", tinted, "B")
    bright = x(material, unreal.MaterialExpressionMultiply, -150, 160)
    link(tinted, "", bright, "A")
    link(rwb_layers._scalar_param(material, "Glow", 4.0, -350, 300), "", bright, "B")
    mel.connect_material_property(bright, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)


def _material(ctx, name, colour, glow, wings=None):
    import rwb_layers
    master = rwb_layers._material(FAIRY_MATERIAL, _build_fairy_glow)
    textures = {"Wings": wings} if wings else None
    return rwb_layers._mic(ctx["folder"] + "/Fairies", name, master,
                           vectors={"Colour": unreal.LinearColor(colour[0], colour[1], colour[2], 1.0)},
                           scalars={"Glow": glow}, textures=textures)


def wings_texture(ctx, cache_dir):
    """The wing texture, made once per world and imported."""
    import rwb_scene
    path = Path(cache_dir) / WING_TEXTURE
    if not path.is_file():
        write_png(path, 128, 128, wing_pixels(128))
    texture = rwb_scene._import_texture(str(path), ctx["folder"] + "/Fairies", "T_RWB_fairy_wings", str(cache_dir))
    if texture:
        try:
            texture.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_DEFAULT)
            texture.set_editor_property("srgb", True)
            unreal.EditorAssetLibrary.save_loaded_asset(texture, False)
        except Exception:
            pass
    return texture


def _add_float(actor, path, seconds):
    """Make ``actor`` drift along ``path`` (relative cm) in a loop when the game runs.  False if this
    engine won't add the component (the fairy then just hovers)."""
    try:
        sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
        handles = sds.k2_gather_subobject_data_for_instance(actor)
        params = unreal.AddNewSubobjectParams(parent_handle=handles[0], new_class=unreal.InterpToMovementComponent,
                                              blueprint_context=None)
        handle, reason = sds.add_new_subobject(params)
        if not unreal.SubobjectDataBlueprintFunctionLibrary.is_handle_valid(handle):
            return False
        data = unreal.SubobjectDataBlueprintFunctionLibrary.get_data(handle)
        mover = unreal.SubobjectDataBlueprintFunctionLibrary.get_object(data)
        points = [unreal.InterpControlPoint(position_control_point=unreal.Vector(*p), position_is_relative=True)
                  for p in path]
        mover.set_editor_property("control_points", points)
        mover.set_editor_property("duration", float(seconds))
        mover.set_editor_property("behaviour_type", unreal.InterpToBehaviourType.LOOP_RESET)
        try:
            mover.finalise_control_points()
        except Exception:
            pass
        return True
    except Exception as error:
        unreal.log_warning("Reference World Builder: fairy can't float (%s)" % str(error)[:160])
        return False


def spawn_fairy(ctx, location, colour, label, rng, wings=None):
    """One fairy at ``location`` (unreal.Vector): returns (core actor, floats?)."""
    import rwb_layers
    wb = rwb_layers._wb()
    sphere = unreal.load_asset("/Engine/BasicShapes/Sphere")
    plane = unreal.load_asset("/Engine/BasicShapes/Plane")
    key = "%02X%02X%02X" % tuple(int(round(255 * c)) for c in colour)
    core, core_mesh = rwb_layers._spawn(ctx, sphere, location, (CORE_CM / 100.0,) * 3, label,
                                        collision=False, shadow=False)
    core_mesh.set_material(0, _material(ctx, "MI_RWB_fairy_core_%s" % key, colour, CORE_GLOW))
    wing_mat = _material(ctx, "MI_RWB_fairy_wings_%s" % key, colour, WING_GLOW, wings)
    parts = []
    for n, yaw in enumerate((0.0, 90.0)):                # two crossed cards: wings from every side
        card, card_mesh = rwb_layers._spawn(ctx, plane, location + unreal.Vector(0, 0, 3), (WING_CM / 100.0, WING_CM / 100.0, 1.0),
                                            "%s_wings%d" % (label, n), yaw=yaw, roll=90.0, collision=False, shadow=False)
        card_mesh.set_material(0, wing_mat)
        parts.append(card)
    light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight, location)
    if light:
        comp = light.get_component_by_class(unreal.PointLightComponent)
        comp.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        try:
            comp.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
        except Exception:
            pass
        comp.set_editor_property("intensity", LIGHT_CANDELA)
        comp.set_editor_property("attenuation_radius", LIGHT_RADIUS)
        comp.set_editor_property("light_color", unreal.Color(*[int(round(255 * c ** (1 / 2.2))) for c in colour], 255))
        comp.set_editor_property("cast_shadows", False)
        wb._tag_actor(light, wb.GENERATED_TAG, wb.PROP_TAG, label="%s_light" % label)
        ctx["count"] += 1
        parts.append(light)
    for part in parts:
        part.attach_to_actor(core, "", unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD,
                             unreal.AttachmentRule.KEEP_WORLD, False)
    floats = _add_float(core, float_path(rng), rng.uniform(*FLOAT_SECONDS))
    return core, floats
