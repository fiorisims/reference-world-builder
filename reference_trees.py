"""Trees in a painting: what they are, how big their canopies are, and how to grow more like them.

Paintings of forests rarely separate one tree from the next.  The detector sees a trunk, while the
canopy above it melts into every other canopy, and anything built in a tree (a treehouse, a
glasshouse round a trunk) gets called a castle.  This module reads trees by colour and structure,
not by name:

* **foliage** and **bark** are colour classes (green-to-teal leaves, red-to-brown wood).  They hold
  across styles: a muted teal canopy and a pink-brown trunk on the Sleeping Beauty painting, green
  and brown on a photo.  Sky, mist and stone fall in neither;
* a tree's **canopy** is the foliage joined to the top of its trunk, inside a window a few times the
  trunk's width, above the horizon (foliage below it is grass);
* a "castle" with a canopy over it and a trunk running up through it is a **treehouse**; a weak
  "castle" that is mostly one bark-coloured column is a **tree** (a trunk in the mist);
* the trunks across the whole picture give the **tree style**: how thick they are, where they fork,
  how far the branches spread, how wide the canopies grow and their colours.  ``lookalikes`` draws
  and paints new trees in that style, which are then modelled exactly like the painted ones, so the
  forest outside the painted view looks like the painting rather than like a model pack.

numpy only (no scipy), so it runs wherever the rest of the analysis runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

TREES = {"tree", "pine"}
ARCHITECTURE = {"castle", "house", "tower"}    # what a treehouse gets mistaken for
FOLIAGE_HUE = (55.0, 205.0)    # degrees: yellow-green .. teal
FOLIAGE_SAT = 0.10
FOLIAGE_VAL = 0.80             # brighter than this is sky or mist, not leaves
BARK_HUE = (335.0, 45.0)       # wraps round red: pink-brown .. orange-brown
BARK_SAT = 0.18
BARK_VAL = (0.22, 0.88)
CANOPY_REACH = 1.2             # the canopy window reaches this many object widths either side of it
CANOPY_REACH_MIN = 0.075       # ... but at least this share of the picture's width
CANOPY_REACH_MAX = 0.30        # ... and at most this
CANOPY_BAND = 0.06             # of the picture's height: how far above an object's top to look for leaves
CANOPY_MIN = 0.35              # foliage share of that band for the object to be under a canopy
CANOPY_LOW = 0.5               # a canopy hangs no lower than this share down the object
TRUNK_COLUMN = 0.45            # rows of the object crossed by one bark column, for "a trunk runs through it"
TRUNK_WINDOW = 0.25            # that column's width, as a share of the object's width
TREEHOUSE_MAX_BARK = 0.45      # more of the object than this is bark: a red brick house, not a trunk
WEAK_BUILDING = 0.3            # detector scores below this are "maybe a building"
TRUNK_ONLY = 0.7               # a weak building this much bark column is just a trunk
TRUNK_MIN_HEIGHT = 0.12        # of the picture's height, for a bark shape to count as a trunk
TRUNK_MIN_ASPECT = 3.0         # height / width
LOOKALIKES = 4                 # new tree designs per picture
LOOKALIKE_HEIGHT = 512         # pixels, of each look-alike's cut-out
STYLE_VERSION = "rwb-trees-1"

DEFAULT_STYLE = {
    "thickness": 0.07,         # base width / trunk height
    "taper": 0.55,             # width at the fork / base width
    "fork": 0.55,              # where the trunk splits, as a share of its height (None = never)
    "branches": 2,
    "spread": 0.45,            # how far apart the branch tips are / trunk height
    "lean": 0.0,
    "canopy_width": 0.7,       # canopy width / the whole tree's height
    "canopy_low": 0.5,         # canopy bottom above the ground / the whole tree's height
    "bark": ["#6b4a3a", "#8a6650", "#a3806a"],
    "leaves": ["#335533", "#4c7048", "#6a8f5e"],
    "trunks": 0,
    "canopies": 0,
}


# --------------------------------------------------------------------------- #
# colour classes
# --------------------------------------------------------------------------- #
def hsv(rgb):
    """Hue (degrees), saturation and value of an RGB uint8 array."""
    x = np.asarray(rgb, np.float32)[..., :3] / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    high, low = x.max(-1), x.min(-1)
    d = high - low
    hue = np.zeros_like(high)
    nz = d > 1e-6
    rm = nz & (high == r)
    gm = nz & (high == g) & ~rm
    bm = nz & ~rm & ~gm
    hue[rm] = (60.0 * ((g - b)[rm] / d[rm])) % 360.0
    hue[gm] = 60.0 * ((b - r)[gm] / d[gm]) + 120.0
    hue[bm] = 60.0 * ((r - g)[bm] / d[bm]) + 240.0
    sat = np.where(high > 0, d / np.maximum(high, 1e-6), 0.0)
    return hue, sat, high


def colour_classes(rgb):
    """(foliage, bark) bool maps."""
    hue, sat, val = hsv(rgb)
    foliage = (hue >= FOLIAGE_HUE[0]) & (hue <= FOLIAGE_HUE[1]) & (sat >= FOLIAGE_SAT) & (val < FOLIAGE_VAL)
    bark = (((hue >= BARK_HUE[0]) | (hue <= BARK_HUE[1])) & (sat >= BARK_SAT)
            & (val >= BARK_VAL[0]) & (val <= BARK_VAL[1]))
    return foliage, bark


def _dilate(mask):
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def grow_within(seed, allowed, limit=2000):
    """Everything in ``allowed`` connected to ``seed`` (4-connected), by repeated dilation."""
    region = seed & allowed
    for _ in range(limit):
        grown = _dilate(region) & allowed
        if grown.sum() == region.sum():
            break
        region = grown
    return region


def components(mask):
    """Connected regions of a bool map, as a label map (0 = none) and the number of regions."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    count = 0
    for start in zip(*np.nonzero(mask)):
        if labels[start]:
            continue
        count += 1
        stack = [start]
        labels[start] = count
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = count
                    stack.append((ny, nx))
    return labels, count


# --------------------------------------------------------------------------- #
# one object: under a canopy?  a trunk inside?
# --------------------------------------------------------------------------- #
def _extent(mask):
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def trunk_column(bark, mask):
    """Share of the object's rows crossed by one narrow bark column (a trunk running up it), 0..1."""
    if not mask.any():
        return 0.0
    top, bottom, left, right = _extent(mask)
    wood = (bark & mask)[top:bottom + 1, left:right + 1]
    rows, width = wood.shape
    window = max(2, int(round(width * TRUNK_WINDOW)))
    if window >= width:
        return float(wood.any(1).mean())
    sums = np.concatenate([np.zeros((rows, 1)), np.cumsum(wood, 1)], 1)
    per_window = sums[:, window:] - sums[:, :-window]           # rows x positions
    return float((per_window > 0).mean(0).max())


def canopy_evidence(foliage, mask, picture_h):
    """Share of the band just above (and around) the object's top that is foliage, 0..1."""
    top, bottom, left, right = _extent(mask)
    band_top = max(0, top - int(CANOPY_BAND * picture_h))
    band_bottom = top + max(2, int(0.15 * (bottom - top + 1)))
    band = np.zeros_like(mask)
    band[band_top:band_bottom, left:right + 1] = True
    band &= ~mask
    if band.sum() < 6:
        # the object runs off the top of the picture: look at its own top edge
        band[top:band_bottom, max(0, left - 3):right + 4] = True
        band &= ~mask
        if band.sum() < 6:
            return 0.0
    return float(foliage[band].mean())


def _top_centre(mask):
    """x of the middle of an outline's top fifth (where its trunk meets the leaves)."""
    top, bottom, _left, _right = _extent(mask)
    ys, xs = np.nonzero(mask[top:top + max(2, (bottom - top + 1) // 5)])
    return float(xs.mean())


def canopy_window(shape, mask, horizon=None):
    """Where this object's canopy may be: a rounded window over its top, above the horizon."""
    H, W = shape
    top, bottom, left, right = _extent(mask)
    width = right - left + 1
    reach = min(CANOPY_REACH_MAX * W, max(CANOPY_REACH_MIN * W, CANOPY_REACH * width))
    low = top + int(CANOPY_LOW * (bottom - top + 1))
    if horizon is not None:
        low = min(low, int(horizon))
    low = max(low, top + 2)
    cx = _top_centre(mask)
    rx = width / 2.0 + reach
    up = max(float(top), 0.3 * H, 1.0)
    down = max(1.0, float(low - top))
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    ry = np.where(ys < top, up, down)
    window = ((xs - cx) / rx) ** 2 + ((ys - top) / ry) ** 2 <= 1.0
    window[low:] = False
    return window, low


def canopy_for(foliage, mask, horizon=None):
    """The canopy joined to this object's top (a bool map, maybe empty)."""
    H, W = mask.shape
    top, bottom, left, right = _extent(mask)
    window, low = canopy_window(mask.shape, mask, horizon)
    allowed = foliage & window
    seed = np.zeros_like(mask)
    seed[max(0, top - int(CANOPY_BAND * H)):min(low, top + max(2, (bottom - top) // 6)), left:right + 1] = True
    seed |= _dilate(_dilate(mask))
    canopy = grow_within(seed & allowed, allowed)
    return fill_enclosed(canopy, mask) & ~mask


def fill_enclosed(region, also=None):
    """``region`` with the holes it (together with ``also``) surrounds filled in: darker leaves in a canopy
    fall outside the leaf colours, but they are still canopy."""
    if not region.any():
        return region
    top, bottom, left, right = _extent(region)
    solid = region[top:bottom + 1, left:right + 1]
    if also is not None:
        solid = solid | also[top:bottom + 1, left:right + 1]
    border = np.zeros_like(solid)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    outside = grow_within(border & ~solid, ~solid)
    out = region.copy()
    out[top:bottom + 1, left:right + 1] |= ~outside & ~solid | solid & region[top:bottom + 1, left:right + 1]
    return out


def share_canopies(items):
    """Foliage claimed by several trees goes to the nearest one (by distance from its top, in its widths)."""
    owners = [it for it in items if it.get("canopy_mask") is not None and it["canopy_mask"].any()]
    if len(owners) < 2:
        return
    H, W = owners[0]["canopy_mask"].shape
    claims = np.stack([it["canopy_mask"] for it in owners])
    contested = claims.sum(0) > 1
    if not contested.any():
        return
    ys, xs = np.nonzero(contested)
    cost = []
    for it in owners:
        top, bottom, left, right = _extent(it["own_mask"])
        width = max(4.0, right - left + 1.0)
        # a confidently seen tree reaches further than a doubtful one: its canopy is the one painted
        confidence = 0.2 + float(it.get("score", 0.0))
        cost.append(np.hypot(xs - _top_centre(it["own_mask"]), (ys - top) * 0.6) / (width * confidence))
    cost = np.where(claims[:, ys, xs], np.array(cost), np.inf)
    best = np.argmin(cost, 0)
    for n, it in enumerate(owners):
        lost = best != n
        it["canopy_mask"][ys[lost], xs[lost]] = False


def read_object(foliage, bark, mask, picture_h):
    """The structural facts about one outline."""
    if not mask.any():
        return {"canopy": 0.0, "trunk": 0.0, "bark": 0.0}
    return {"canopy": round(canopy_evidence(foliage, mask, picture_h), 3),
            "trunk": round(trunk_column(bark, mask), 3),
            "bark": round(float(bark[mask].mean()), 3)}


def compound_label(category, score, facts):
    """What an outline really is, from its structure.  Returns a category, or None to keep its own."""
    under_canopy = facts["canopy"] >= CANOPY_MIN
    if category == "treehouse":
        # the detector's own word, checked: all bark is a tree with no house, no trunk is just a house
        if facts["bark"] > TREEHOUSE_MAX_BARK:
            return "tree"
        if facts["trunk"] < TRUNK_COLUMN:
            return "house"
    if category in ARCHITECTURE:
        if score >= WEAK_BUILDING:
            if under_canopy and facts["trunk"] >= TRUNK_COLUMN and facts["bark"] <= TREEHOUSE_MAX_BARK:
                return "treehouse"                          # a building with a tree running up through it
        elif facts["trunk"] >= TRUNK_ONLY:
            return "tree"                                   # a weak "castle" that is one bark column
    return None


def trunk_only(bark, mask, min_share=0.05):
    """The bark in an outline, cleaned up: a misty "castle" that is really a tree keeps only its trunk.

    Returns the new mask, or None when too little of it is bark to be sure.
    """
    wood = bark & mask
    if wood.sum() < max(20, 0.1 * mask.sum()):
        return None
    top, bottom, left, right = _extent(wood)
    crop = wood[top:bottom + 1, left:right + 1]
    labels, count = components(crop)
    keep = np.zeros_like(crop)
    total = float(crop.sum())
    for n in range(1, count + 1):
        piece = labels == n
        if piece.sum() >= min_share * total:
            keep |= piece
    out = np.zeros_like(mask)
    out[top:bottom + 1, left:right + 1] = _dilate(_dilate(keep))
    out &= mask
    # paint texture leaves holes in the bark: close them along each row
    for r in range(top, bottom + 1):
        cols = np.nonzero(out[r])[0]
        if len(cols) >= 2:
            runs = _runs(out[r, cols.min():cols.max() + 1])
            for (a, _b), (c, _d) in zip(runs, runs[1:]):
                if c - _b <= 3:
                    out[r, cols.min() + _b:cols.min() + c] = True
    return out if out.any() else None


def refine(rgb, items, horizon=None, describe=None):
    """Relabel compound objects and give trees (and treehouses) their canopies.

    items: dicts with "mask" (bool HxW), "category", "label", "score"; changed in place.  Trees that
    get a canopy also get "canopy_mask" (bool HxW, not saved).
    describe(mask) -> placement dict (reference_objects.describe_mask), refreshed when a mask grows.
    Returns the list of changes, for the report.
    """
    if not items:
        return []
    foliage, bark = colour_classes(rgb)
    H = rgb.shape[0]
    changes = []
    growing = []
    for item in items:
        mask = item["mask"]
        if not mask.any():
            continue
        facts = read_object(foliage, bark, mask, H)
        item["tree_facts"] = facts
        new = compound_label(item["category"], float(item.get("score", 0.0)), facts)
        if new and new != item["category"]:
            changes.append("%s -> %s" % (item["label"], new))
            item["was"] = item["category"]
            item["category"] = new
            item["label"] = new
            if new == "tree" and item["was"] in ARCHITECTURE:
                trunk = trunk_only(bark, mask)
                if trunk is not None:
                    item["mask"] = mask = trunk
                    if describe is not None:
                        item["place"] = describe(mask)
        if item["category"] in TREES | {"treehouse"} and facts["canopy"] >= CANOPY_MIN:
            item["own_mask"] = mask
            item["canopy_mask"] = canopy_for(foliage, mask, horizon)
            growing.append(item)
    share_canopies(growing)
    for item in growing:
        canopy = item["canopy_mask"]
        mask = item.pop("own_mask")
        if canopy.sum() > 0.08 * mask.sum():
            item["mask"] = mask | canopy
            item["canopy_share"] = round(float(canopy.sum()) / float(item["mask"].sum()), 3)
            if describe is not None:
                item["place"] = describe(item["mask"])
        else:
            item["canopy_mask"] = None
    return changes


# --------------------------------------------------------------------------- #
# the picture's tree style
# --------------------------------------------------------------------------- #
def _runs(row):
    """(start, end) of each run of True in a 1-D bool array (end exclusive)."""
    padded = np.concatenate([[False], row, [False]])
    edges = np.nonzero(padded[1:] != padded[:-1])[0]
    return list(zip(edges[::2], edges[1::2]))


def trunk_shape(mask):
    """Thickness, fork, branches, spread and lean of one trunk-shaped region (or None)."""
    top, bottom, left, right = _extent(mask)
    height = bottom - top + 1
    rows = [_runs(mask[r, left:right + 1]) for r in range(top, bottom + 1)]
    # gaps of a pixel or two are brush texture, not two branches
    merged = []
    for runs in rows:
        out = []
        for a, b in runs:
            if out and a - out[-1][1] <= 2:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
        merged.append(out)
    widths = [sum(b - a for a, b in runs) for runs in merged]
    base = [w for w, runs in zip(widths[-max(3, height // 5):], merged[-max(3, height // 5):]) if len(runs) == 1]
    if not base:
        return None
    base_w = float(np.median(base))
    centres = [(r[0][0] + r[-1][1]) / 2.0 if r else np.nan for r in merged]
    lean = (np.nanmean(centres[height // 3: height // 2 + 1]) - np.nanmean(centres[-max(2, height // 6):])) / height
    if abs(lean) > 0.3:
        return None                                  # a root or a fallen branch, not a standing trunk
    # the fork: the lowest place (above the roots) where it stays split into two or more limbs
    fork = None
    split_rows = 0
    sustain = max(3, height // 12)
    for i in range(height - 1 - int(height * 0.15), -1, -1):
        limbs = [run for run in merged[i] if run[1] - run[0] >= 2]
        if len(limbs) >= 2:
            split_rows += 1
            if split_rows >= sustain:
                fork = (height - 1 - i - split_rows + 1) / float(height)
                break
        else:
            split_rows = 0
    above = merged[: max(1, int(height * (1 - fork)))] if fork else merged[: max(1, height // 4)]
    counts = [len(r) for r in above if len(r) >= 2]
    branches = int(round(np.median(counts))) if counts and fork else 1
    spans = [r[-1][1] - r[0][0] for r in above if r]
    spread = float(np.percentile(spans, 90)) / height if spans else 0.0
    at_fork = widths[min(height - 1, int(height * (1 - fork)))] if fork else widths[height // 3]
    return {"height": height, "thickness": base_w / height, "fork": fork, "branches": max(1, min(4, branches)),
            "spread": spread, "lean": float(lean), "taper": min(1.0, at_fork / max(1.0, base_w * max(1, branches)))}


def _hexes(pixels, n=3):
    """n colours from dark to light: the 20/50/80% luminance quantiles."""
    if len(pixels) == 0:
        return None
    lum = pixels @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    picks = [pixels[order[int(q * (len(order) - 1))]] for q in np.linspace(0.2, 0.8, n)]
    return ["#%02x%02x%02x" % tuple(int(c) for c in p) for p in picks]


def _sample(pixels, limit, rng):
    if len(pixels) > limit:
        pixels = pixels[rng.choice(len(pixels), limit, replace=False)]
    return pixels


def read_style(rgb, tree_masks=(), canopy_masks=(), exclude=None, horizon=None):
    """The picture's tree style (see DEFAULT_STYLE), plus colour samples for painting look-alikes.

    tree_masks: outlines of the painted trees (trunk + canopy); canopy_masks: their canopies.
    exclude: a bool map of non-tree objects whose bark-coloured paint isn't trunks (a canoe).
    """
    H, W = rgb.shape[:2]
    rng = np.random.default_rng(7)
    foliage, bark = colour_classes(rgb)
    if exclude is not None:
        bark = bark & ~exclude
    # trunks: tall thin bark shapes, read at half size (they are big and the flood fill is python)
    small = bark[::2, ::2]
    labels, count = components(small)
    shapes, trunk_pixels = [], []
    for n in range(1, count + 1):
        region = labels == n
        top, bottom, left, right = _extent(region)
        h = bottom - top + 1
        area = int(region.sum())
        if h < TRUNK_MIN_HEIGHT * H / 2.0 or area < 20 or h / max(1.0, area / h) < TRUNK_MIN_ASPECT:
            continue
        shape = trunk_shape(region)
        if shape:
            shapes.append(shape)
            ys, xs = np.nonzero(region)
            trunk_pixels.append(rgb[ys * 2, xs * 2])
    style = dict(DEFAULT_STYLE)
    if shapes:
        weights = np.array([s["height"] for s in shapes], np.float64)

        def weighted(key):
            values = np.array([s[key] for s in shapes], np.float64)
            order = np.argsort(values)
            cum = np.cumsum(weights[order])
            return float(values[order][np.searchsorted(cum, cum[-1] / 2.0)])
        forks = [s["fork"] for s in shapes if s["fork"]]
        style.update({
            "thickness": float(np.clip(weighted("thickness"), 0.025, 0.25)),
            "taper": float(np.clip(weighted("taper"), 0.3, 1.0)),
            "fork": float(np.clip(np.median(forks), 0.2, 0.85)) if len(forks) * 3 >= len(shapes) else None,
            "branches": int(np.clip(round(float(np.median([s["branches"] for s in shapes]))), 1, 4)),
            "spread": float(np.clip(np.median([s["spread"] for s in shapes if s["fork"]] or [DEFAULT_STYLE["spread"]]),
                                    0.1, 1.2)),
            "lean": float(np.clip(weighted("lean"), -0.15, 0.15)),
            "trunks": len(shapes),
        })
        if style["fork"] and style["branches"] < 2:
            style["branches"] = 2
    canopies = []
    for tree, canopy in zip(tree_masks, canopy_masks):
        if canopy is None or not canopy.any() or not tree.any():
            continue
        c_top, c_bottom, c_left, c_right = _extent(canopy)
        t_top, t_bottom, _l, _r = _extent(tree)
        whole = float(max(1, t_bottom - t_top))
        canopies.append(((c_right - c_left + 1) / whole, (t_bottom - c_bottom) / whole))
    if canopies:
        widths, lows = zip(*canopies)
        style.update({"canopy_width": float(np.clip(np.median(widths), 0.35, 1.6)),
                      "canopy_low": float(np.clip(np.median(lows), 0.3, 0.75)), "canopies": len(canopies)})
    bark_px = np.concatenate(trunk_pixels) if trunk_pixels else rgb[bark]
    leaf_area = np.zeros((H, W), bool)
    for canopy in canopy_masks:
        if canopy is not None:
            leaf_area |= canopy
    if leaf_area.sum() < 200:
        leaf_area = foliage.copy()
        if horizon is not None:
            leaf_area[int(horizon):] = False            # below the horizon it is grass
    leaf_px = rgb[leaf_area & foliage]
    bark_px = _sample(np.asarray(bark_px, np.float64).reshape(-1, 3), 6000, rng)
    leaf_px = _sample(np.asarray(leaf_px, np.float64).reshape(-1, 3), 6000, rng)
    style["bark"] = _hexes(bark_px) or DEFAULT_STYLE["bark"]
    style["leaves"] = _hexes(leaf_px) or DEFAULT_STYLE["leaves"]
    style["version"] = STYLE_VERSION
    return style, bark_px, leaf_px


def line(style):
    """One report line about the style."""
    fork = "fork at %d%%" % round(style["fork"] * 100) if style.get("fork") else "single trunk"
    return "Tree style: %d trunks read, %s, %d branches, leaves from %d%% of the height up, %.1fx as wide as tall" % (
        style.get("trunks", 0), fork, style.get("branches", 1), round(100 * style.get("canopy_low", 0.5)),
        style.get("canopy_width", 0.7) / max(0.05, 1.0 - style.get("canopy_low", 0.5)))


# --------------------------------------------------------------------------- #
# new trees in that style
# --------------------------------------------------------------------------- #
def _noise(h, w, cells, rng, stretch=(1.0, 1.0)):
    """Smooth value noise in 0..1: a coarse random grid, bilinearly enlarged."""
    gh = max(2, int(cells * stretch[0]))
    gw = max(2, int(cells * stretch[1] * w / float(max(1, h))))
    grid = rng.random((gh + 1, gw + 1))
    ys = np.linspace(0, gh, h)
    xs = np.linspace(0, gw, w)
    y0 = np.minimum(ys.astype(int), gh - 1)
    x0 = np.minimum(xs.astype(int), gw - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    a = grid[y0][:, x0]
    b = grid[y0][:, x0 + 1]
    c = grid[y0 + 1][:, x0]
    d = grid[y0 + 1][:, x0 + 1]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def _capsule(shape, p0, p1, r0, r1):
    """(inside bool map, signed cross position -1..1) of a tapered segment, in pixel coordinates."""
    h, w = shape
    x0, x1 = int(max(0, min(p0[0], p1[0]) - max(r0, r1) - 2)), int(min(w, max(p0[0], p1[0]) + max(r0, r1) + 3))
    y0, y1 = int(max(0, min(p0[1], p1[1]) - max(r0, r1) - 2)), int(min(h, max(p0[1], p1[1]) + max(r0, r1) + 3))
    inside = np.zeros(shape, bool)
    cross = np.zeros(shape, np.float64)
    if x1 <= x0 or y1 <= y0:
        return inside, cross
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length2 = max(1e-6, dx * dx + dy * dy)
    t = np.clip(((xs - p0[0]) * dx + (ys - p0[1]) * dy) / length2, 0.0, 1.0)
    px, py = p0[0] + t * dx, p0[1] + t * dy
    radius = r0 + (r1 - r0) * t
    dist = np.hypot(xs - px, ys - py)
    side = np.sign((xs - px) * dy - (ys - py) * dx)
    inside[y0:y1, x0:x1] = dist <= radius
    cross[y0:y1, x0:x1] = side * dist / np.maximum(radius, 1e-6)
    return inside, cross


def _blur(a, radius):
    """Separable box blur (numpy)."""
    out = a.astype(np.float64)
    k = max(1, int(radius))
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (k, k)
        padded = np.pad(out, pad, mode="edge")
        cum = np.cumsum(padded, axis)
        cum = np.concatenate([np.zeros_like(np.take(cum, [0], axis)), cum], axis)
        n = out.shape[axis]
        out = (np.take(cum, np.arange(2 * k + 1, 2 * k + 1 + n), axis) - np.take(cum, np.arange(n), axis)) / (2 * k + 1)
    return out


def draw_tree(style, seed, height=LOOKALIKE_HEIGHT):
    """A new tree's silhouette in the style: (wood bool map, wood cross-shading, canopy bool map,
    canopy shading 0..1).  Everything is in units of the whole tree's height; it stands at the bottom centre."""
    rng = np.random.default_rng(seed)
    jitter = lambda value, amount: value * float(rng.uniform(1.0 - amount, 1.0 + amount))
    canopy_low = float(np.clip(jitter(style["canopy_low"], 0.12), 0.25, 0.8))
    canopy_w = float(np.clip(jitter(style["canopy_width"], 0.18), 0.3, 1.8))
    canopy_h = 1.0 - canopy_low
    trunk_top = canopy_low + canopy_h * 0.62           # the branches end inside the leaves
    unit = height * 0.98
    # fork and spread were read on the trunks' visible part, below the leaves
    spread = jitter(style["spread"], 0.25) * canopy_low
    lean = jitter(style["lean"], 0.6) + float(rng.uniform(-0.03, 0.03))
    width = int((max(canopy_w * 1.12, spread * 1.4) + 2.0 * abs(lean) + 0.08) * unit) + 16
    shape = (height, width)
    cx, ground = width / 2.0, height - 2.0
    wood = np.zeros(shape, bool)
    cross = np.zeros(shape, np.float64)

    def stroke(p0, p1, r0, r1):
        inside, c = _capsule(shape, p0, p1, r0, r1)
        cross[inside & ~wood] = c[inside & ~wood]
        wood[inside] = True
    base_r = max(2.0, jitter(style["thickness"], 0.2) * trunk_top * unit / 2.0)
    fork = style.get("fork")
    fork_h = float(np.clip(jitter(fork, 0.15), 0.2, 0.9)) * canopy_low if fork else trunk_top * 0.97
    fork_pt = (cx + lean * fork_h * unit, ground - fork_h * unit)
    fork_r = max(1.5, base_r * float(np.clip(style.get("taper", 0.6), 0.4, 0.9)))
    stroke((cx, ground + base_r * 0.3), fork_pt, base_r * 1.25, fork_r)      # a little flare at the foot
    tips = []
    branches = max(1, int(style.get("branches", 2)) if fork else 1)
    for i in range(branches):
        side = (i / float(branches - 1) - 0.5) if branches > 1 else 0.0
        tip = (fork_pt[0] + (side * spread + float(rng.uniform(-0.04, 0.04))) * unit,
               ground - jitter(trunk_top, 0.05) * unit)
        stroke(fork_pt, tip, fork_r * (0.8 if branches > 1 else 1.0), max(1.0, fork_r * 0.3))
        tips.append(tip)
        # a side twig off each branch, so the crown isn't a bare Y
        t = float(rng.uniform(0.4, 0.75))
        mid = (fork_pt[0] + (tip[0] - fork_pt[0]) * t, fork_pt[1] + (tip[1] - fork_pt[1]) * t)
        out = 1.0 if side >= 0 else -1.0
        twig = (mid[0] + out * float(rng.uniform(0.06, 0.14)) * unit, mid[1] - float(rng.uniform(0.06, 0.12)) * unit)
        stroke(mid, twig, max(1.0, fork_r * 0.4), 1.0)
    # the canopy: overlapping clumps inside an ellipse from canopy_low to the top
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    ccx = cx + lean * trunk_top * unit + float(rng.uniform(-0.03, 0.03)) * unit
    ccy = ground - (canopy_low + canopy_h / 2.0) * unit
    rx, ry = canopy_w * unit / 2.0, canopy_h * unit / 2.0
    canopy = np.zeros(shape, bool)
    count = int(rng.integers(7, 12))
    for k in range(count + 1):
        if k == count:
            qx, qy, r = ccx, ccy, min(rx, ry) * 0.7
        else:
            a = 2 * math.pi * k / count + float(rng.uniform(-0.3, 0.3))
            reach = float(rng.uniform(0.45, 0.8))
            qx, qy = ccx + rx * reach * math.cos(a), ccy + ry * reach * math.sin(a)
            r = min(rx, ry) * float(rng.uniform(0.35, 0.55))
        canopy |= np.hypot(xs - qx, (ys - qy) * (rx / max(1.0, ry)) ** 0.5) <= r
    canopy &= np.hypot((xs - ccx) / rx, (ys - ccy) / ry) <= 1.02
    # soft light from the upper left over the whole crown, blurred so clumps don't show seams
    lit = 1.0 - np.hypot((xs - (ccx - 0.35 * rx)) / rx, (ys - (ccy - 0.45 * ry)) / ry) / 1.8
    edge = _blur(canopy.astype(np.float64), max(2, int(min(rx, ry) * 0.12)))
    shade = np.clip(0.7 * lit + 0.3 * edge, 0.0, 1.0)
    # leaves hide the wood they cover, except where branches show through gaps
    gaps = _noise(height, width, 9, rng) > 0.78
    wood &= ~(canopy & ~gaps) | (ys > ccy + ry * 0.55)
    return wood, cross, canopy & ~wood, shade


def _ramp(colors, t):
    """Colour along a dark->light list of colours, t 0..1 (array)."""
    stops = np.array(colors, np.float64)
    pos = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def _palette(pixels, fallback, n=9):
    """n colours from dark to light out of real painted pixels (so the hue stays the painter's)."""
    if pixels is None or len(pixels) < 20:
        return [tuple(int(fallback[i][j:j + 2], 16) for j in (1, 3, 5)) for i in range(len(fallback))]
    lum = pixels @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    chunks = np.array_split(pixels[order], n)
    return [tuple(c.mean(0)) for c in chunks if len(c)]


def paint_tree(style, bark_px, leaf_px, seed, height=LOOKALIKE_HEIGHT):
    """A painted RGBA cut-out of a new tree in the picture's style."""
    rng = np.random.default_rng(seed + 1000)
    wood, cross, canopy, shade = draw_tree(style, seed, height)
    h, w = wood.shape
    out = np.zeros((h, w, 4), np.uint8)
    bark = _palette(bark_px, style["bark"])
    leaves = _palette(leaf_px, style["leaves"])
    # bark: long vertical streaks, lit from the left
    streaks = 0.6 * _noise(h, w, 10, rng, (1.0, 6.0)) + 0.4 * _noise(h, w, 40, rng, (1.0, 6.0))
    lit = 0.5 - 0.35 * cross
    t = np.clip(0.55 * lit + 0.45 * streaks, 0.0, 1.0)
    colour = _ramp(bark, t)
    out[wood, :3] = np.clip(colour[wood], 0, 255).astype(np.uint8)
    # leaves: blotches, lighter on each clump's lit side and toward the top
    blotch = 0.6 * _noise(h, w, 8, rng) + 0.4 * _noise(h, w, 30, rng)
    top_light = 1.0 - np.linspace(0.0, 1.0, h)[:, None] * 0.6
    t = np.clip(0.4 * shade + 0.4 * blotch + 0.2 * top_light, 0.0, 1.0)
    colour = _ramp(leaves, t)
    out[canopy, :3] = np.clip(colour[canopy], 0, 255).astype(np.uint8)
    out[..., 3] = np.where(wood | canopy, 255, 0).astype(np.uint8)
    return out


def lookalikes(style, bark_px, leaf_px, out_dir, count=LOOKALIKES, seed=1):
    """Write ``count`` painted look-alike cut-outs.  Returns [{cutout, category, label, ...}]."""
    import pygame
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("lookalike_tree_*.png"):  # (and their _leaves masks)
        old.unlink()
    made = []
    for n in range(count):
        rgba = paint_tree(style, bark_px, leaf_px, seed * 97 + n)
        h, w = rgba.shape[:2]
        path = out_dir / ("lookalike_tree_%02d.png" % n)
        pygame.image.save(pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), (w, h), "RGBA"), str(path))
        leaves = out_dir / ("lookalike_tree_%02d_leaves.png" % n)
        _wood, _cross, canopy, _shade = draw_tree(style, seed * 97 + n)
        mask = np.zeros((h, w, 4), np.uint8)
        mask[..., 3] = np.where(canopy, 255, 0)
        pygame.image.save(pygame.image.frombuffer(mask.tobytes(), (w, h), "RGBA"), str(leaves))
        made.append({"id": "la_tree_%02d" % n, "cutout": str(path), "leaves": str(leaves), "category": "tree",
                     "label": "tree like the painted ones", "quality": 1.0, "roughness": 0.0,
                     "area": 0.05, "lookalike": "tree", "aspect": h / float(w)})
    return made


def save_style(style, path):
    Path(path).write_text(json.dumps(style, indent=1), encoding="utf-8")
