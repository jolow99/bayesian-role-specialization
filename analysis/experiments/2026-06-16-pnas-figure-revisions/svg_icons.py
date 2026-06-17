"""Render Twemoji role icons as TRUE VECTOR matplotlib artists.

Each Twemoji SVG (assets/role_*.svg, viewBox 0 0 36 36) is a stack of
filled <path> elements. We parse every path's `d` + `fill`, convert the
`d` to a matplotlib Path (svgpath2mpl), and draw them as PathPatches.
The result embeds as real vector paths in the PDF — no rasterization,
multicolor preserved — which is what the advisor asked for.

SVG y grows downward; matplotlib y grows upward, so we flip y and place
the glyph in a unit box, then an Affine2D maps that box anywhere on the
target axes.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path as FilePath

import matplotlib.transforms as mtransforms
from matplotlib.patches import PathPatch
from svgpath2mpl import parse_path

ASSETS = FilePath(__file__).resolve().parent / "assets"
ROLE_SVG = {0: "role_fighter.svg", 1: "role_tank.svg", 2: "role_medic.svg"}
_NS = "{http://www.w3.org/2000/svg}"


def _viewbox(root) -> tuple[float, float, float, float]:
    vb = root.get("viewBox")
    if vb:
        x0, y0, w, h = (float(v) for v in re.split(r"[ ,]+", vb.strip()))
        return x0, y0, w, h
    return 0.0, 0.0, float(root.get("width", 36)), float(root.get("height", 36))


@lru_cache(maxsize=8)
def _load_subpaths(role_idx: int):
    """[(Path_in_unit_box, fill_color)] for one role's Twemoji SVG.

    Each Path is pre-transformed into a [0,1]x[0,1] box with y flipped so
    the glyph is upright; callers compose with a placement transform.
    """
    tree = ET.parse(ASSETS / ROLE_SVG[role_idx])
    root = tree.getroot()
    _x0, _y0, w, h = _viewbox(root)
    scale = 1.0 / max(w, h)
    # center the glyph in the unit box, flip y
    norm = (mtransforms.Affine2D()
            .translate(-_x0, -_y0)
            .scale(scale, -scale)
            .translate((1 - w * scale) / 2, (1 + h * scale) / 2))
    out = []
    for el in root.iter():
        if not el.tag.endswith("path"):
            continue
        d = el.get("d")
        if not d:
            continue
        fill = el.get("fill", "#000000")
        if fill.lower() == "none":
            continue
        mpath = parse_path(d).transformed(norm)
        out.append((mpath, fill))
    return out


def draw_role_icon(ax, role_idx: int, cx: float, cy: float, size: float,
                   zorder: float = 6, alpha: float = 1.0):
    """Draw a role's vector Twemoji centered at (cx, cy), spanning `size`
    data units, on `ax`. Returns the list of PathPatches added."""
    place = (mtransforms.Affine2D()
             .translate(-0.5, -0.5).scale(size).translate(cx, cy)
             + ax.transData)
    patches = []
    for mpath, fill in _load_subpaths(role_idx):
        pp = PathPatch(mpath, transform=place, facecolor=fill, edgecolor="none",
                       linewidth=0, zorder=zorder, alpha=alpha,
                       antialiased=True)
        ax.add_patch(pp)
        patches.append(pp)
    return patches
