"""Render game-UI Twemoji icons as TRUE VECTOR matplotlib artists.

Generalizes the 06-16 `svg_icons.py` (which only knew the three role
glyphs) to any committed `assets/<name>.svg` — so the action / stat icons
(⚔️ attack, 🛡️ block, 💚 heal) render the same vector way as the role
icons (🤺 / 💂 / 👩‍⚕️). Each Twemoji SVG (viewBox 0 0 36 36) is a stack of
filled <path>s; we parse every path's `d` + `fill`, convert to a
matplotlib Path (svgpath2mpl), and draw PathPatches — real vector paths in
the PDF, multicolor, no rasterization.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import matplotlib.transforms as mtransforms
from matplotlib.patches import PathPatch
from svgpath2mpl import parse_path

ASSETS = Path(__file__).resolve().parent / "assets"

# Logical name -> committed SVG file.
ROLE_SVG = {0: "role_fighter.svg", 1: "role_tank.svg", 2: "role_medic.svg"}
ACTION_SVG = {"A": "action_attack.svg", "B": "action_block.svg",
              "H": "action_heal.svg"}
STAT_SVG = {0: "action_attack.svg",   # STR  (⚔️)
            1: "action_block.svg",    # DEF  (🛡️)
            2: "action_heal.svg"}     # SUP  (💚)


def _viewbox(root):
    vb = root.get("viewBox")
    if vb:
        x0, y0, w, h = (float(v) for v in re.split(r"[ ,]+", vb.strip()))
        return x0, y0, w, h
    return 0.0, 0.0, float(root.get("width", 36)), float(root.get("height", 36))


@lru_cache(maxsize=32)
def _load_subpaths(svg_name: str):
    """[(Path_in_unit_box, fill_color)] for one Twemoji SVG, pre-transformed
    into a centered [0,1]x[0,1] box with y flipped (upright)."""
    tree = ET.parse(ASSETS / svg_name)
    root = tree.getroot()
    _x0, _y0, w, h = _viewbox(root)
    scale = 1.0 / max(w, h)
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
        out.append((parse_path(d).transformed(norm), fill))
    return out


def draw_svg(ax, svg_name: str, cx: float, cy: float, size: float,
             zorder: float = 6, alpha: float = 1.0):
    """Draw a committed Twemoji SVG centered at (cx, cy), spanning `size`
    data units, on an equal-aspect `ax`. Returns the PathPatches added."""
    place = (mtransforms.Affine2D()
             .translate(-0.5, -0.5).scale(size).translate(cx, cy)
             + ax.transData)
    patches = []
    for mpath, fill in _load_subpaths(svg_name):
        pp = PathPatch(mpath, transform=place, facecolor=fill, edgecolor="none",
                       linewidth=0, zorder=zorder, alpha=alpha, antialiased=True)
        ax.add_patch(pp)
        patches.append(pp)
    return patches


def draw_role_icon(ax, role_idx, cx, cy, size, zorder=6, alpha=1.0):
    return draw_svg(ax, ROLE_SVG[role_idx], cx, cy, size, zorder, alpha)


def draw_action_icon(ax, action_letter, cx, cy, size, zorder=6, alpha=1.0):
    return draw_svg(ax, ACTION_SVG[action_letter], cx, cy, size, zorder, alpha)


def draw_stat_icon(ax, stat_idx, cx, cy, size, zorder=6, alpha=1.0):
    return draw_svg(ax, STAT_SVG[stat_idx], cx, cy, size, zorder, alpha)
