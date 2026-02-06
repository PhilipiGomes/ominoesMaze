#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect colored polyomino placements from a board image and match them to placements.
Compatível com polinômios de qualquer tamanho.
"""
from collections import deque, defaultdict
import json
import sys
from typing import List, Tuple, Iterable, Optional

from PIL import Image, ImageDraw
import numpy as np

# ------------------ parameters ------------------
SAMPLE_RADIUS = 3  # pixel radius to sample around each cell center
WHITE_THRESH = 230  # threshold for white cell detection
COLOR_DIST_THRESH = 2000  # squared distance threshold to consider same color
OUT_JSON = "init_selection_from_image.json"
OUT_OVERLAY = "regionsn_overlay.png"

# ------------------ image sampling ------------------
def sample_grid_colors(img_path: str, w: int, h: int) -> Tuple[List[List[Tuple[int,int,int]]], Tuple[List[int],List[int]], Image.Image]:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    img_h, img_w, _ = arr.shape

    # compute cell centers assuming uniform grid
    col_centers: List[int] = [int((i + 0.5) * img_w / w) for i in range(w)]
    row_centers: List[int] = [int((i + 0.5) * img_h / h) for i in range(h)]

    def sample_cell(cx: int, cy: int) -> Tuple[int,int,int]:
        r0 = max(0, cy - SAMPLE_RADIUS)
        r1 = min(img_h, cy + SAMPLE_RADIUS + 1)
        c0 = max(0, cx - SAMPLE_RADIUS)
        c1 = min(img_w, cx + SAMPLE_RADIUS + 1)
        patch = arr[r0:r1, c0:c1]
        if patch.size == 0:
            return (255, 255, 255)
        avg = tuple(np.round(patch.reshape(-1, 3).mean(axis=0)).astype(int))
        return avg  # type: ignore[return-value]

    # initialize with valid RGB tuples so static type checkers know element type
    colors: List[List[Tuple[int,int,int]]] = [[(255,255,255) for _ in range(w)] for _ in range(h)]
    for ry, cy in enumerate(row_centers):
        for cx, cx_px in enumerate(col_centers):
            colors[ry][cx] = sample_cell(cx_px, cy)
    return colors, (row_centers, col_centers), img


def is_white(rgb: Tuple[int,int,int], th: int = WHITE_THRESH) -> bool:
    return rgb[0] >= th and rgb[1] >= th and rgb[2] >= th


# ------------------ region clustering ------------------
def find_colored_regions(colors: List[List[Tuple[int,int,int]]], w: int, h: int) -> List[List[Tuple[int,int]]]:
    occ: List[List[bool]] = [[not is_white(colors[r][c]) for c in range(w)] for r in range(h)]
    visited: List[List[bool]] = [[False] * w for _ in range(h)]
    regions: List[List[Tuple[int,int]]] = []
    for r in range(h):
        for c in range(w):
            if not occ[r][c] or visited[r][c]:
                continue
            base = colors[r][c]
            q = deque([(r, c)])
            visited[r][c] = True
            comp: List[Tuple[int,int]] = [(c, r)]  # store as (x,y)
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and not visited[ny][nx]
                        and occ[ny][nx]
                    ):
                        c2 = colors[ny][nx]
                        d = (
                            (c2[0] - base[0]) ** 2
                            + (c2[1] - base[1]) ** 2
                            + (c2[2] - base[2]) ** 2
                        )
                        if d <= COLOR_DIST_THRESH:
                            visited[ny][nx] = True
                            q.append((ny, nx))
                            comp.append((nx, ny))
            regions.append(sorted(comp))
    return regions


# ------------------ free polyomino generation & symmetry helpers ------------------

_ROTATIONS = (
    lambda x, y: (x, y),
    lambda x, y: (-y, x),
    lambda x, y: (-x, -y),
    lambda x, y: (y, -x),
)


def _normalize_variants(cells: Iterable[Tuple[int, int]]):
    pts = tuple(cells)
    variants = []
    for reflect in (False, True):
        for rot_fn in _ROTATIONS:
            transformed = []
            if reflect:
                for x, y in pts:
                    xr = -x
                    yr = y
                    tx, ty = rot_fn(xr, yr)
                    transformed.append((tx, ty))
            else:
                for x, y in pts:
                    tx, ty = rot_fn(x, y)
                    transformed.append((tx, ty))
            minx = min(p[0] for p in transformed)
            miny = min(p[1] for p in transformed)
            norm = tuple(sorted(((p[0] - minx, p[1] - miny) for p in transformed)))
            variants.append(norm)
    return variants


def normalize(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    return min(_normalize_variants(cells))


def all_symmetries(canonical: Iterable[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], ...]]:
    pts = tuple(canonical)
    syms = set()
    for reflect in (False, True):
        for rot_fn in _ROTATIONS:
            transformed = []
            if reflect:
                for x, y in pts:
                    xr = -x
                    yr = y
                    tx, ty = rot_fn(xr, yr)
                    transformed.append((tx, ty))
            else:
                for x, y in pts:
                    tx, ty = rot_fn(x, y)
                    transformed.append((tx, ty))
            minx = min(p[0] for p in transformed)
            miny = min(p[1] for p in transformed)
            norm = tuple(sorted(((p[0] - minx, p[1] - miny) for p in transformed)))
            syms.add(norm)
    return sorted(syms)


def generate_free_polyominoes(n: int, ominoes_dict: Optional[dict] = None) -> List[Tuple[Tuple[int,int], ...]]:
    if n <= 0:
        return []
    if ominoes_dict is not None and n in ominoes_dict:
        return [normalize(c) for c in ominoes_dict[n]]

    seen = set()
    results: List[Tuple[Tuple[int,int], ...]] = []

    def recurse(cells: set):
        if len(cells) == n:
            key = normalize(cells)
            if key not in seen:
                seen.add(key)
                results.append(key)
            return
        frontier = set()
        for x, y in cells:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb not in cells:
                    frontier.add(nb)
        for nb in sorted(frontier):
            cells.add(nb)
            recurse(cells)
            cells.remove(nb)

    recurse({(0, 0)})
    return results


# ------------------ placements (shape -> board placements) ------------------
def placements_for_shape(shape: Iterable[Tuple[int, int]], w: int, h: int):
    syms = all_symmetries(shape)
    placements = []
    seen_masks = set()
    app = placements.append
    for s in syms:
        maxx = max(x for x, y in s)
        maxy = max(y for x, y in s)
        width_range = w - maxx
        height_range = h - maxy
        if width_range <= 0 or height_range <= 0:
            continue
        base_idxs = [y * w + x for x, y in s]
        base_mask = 0
        for bi in base_idxs:
            base_mask |= 1 << bi
        for ox in range(width_range):
            for oy in range(height_range):
                offset = oy * w + ox
                mask = base_mask << offset
                if mask in seen_masks:
                    continue
                seen_masks.add(mask)
                cells = [(x + ox, y + oy) for x, y in s]
                app({"cells": cells, "mask": mask})
    return placements


# ------------------ matching regions -> placements ------------------
def region_to_mask(region: Iterable[Tuple[int,int]], w: int, h: int) -> int:
    m = 0
    for x, y in region:
        idx = y * w + x
        m |= 1 << idx
    return m


def match_regions_to_placements(regions: List[List[Tuple[int,int]]], placements: List[dict], n: int, w: int, h: int):
    mask_to_idx = defaultdict(list)
    for i, p in enumerate(placements):
        mask_to_idx[p["mask"]].append(i)
    matched = {}
    unmatched = []
    for ridx, region in enumerate(regions):
        if len(region) != n:
            unmatched.append((ridx, region))
            continue
        rm = region_to_mask(region, w, h)
        if rm in mask_to_idx:
            matched[ridx] = mask_to_idx[rm][0]
        else:
            target = set(region)
            found = None
            for i, p in enumerate(placements):
                if set(p["cells"]) == target:
                    found = i
                    break
            if found is not None:
                matched[ridx] = found
            else:
                unmatched.append((ridx, region))
    return matched, unmatched


# ------------------ main flow ------------------
def main(img_path: str, n: int, w: int, h: int, ominoes_dict: Optional[dict] = None):
    colors, (row_centers, col_centers), pil_img = sample_grid_colors(img_path, w, h)
    regions = find_colored_regions(colors, w, h)
    regionsn = [r for r in regions if len(r) == n]
    print("Total regions found:", len(regions), f", regions of size {n}:", len(regionsn))

    shapes = generate_free_polyominoes(n, ominoes_dict=ominoes_dict)
    placements = []
    shape_map = []
    for sid, shape in enumerate(shapes):
        p = placements_for_shape(shape, w, h)
        base = len(placements)
        for item in p:
            item["shape_id"] = sid
            placements.append(item)
        shape_map.append(list(range(base, base + len(p))))
    print("Generated shapes:", len(shapes), " placements total:", len(placements))

    matched, unmatched = match_regions_to_placements(regionsn, placements, n, w, h)
    print("Matched regions:", len(matched), " Unmatched:", len(unmatched))

    selection = [matched[r] for r in sorted(matched.keys())]
    print("Detected selection (placement indices):", selection)

    out = {
        "grid": {"w": w, "h": h},
        "regions_total": len(regions),
        f"regions_size_{n}_count": len(regionsn),
        f"regions_size_{n}": regionsn,
        "matched": matched,
        "unmatched": unmatched,
        "selection": selection,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    vis = pil_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(vis)
    for ridx, region in enumerate(regionsn):
        xs = [c[0] for c in region]
        ys = [c[1] for c in region]
        cx = int(sum(col_centers[x] for x in xs) / len(xs))
        cy = int(sum(row_centers[y] for y in ys) / len(ys))
        draw.text((cx - 6, cy - 6), str(ridx), fill=(0, 0, 0))
    vis.save(OUT_OVERLAY)
    print("Saved JSON ->", OUT_JSON)
    print("Saved overlay ->", OUT_OVERLAY)
    return selection, regionsn, matched, unmatched


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python create.py </path/to/image.png> <omino_size> board_w board_h")
        sys.exit(1)
    imgf = sys.argv[1]
    n = int(sys.argv[2])
    w = int(sys.argv[3])
    h = int(sys.argv[4])
    sel, regs, matched, unmatched = main(imgf, n, w, h)
    print("\ninit_selection = {}".format(sel))
