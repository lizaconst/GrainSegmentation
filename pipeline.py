# -*- coding: utf-8 -*-
"""
pipeline.py — WC grain segmentation for BSE-SEM images of WC-Co cemented carbides.

Single-pass pipeline (final version). For every image of a given grade:

  Stage 1 — SAM segmentation
    * SamAutomaticMaskGenerator (ViT-H, no fine-tuning) with per-grade
      parameters from GRADE_CONFIGS.
    * Intensity filter: masks whose mean brightness is below
      ``intensity_threshold`` (Co binder / pores) are discarded.

  Stage 2 — flatten (overlapping-mask resolution)
    * Masks are rasterised into a single label map from largest to
      smallest, so smaller masks overwrite larger ones and every pixel
      belongs to exactly one mask. This replaces pairwise IoU
      deduplication, which is blind to nested masks (IoU of a grain
      inside a cluster equals the area ratio, far below any threshold).
    * Each connected component is validated as a WC-phase region:
      area >= min_mask_region_area, erosion test (films thinner than
      ~5 px are removed), and relative core brightness >= REL_BRIGHTNESS_MIN
      of the WC level (flat-field correction + Otsu).
    * Components are vectorised with Douglas-Peucker at EPS_FINE
      ("fine" contours used for all final descriptors).

  Stage 3 — convex decomposition of merged grains
    * Each fine contour is coarsened with Douglas-Peucker at EPS_COARSE
      (a coarser polygon makes the cut search robust to contour noise;
      DP returns a subset of the input vertices, so cut endpoints are
      guaranteed to exist on the fine contour).
    * A contour is split only if it is genuinely non-convex: fails the
      angular convexity test with tolerance CONVEX_TOL AND has a convex
      hull area defect above both the relative (per-grade
      CONVEXITY_DEFECT_MIN) and the absolute (DEFECT_AREA_MIN_PX)
      thresholds, AND is large enough to contain two admissible grains.
    * Exhaustive search over decompositions with k = 1..MAX_CUTS
      non-crossing chords into k+1 convex parts, memoised over
      sub-polygons. Among valid decompositions with the smallest k, the
      one maximising the log-likelihood of interior angles (empirical
      angle distribution from ``angles.txt``) is selected.
    * Anti-oversplit guards on every chord:
        - both parts >= MIN_PART_AREA_PX (the splitter cannot create a
          grain smaller than what SAM itself is allowed to produce);
        - at least one endpoint at a reflex vertex (a cut that resolves
          no concavity passes through the grain body, not a junction);
        - neck gate: chord length <= MAX_CUT_NECK_RATIO * sqrt(min area)
          (an invisible WC/WC boundary is short relative to the grains
          it separates).
    * Cuts are transferred back to the fine contour by exact vertex
      matching; if a chord crosses a noise concavity of the fine contour,
      the coarse parts are used as a fallback (counted in the stats).

  Stage 4 — final validation and output
    * Every output polygon is re-validated for relative brightness
      (closes dark wedges produced by the splitter).
    * Per image the pipeline writes:
        <stem>_grains.json  — final grain polygons (fine contours),
        <stem>_cuts.json    — applied cut chords,
        <stem>_stats.json   — full attrition cascade of the image,
        <stem>_viz.png      — contours and cuts over the image,
        <stem>_debug_panel.png — per-blob decomposition panel.

Compared to the two-script version (pipeline_hpc_v2.py +
resplit_postprocess_v2.py), the redundant single-cut splitting stage and
its intermediate files (*_split_candidates.json, *_cut_lines.json) are
removed: the post-processor always reconstructed the pre-split polygons
and re-ran the decomposition from scratch, so the first-pass splitting
never influenced the final result.

Usage:
    python -u pipeline.py --alloy Ultra_Co11
or via a SLURM job array (see run_pipeline.sbatch).
"""

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--alloy", required=True,
                     help="Grade name: Ultra_Co6_2 | Ultra_Co8 | Ultra_Co11 | "
                          "Ultra_Co15 | Ultra_Co25")
_args = _parser.parse_args()

ALLOY = _args.alloy

# ── per-grade configuration ───────────────────────────────────────────────────
# sam_params selected by a sweep over the WC-phase coverage metric.
# Co8 / Co25 (finest grains): crop_n_layers=1 (the crop layer catches small
# grains) + relaxed stability threshold 0.80.
# convexity_defect_min: contours from the crop layer are rougher, so for
# Co8 / Co25 a 3-5 % hull defect is normal for a single grain; the splitter
# should fire only on clear agglomerates (a junction pocket gives >~8 %).
# min_part_area_px equals min_mask_region_area of the same grade: the
# splitter cannot create a grain smaller than what SAM itself may output.
GRADE_CONFIGS = {
    "Ultra_Co6_2": {
        "sam_params": {
            "points_per_side":        150,
            "pred_iou_thresh":        0.7,
            "stability_score_thresh": 0.85,
            "min_mask_region_area":   300,
        },
        "intensity_threshold":  100,
        "convexity_defect_min": 0.03,
        "min_part_area_px":     300,
    },
    "Ultra_Co8": {
        "sam_params": {
            "points_per_side":                150,
            "pred_iou_thresh":                0.7,
            "stability_score_thresh":         0.80,
            "min_mask_region_area":           210,
            "crop_n_layers":                  1,
            "crop_n_points_downscale_factor": 2,
        },
        "intensity_threshold":  100,
        "convexity_defect_min": 0.08,
        "min_part_area_px":     210,
    },
    "Ultra_Co11": {
        "sam_params": {
            "points_per_side":        150,
            "pred_iou_thresh":        0.9,
            "stability_score_thresh": 0.85,
            "min_mask_region_area":   300,
        },
        "intensity_threshold":  100,
        "convexity_defect_min": 0.03,
        "min_part_area_px":     300,
    },
    "Ultra_Co15": {
        "sam_params": {
            "points_per_side":        150,
            "pred_iou_thresh":        0.8,
            "stability_score_thresh": 0.85,
            "min_mask_region_area":   300,
        },
        "intensity_threshold":  100,
        "convexity_defect_min": 0.03,
        "min_part_area_px":     300,
    },
    "Ultra_Co25": {
        "sam_params": {
            "points_per_side":                150,
            "pred_iou_thresh":                0.7,
            "stability_score_thresh":         0.80,
            "min_mask_region_area":           210,
            "crop_n_layers":                  1,
            "crop_n_points_downscale_factor": 2,
        },
        "intensity_threshold":  100,
        "convexity_defect_min": 0.08,
        "min_part_area_px":     210,
    },
}

if ALLOY not in GRADE_CONFIGS:
    raise ValueError(f"Unknown grade '{ALLOY}'. Available: {list(GRADE_CONFIGS)}")

_cfg = GRADE_CONFIGS[ALLOY]

# ===================== CONFIGURATION =====================
# A Dropbox zip creates a nested folder Ultra_CoXX/Ultra_CoXX/;
# a flat layout ./images/Ultra_CoXX/ is found as a second fallback.
IMAGES_FOLDER_NESTED = f"./images/{ALLOY}/{ALLOY}"
IMAGES_FOLDER_FLAT   = f"./images/{ALLOY}"
SAVE_FOLDER          = f"./results/{ALLOY}"

SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"
ANGLES_FILE    = "./angles.txt"

N_IMAGES = 100          # max images per grade

# contour vectorisation
EPS_FINE   = 0.005      # DP factor for fine contours (descriptors)
EPS_COARSE = 0.02       # DP factor for the cut search
CONVEX_TOL = 0.02       # angular convexity tolerance (normalised cross products)

# decomposition
MAX_CUTS                = 3      # max chords per blob (None = up to triangulation)
DEFECT_AREA_MIN_PX      = 100    # absolute hull-defect threshold, px^2
REQUIRE_REFLEX_ENDPOINT = True   # each chord must touch a reflex vertex
MAX_CUT_NECK_RATIO      = 1.3    # L_cut <= ratio * sqrt(min(A1, A2))

# WC-phase brightness validation (relative criterion; an absolute threshold
# fails because of smooth shading gradients across the images)
REL_BRIGHTNESS_MIN = 0.85   # grain core >= 85 % of the WC-phase level
FLATFIELD_SIGMA_PX = 80     # Gaussian scale of the flat-field correction

# visualisation
SAVE_VIZ         = True
SAVE_DEBUG_PANEL = True
MAX_PANELS       = 40       # max blobs on the debug panel
CUT_COLOR        = "magenta"
CUT_LINEWIDTH    = 2.0

SAM_PARAMS           = _cfg["sam_params"]
INTENSITY_THRESHOLD  = _cfg["intensity_threshold"]
CONVEXITY_DEFECT_MIN = _cfg["convexity_defect_min"]
MIN_PART_AREA_PX     = _cfg["min_part_area_px"]
# =========================================================

import os
import sys
import json

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from PIL import Image
from shapely.geometry import Polygon, LineString
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ── angle prior ───────────────────────────────────────────────────────────────

def read_txt_to_array(filename, dtype=float):
    """Read a whitespace-separated numeric file into a flat 1-D array."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            values = line.strip().split()
            if values:
                data.append([dtype(v) for v in values])
    return np.array(data).reshape(1, -1)[0]


def load_log_p_angles(path):
    """Load the empirical interior-angle distribution and return log p.

    Zero-probability bins are floored at 1e-3 of the smallest positive
    value (or 1e-12) so that the log-likelihood stays finite.
    """
    p = read_txt_to_array(path)
    p_floor = max(p[p > 0].min() * 1e-3, 1e-12)
    p = np.where(p > 0, p, p_floor)
    return np.log(p)


# ── mask filtering and vectorisation ──────────────────────────────────────────

def filter_black_regions(masks, image, intensity_threshold=50):
    """Drop SAM masks whose mean brightness is below the threshold
    (Co binder pools and pores are dark in BSE contrast)."""
    return [m for m in masks if np.mean(image[m["segmentation"]]) > intensity_threshold]


def extract_polygon(mask, epsilon_factor=EPS_FINE):
    """Vectorise a binary mask: largest external contour, simplified with
    Douglas-Peucker at ``epsilon_factor`` * perimeter. Returns a vertex list."""
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = epsilon_factor * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    return approx.reshape(-1, 2).tolist()


# ── relative brightness (WC-phase validation) ─────────────────────────────────

def flatfield_and_wc_level(gray):
    """Flat-field correction of smooth shading + WC-phase brightness level.

    Returns ``corr`` (dimensionless brightness ~1 after dividing by a
    Gaussian background of scale FLATFIELD_SIGMA_PX) and ``wc_level``
    (median of the bright Otsu class on the corrected image).
    """
    g = gray.astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), FLATFIELD_SIGMA_PX)
    corr = g / np.maximum(bg, 1e-3)
    u8 = np.clip(corr * 128.0, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    wc_level = float(np.median(corr[u8 > thr]))
    return corr, wc_level


_ERODE_K = np.ones((3, 3), np.uint8)


def component_is_bright(corr_crop, comp_u8, wc_level):
    """True if the component core (after a 2-px erosion) is not darker than
    REL_BRIGHTNESS_MIN * WC level. The median is robust to the boundary rim."""
    core = cv2.erode(comp_u8, _ERODE_K, iterations=2)
    sel = core.astype(bool) if int(core.sum()) >= 9 else comp_u8.astype(bool)
    return float(np.median(corr_crop[sel])) >= REL_BRIGHTNESS_MIN * wc_level


def polygon_is_bright(poly, corr, wc_level):
    """Same criterion as :func:`component_is_bright`, applied to a polygon:
    rasterise, erode 2 px, compare the core median with the WC level."""
    pts = np.asarray(poly, dtype=np.float32)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round(pts - [x0, y0]).astype(np.int32)], 1)
    core = cv2.erode(m, _ERODE_K, iterations=2)
    sel = core.astype(bool) if int(core.sum()) >= 9 else m.astype(bool)
    return float(np.median(corr[y0:y1, x0:x1][sel])) >= REL_BRIGHTNESS_MIN * wc_level


# ── flatten: overlapping-mask resolution ──────────────────────────────────────

def flatten_masks_to_polygons(masks, gray, corr, wc_level, min_region_area):
    """Resolve mask overlaps via a label map.

    Masks are rasterised from largest to smallest (small ON TOP of large),
    so every pixel belongs to exactly one mask. This fully replaces
    pairwise IoU deduplication: nested masks (a grain inside a cluster)
    and duplicates from the crop layer are resolved automatically. Every
    resulting connected component is validated as a WC-phase region:
      * area >= min_region_area,
      * compactness: survives 2 erosions (films < ~5 px die),
      * relative core brightness >= REL_BRIGHTNESS_MIN of the WC level.
    Returns ``(polygons, stats_dict)``. Polygons are disjoint by construction.
    """
    h, w = gray.shape[:2]
    order = sorted(range(len(masks)),
                   key=lambda i: int(masks[i]["segmentation"].sum()),
                   reverse=True)
    label = np.zeros((h, w), dtype=np.int32)
    for lbl, idx in enumerate(order, start=1):
        label[masks[idx]["segmentation"]] = lbl

    polygons = []
    n_small = n_thin = n_dark = 0
    for lbl, idx in enumerate(order, start=1):
        seg = masks[idx]["segmentation"]
        ys, xs = np.where(seg)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = (label[y0:y1, x0:x1] == lbl).astype(np.uint8)
        n_cc, cc = cv2.connectedComponents(crop)
        for c in range(1, n_cc):
            comp = (cc == c).astype(np.uint8)
            if int(comp.sum()) < min_region_area:
                n_small += 1
                continue
            if cv2.erode(comp, _ERODE_K, iterations=2).sum() == 0:
                n_thin += 1
                continue
            if not component_is_bright(corr[y0:y1, x0:x1], comp, wc_level):
                n_dark += 1
                continue
            poly = extract_polygon(comp.astype(bool), EPS_FINE)
            if len(poly) >= 3:
                polygons.append([[int(x + x0), int(y + y0)] for x, y in poly])

    stats = {"n_flatten_small": n_small, "n_flatten_thin": n_thin,
             "n_flatten_dark": n_dark}
    print(f"  flatten: {len(masks)} masks -> {len(polygons)} segments "
          f"(small {n_small}, thin {n_thin}, dark {n_dark})", flush=True)
    return polygons, stats


# ── polygon geometry ──────────────────────────────────────────────────────────

def is_convex_tol(polygon, tol=CONVEX_TOL):
    """Convexity with tolerance: normalised cross products with
    |s| < tol are treated as zero (noise / collinearity)."""
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)
    if n < 4:
        return True
    signs = []
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        v1, v2 = b - a, c - b
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            continue
        s = (v1[0] * v2[1] - v1[1] * v2[0]) / norm
        if abs(s) >= tol:
            signs.append(np.sign(s))
    return len(set(signs)) <= 1


def polygon_area(polygon):
    """Shoelace area (faster than shapely for checks inside the search)."""
    pts = np.asarray(polygon, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def reflex_vertex_indices(polygon, tol=CONVEX_TOL):
    """Indices of reflex (concave) vertices, orientation-aware.
    Vertices with |normalised cross product| < tol are ignored."""
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)
    if n < 4:
        return set()
    x, y = pts[:, 0], pts[:, 1]
    signed2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    orient = 1.0 if signed2 > 0 else -1.0
    reflex = set()
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = b - a, c - b
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            continue
        s = orient * (v1[0] * v2[1] - v1[1] * v2[0]) / norm
        if s < -tol:
            reflex.add(i)
    return reflex


def effectively_convex(polygon):
    """Convex by angles (with tolerance), OR weakly non-convex by the
    relative hull defect, OR the defect is small in absolute px^2
    (contour noise, not a grain junction)."""
    if is_convex_tol(polygon):
        return True
    p = Polygon(polygon)
    if not p.is_valid or p.area <= 0:
        return True
    defect_rel = float(p.convex_hull.area / p.area - 1.0)
    defect_abs = float(p.convex_hull.area - p.area)
    return defect_rel < CONVEXITY_DEFECT_MIN or defect_abs < DEFECT_AREA_MIN_PX


def coarsen(polygon, eps_factor=EPS_COARSE):
    """DP coarsening of a stored polygon. Returns a SUBSET of its vertices,
    which guarantees that chord endpoints exist on the fine contour."""
    arr = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    eps = eps_factor * cv2.arcLength(arr, True)
    approx = cv2.approxPolyDP(arr, eps, True)
    return approx.reshape(-1, 2).tolist()


def split_polygon(polygon, i, j):
    """Split a polygon along the chord (i, j) into two vertex lists."""
    part1 = polygon[:i + 1] + polygon[j:]
    part2 = polygon[i:j + 1]
    return part1, part2


def is_valid_cut(polygon, i, j):
    """The chord (i, j) must lie strictly inside the polygon and must not
    cross any of its edges."""
    poly = Polygon(polygon)
    if not poly.is_valid:
        return False
    cut = LineString([polygon[i], polygon[j]])
    return (poly.contains(cut) and
            not any(cut.crosses(LineString(poly.exterior.coords[k:k + 2]))
                    for k in range(len(polygon) - 1)))


def compute_angle(p1, p2, p3):
    """Interior angle at p2 in integer degrees."""
    v1 = np.array(p1, dtype=float) - np.array(p2, dtype=float)
    v2 = np.array(p3, dtype=float) - np.array(p2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0
    cos_t = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return int(round(np.degrees(np.arccos(cos_t))))


def compute_log_likelihood(polygon, log_p_angles):
    """Sum of log-probabilities of all interior angles under the
    empirical angle prior."""
    n = len(polygon)
    return sum(log_p_angles[compute_angle(polygon[i - 1], polygon[i], polygon[(i + 1) % n])]
               for i in range(n))


# ── exhaustive k-cut convex decomposition ─────────────────────────────────────

def _decompose_k(polygon, k, log_p_angles, memo, min_part_area=0.0):
    """Best decomposition of ``polygon`` by exactly k chords into k+1
    convex parts. Returns ``(log_L, parts, cuts)`` or None. The search is
    exhaustive with memoisation over sub-polygons; ``cuts`` are listed in
    application order (this level first, then cuts of the parts).

    Guards on every chord: at least one endpoint at a reflex vertex (when
    reflex vertices exist), both parts >= min_part_area, and the neck gate
    L_cut <= MAX_CUT_NECK_RATIO * sqrt(min(A1, A2)).
    """
    key = (tuple(map(tuple, polygon)), k)
    if key in memo:
        return memo[key]

    if k == 0:
        if effectively_convex(polygon):
            res = (compute_log_likelihood(polygon, log_p_angles), [polygon], [])
        else:
            res = None
        memo[key] = res
        return res

    n = len(polygon)
    reflex = reflex_vertex_indices(polygon) if REQUIRE_REFLEX_ENDPOINT else set()
    # if there are no reflex vertices (non-convexity lives only in the hull
    # defect with near-collinear angles), the constraint is lifted
    enforce_reflex = REQUIRE_REFLEX_ENDPOINT and len(reflex) > 0

    best = None
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent vertices across the contour closure
            if enforce_reflex and i not in reflex and j not in reflex:
                continue  # the cut resolves no concavity
            if not is_valid_cut(polygon, i, j):
                continue
            part1, part2 = split_polygon(polygon, i, j)
            if len(part1) < 3 or len(part2) < 3:
                continue
            a1, a2 = polygon_area(part1), polygon_area(part2)
            if min_part_area > 0 and (a1 < min_part_area or a2 < min_part_area):
                continue  # a fragment smaller than an admissible grain
            cut_len = float(np.hypot(polygon[j][0] - polygon[i][0],
                                     polygon[j][1] - polygon[i][1]))
            if cut_len > MAX_CUT_NECK_RATIO * np.sqrt(min(a1, a2)):
                continue  # the chord is long relative to the parts — not a neck
            cut = [list(polygon[i]), list(polygon[j])]
            # distribute the remaining k-1 chords between the parts
            for k1 in range(k):
                r1 = _decompose_k(part1, k1, log_p_angles, memo, min_part_area)
                if r1 is None:
                    continue
                r2 = _decompose_k(part2, k - 1 - k1, log_p_angles, memo, min_part_area)
                if r2 is None:
                    continue
                ll = r1[0] + r2[0]
                if best is None or ll > best[0]:
                    best = (ll, r1[1] + r2[1], [cut] + r1[2] + r2[2])

    memo[key] = best
    return best


def find_best_decomposition(coarse_poly, log_p_angles, max_cuts=MAX_CUTS,
                            min_part_area=0.0):
    """Prefer fewer chords: try k = 1, then 2, ... up to triangulation.
    Returns ``(parts, cuts)`` or ``(None, None)``."""
    n = len(coarse_poly)
    limit = n - 3 if max_cuts is None else min(max_cuts, n - 3)
    memo = {}
    for k in range(1, limit + 1):
        res = _decompose_k(coarse_poly, k, log_p_angles, memo, min_part_area)
        if res is not None:
            return res[1], res[2]
    return None, None


# ── transferring cuts to the fine contour ─────────────────────────────────────

def index_exact(polygon, pt):
    """Index of the vertex with exactly matching coordinates, or None."""
    for k, v in enumerate(polygon):
        if v[0] == pt[0] and v[1] == pt[1]:
            return k
    return None


def valid_part(part):
    """A part is valid if it has >= 3 vertices and positive shapely area."""
    if len(part) < 3:
        return False
    p = Polygon(part)
    return p.is_valid and p.area > 0


def apply_cuts_to_fine(fine_poly, cuts):
    """Apply chords found on the coarse contour to the fine contour.

    Chord endpoints are vertices of the coarse contour, i.e. a subset of
    the fine contour's vertices, so exact coordinate matching is used.
    Returns the list of parts, or None (then the coarse parts are used
    as a fallback).
    """
    parts = [fine_poly]
    for cut in cuts:
        placed = False
        for idx, part in enumerate(parts):
            i = index_exact(part, cut[0])
            j = index_exact(part, cut[1])
            if i is None or j is None or i == j:
                continue
            if i > j:
                i, j = j, i
            if j - i < 2 or (i == 0 and j == len(part) - 1):
                continue  # the chord coincides with an edge of the fine contour
            p1, p2 = split_polygon(part, i, j)
            if valid_part(p1) and valid_part(p2):
                parts[idx:idx + 1] = [p1, p2]
                placed = True
                break
        if not placed:
            return None
    return parts


# ── visualisation ─────────────────────────────────────────────────────────────

def save_viz(image, polygons, cuts, save_path):
    """Final contours (random colours) and cut chords over the image."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    rng = np.random.default_rng(0)
    for poly in polygons:
        pts = np.array(poly)
        ax.plot(*zip(*np.vstack([pts, pts[:1]])), color=rng.random(3), linewidth=0.8)
    for line in cuts:
        arr = np.array(line)
        ax.plot(arr[:, 0], arr[:, 1], color=CUT_COLOR,
                linewidth=CUT_LINEWIDTH, zorder=5)
    ax.axis("off")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_debug_panel(records, save_path):
    """One panel per decomposed blob: original (grey), parts (colours),
    chords (magenta). Fallback-to-coarse cases are labelled."""
    records = records[:MAX_PANELS]
    n = len(records)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    for ax, rec in zip(axes, records):
        orig = np.array(rec["original"])
        ax.fill(orig[:, 0], orig[:, 1], color="lightgray", alpha=0.5,
                edgecolor="black", linewidth=1)
        if len(rec["parts"]) > 1:
            for k, part in enumerate(rec["parts"]):
                arr = np.array(part)
                c = colors[k % len(colors)]
                ax.fill(arr[:, 0], arr[:, 1], color=c, alpha=0.35,
                        edgecolor=c, linewidth=1.5)
            for line in rec["cuts"]:
                arr = np.array(line)
                ax.plot(arr[:, 0], arr[:, 1], color=CUT_COLOR, linewidth=3, zorder=5)
            title = f"{len(rec['cuts'])} cuts, {len(rec['parts'])} parts"
            if rec.get("fine_fallback"):
                title += " (coarse)"
            ax.set_title(title, fontsize=10)
        else:
            ax.set_title("not split", fontsize=10, color="red")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ── per-image processing ──────────────────────────────────────────────────────

def process_image(image_path, sam, log_p_angles, save_folder):
    """Full pipeline for a single image: SAM -> flatten -> convex
    decomposition -> final brightness validation -> files."""
    image = np.array(Image.open(image_path))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # flat-field correction is computed once and reused by all stages
    corr, wc_level = flatfield_and_wc_level(gray)

    # ── Stage 1: SAM + intensity filter ──
    mask_generator = SamAutomaticMaskGenerator(model=sam, **SAM_PARAMS)
    masks = mask_generator.generate(image)
    print(f"  SAM: {len(masks)} masks", flush=True)

    filtered = filter_black_regions(masks, image, INTENSITY_THRESHOLD)
    print(f"  after intensity filter: {len(filtered)}", flush=True)

    # ── Stage 2: flatten ──
    polygons, flat_stats = flatten_masks_to_polygons(
        filtered, gray, corr, wc_level,
        SAM_PARAMS.get("min_mask_region_area", 0))

    # ── Stage 3: convex decomposition ──
    final_polygons = []
    all_cuts       = []
    debug_records  = []

    n_convex_after_coarse = 0
    n_almost_convex       = 0
    n_too_small           = 0
    n_nonconvex           = 0
    n_split               = 0
    n_unsplit             = 0
    n_fine_fallback       = 0
    cuts_histogram        = {}   # {number of chords: number of blobs}

    for poly in polygons:
        coarse = coarsen(poly)
        if len(coarse) < 3 or is_convex_tol(coarse):
            n_convex_after_coarse += 1
            final_polygons.append(poly)
            continue
        if effectively_convex(coarse):
            # non-convex by angles, but the hull defect is small — keep as is
            n_almost_convex += 1
            final_polygons.append(poly)
            continue
        if polygon_area(poly) < 2 * MIN_PART_AREA_PX:
            # the blob physically cannot contain two admissible grains
            n_too_small += 1
            final_polygons.append(poly)
            continue

        n_nonconvex += 1
        parts_coarse, cuts = find_best_decomposition(
            coarse, log_p_angles, min_part_area=MIN_PART_AREA_PX)

        if parts_coarse is None:
            n_unsplit += 1
            final_polygons.append(poly)
            debug_records.append({"original": poly, "parts": [poly], "cuts": []})
            continue

        fine_fallback = False
        parts = apply_cuts_to_fine(poly, cuts)
        if parts is None:
            # a chord crossed a noise concavity of the fine contour
            parts = parts_coarse
            fine_fallback = True
            n_fine_fallback += 1

        n_split += 1
        k = len(cuts)
        cuts_histogram[k] = cuts_histogram.get(k, 0) + 1
        final_polygons.extend(parts)
        all_cuts.extend(cuts)
        debug_records.append({"original": poly, "parts": parts,
                              "cuts": cuts, "fine_fallback": fine_fallback})

    # ── Stage 4: final brightness validation ──
    kept = [p for p in final_polygons
            if len(p) >= 3 and polygon_is_bright(p, corr, wc_level)]
    n_dark_removed = len(final_polygons) - len(kept)
    final_polygons = kept

    print(f"  non-convex: {n_nonconvex}, split: {n_split}, "
          f"unsplit: {n_unsplit}, dark removed: {n_dark_removed}", flush=True)
    print(f"  final grains: {len(final_polygons)}", flush=True)

    # ── output ──
    stem = os.path.splitext(os.path.basename(image_path))[0]

    stats = {
        "alloy":                    ALLOY,
        "pipeline_version":         "final_merged",
        "eps_fine":                 EPS_FINE,
        "eps_coarse":               EPS_COARSE,
        "convex_tol":               CONVEX_TOL,
        "max_cuts":                 MAX_CUTS,
        "convexity_defect_min":     CONVEXITY_DEFECT_MIN,
        "min_part_area_px":         MIN_PART_AREA_PX,
        "defect_area_min_px":       DEFECT_AREA_MIN_PX,
        "require_reflex_endpoint":  REQUIRE_REFLEX_ENDPOINT,
        "max_cut_neck_ratio":       MAX_CUT_NECK_RATIO,
        "rel_brightness_min":       REL_BRIGHTNESS_MIN,
        "flatfield_sigma_px":       FLATFIELD_SIGMA_PX,
        "n_sam_masks":              len(masks),
        "n_after_intensity_filter": len(filtered),
        **flat_stats,
        "n_after_flatten":          len(polygons),
        "n_convex_after_coarsen":   n_convex_after_coarse,
        "n_almost_convex":          n_almost_convex,
        "n_too_small_to_split":     n_too_small,
        "n_nonconvex_after_coarsen": n_nonconvex,
        "n_blobs_split":            n_split,
        "n_blobs_unsplit":          n_unsplit,
        "n_cuts_total":             len(all_cuts),
        "cuts_histogram":           cuts_histogram,
        "n_fine_fallback":          n_fine_fallback,
        "n_dark_removed":           n_dark_removed,
        "n_final_grains":           len(final_polygons),
    }

    with open(os.path.join(save_folder, f"{stem}_grains.json"), "w") as f:
        json.dump(final_polygons, f)
    with open(os.path.join(save_folder, f"{stem}_cuts.json"), "w") as f:
        json.dump(all_cuts, f)
    with open(os.path.join(save_folder, f"{stem}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if SAVE_VIZ:
        save_viz(image, final_polygons, all_cuts,
                 os.path.join(save_folder, f"{stem}_viz.png"))
    if SAVE_DEBUG_PANEL:
        save_debug_panel(debug_records,
                         os.path.join(save_folder, f"{stem}_debug_panel.png"))

    print(f"  -> {save_folder}/{stem}_*", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    IMAGES_FOLDER = IMAGES_FOLDER_NESTED if os.path.isdir(IMAGES_FOLDER_NESTED) \
        else IMAGES_FOLDER_FLAT

    print(f"Grade   : {ALLOY}", flush=True)
    print(f"Images  : {IMAGES_FOLDER}", flush=True)
    print(f"Results : {SAVE_FOLDER}", flush=True)
    print(f"SAM params : {SAM_PARAMS}", flush=True)
    print(f"intensity thresh : {INTENSITY_THRESHOLD}  |  "
          f"rel brightness : {REL_BRIGHTNESS_MIN}", flush=True)
    print(f"defect_min : {CONVEXITY_DEFECT_MIN}  |  "
          f"min_part_area : {MIN_PART_AREA_PX}", flush=True)
    print(f"PyTorch {torch.__version__}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    log_p_angles = load_log_p_angles(ANGLES_FILE)
    print("Angle prior loaded.", flush=True)

    print("Loading SAM...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
    print("SAM ready.", flush=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    file_list = sorted(
        f for f in os.listdir(IMAGES_FOLDER)
        if os.path.splitext(f)[1].lower() in exts
    )[:N_IMAGES]

    if not file_list:
        print(f"No images found in {IMAGES_FOLDER}", flush=True)
        sys.exit(1)

    print(f"\nProcessing {len(file_list)} images -> {SAVE_FOLDER}\n", flush=True)
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    for i, name in enumerate(file_list):
        stem = os.path.splitext(name)[0]
        stats_path = os.path.join(SAVE_FOLDER, f"{stem}_stats.json")
        if os.path.exists(stats_path):
            print(f"[{i+1}/{len(file_list)}] {name}: already done, skipping", flush=True)
            continue
        print(f"[{i+1}/{len(file_list)}] {name}", flush=True)
        process_image(
            image_path=os.path.join(IMAGES_FOLDER, name),
            sam=sam,
            log_p_angles=log_p_angles,
            save_folder=SAVE_FOLDER,
        )

    print("\nDone!", flush=True)
