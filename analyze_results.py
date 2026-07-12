# -*- coding: utf-8 -*-
"""
analyze_results.py — grain statistics over the pipeline output
(results/<alloy>/*_grains.json), producing the tables and figures of the
paper. CPU-only, no scipy, self-contained.

For each grade (after excluding grains touching the frame border and
residual non-convex contours, as described in Methods):
  * grain counts: total and per image (mean +- std)
  * d_eq: median, mean, lognormal fit mu/sigma (px and um), KS statistic
  * F_max, F_min (exact, via the convex hull): means in px and um
  * psi_A = F_min/F_max and sphericity S: mean, median, mode
  * attrition cascade aggregated from *_stats.json

Output (in ANALYSIS_FOLDER):
  grains_<alloy>.csv   — per-grain descriptor table
  summary.csv          — per-grade summary (all numbers for the paper)
  attrition.csv        — per-stage attrition cascade
  latex_tables.txt     — ready-made bodies of the paper tables
  deq_lognormal.png, area_all.png, perimeter_all.png, min_feret_all.png,
  max_feret_all.png, sphericity_all.png, aspect_ratio_all.png

Usage:
    python -u analyze_results.py
"""

# ===================== CONFIGURATION =====================
ALLOYS = ["Ultra_Co6_2", "Ultra_Co8", "Ultra_Co11", "Ultra_Co15", "Ultra_Co25"]

RESULTS_ROOT    = "./results"
IMAGES_ROOT     = "./images"
ANALYSIS_FOLDER = "./analysis"

# um per pixel; 0.0499 is confirmed for Ultra_Co11
# (TESCAN View field 76.32 um / panel width 1530 px at MAG 5.00 kx).
# The script prints the actual image width of every grade — if it differs
# from REFERENCE_WIDTH_PX, the scale of that grade MUST be re-checked
# against the image metadata!
SCALE_UM_PER_PX = {
    "Ultra_Co6_2": 0.0499,
    "Ultra_Co8":   0.0499,
    "Ultra_Co11":  0.0499,
    "Ultra_Co15":  0.0499,
    "Ultra_Co25":  0.0499,
}
REFERENCE_WIDTH_PX = 1530   # width for which the scale is confirmed

BORDER_MARGIN_PX = 2

# "effectively convex" criterion — a COPY of pipeline.py, so that the
# exclusion of residual non-convex contours matches the methodology
CONVEX_TOL = 0.02
CONVEXITY_DEFECT_MIN = {
    "Ultra_Co6_2": 0.03, "Ultra_Co8": 0.08, "Ultra_Co11": 0.03,
    "Ultra_Co15": 0.03, "Ultra_Co25": 0.08,
}
DEFECT_AREA_MIN_PX = 100
# =========================================================

import os
import json
import glob
import csv
from math import erf

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon


# ── geometry ──────────────────────────────────────────────────────────────────

def polygon_area(poly):
    """Shoelace area of a polygon given as a vertex list."""
    pts = np.asarray(poly, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_perimeter(poly):
    """Perimeter of a closed polygon."""
    pts = np.asarray(poly, dtype=float)
    return float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))


def is_convex_tol(poly, tol=CONVEX_TOL):
    """Convexity with tolerance on normalised cross products."""
    pts = np.asarray(poly, dtype=float)
    n = len(pts)
    signs = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = b - a, c - b
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            continue
        signs.append((v1[0] * v2[1] - v1[1] * v2[0]) / norm)
    signs = np.array(signs)
    return bool(np.all(signs >= -tol) or np.all(signs <= tol))


def effectively_convex(poly, alloy):
    """As in pipeline.py: angular tolerance OR small hull defect
    (relative per-grade OR absolute)."""
    if is_convex_tol(poly):
        return True
    p = Polygon(poly)
    if not p.is_valid or p.area <= 0:
        return True
    defect_rel = float(p.convex_hull.area / p.area - 1.0)
    defect_abs = float(p.convex_hull.area - p.area)
    return (defect_rel < CONVEXITY_DEFECT_MIN.get(alloy, 0.03)
            or defect_abs < DEFECT_AREA_MIN_PX)


def feret_diameters(poly):
    """Exact F_max and F_min via the convex hull.

    F_max — diameter of the hull vertex set (max pairwise distance);
    F_min — minimal width: over hull edges, the minimum of the maximal
    vertex distance to the edge line (minimal-width theorem)."""
    pts = np.asarray(poly, dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(float)
    h = len(hull)
    if h < 2:
        return 0.0, 0.0
    if h == 2:
        d = float(np.linalg.norm(hull[0] - hull[1]))
        return d, 0.0
    diff = hull[:, None, :] - hull[None, :, :]
    fmax = float(np.sqrt((diff ** 2).sum(-1).max()))
    fmin = np.inf
    for i in range(h):
        a, b = hull[i], hull[(i + 1) % h]
        e = b - a
        ln = np.linalg.norm(e)
        if ln < 1e-9:
            continue
        n_vec = np.array([-e[1], e[0]]) / ln
        width = float(np.abs((hull - a) @ n_vec).max())
        fmin = min(fmin, width)
    return fmax, float(fmin)


def touches_border(poly, w, h, margin=BORDER_MARGIN_PX):
    """True if any vertex lies within ``margin`` px of the frame border."""
    a = np.asarray(poly, dtype=float)
    return (a[:, 0].min() <= margin or a[:, 1].min() <= margin or
            a[:, 0].max() >= w - 1 - margin or a[:, 1].max() >= h - 1 - margin)


def find_image_file(alloy, stem):
    """Locate the source image for a given grade and file stem
    (nested Dropbox layout first, flat layout second)."""
    for sub in (os.path.join(IMAGES_ROOT, alloy, alloy),
                os.path.join(IMAGES_ROOT, alloy)):
        for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
            p = os.path.join(sub, stem + ext)
            if os.path.exists(p):
                return p
    return None


# ── lognormal and KS (closed forms, no scipy) ─────────────────────────────────

def lognorm_fit(x):
    """MLE fit of a lognormal: returns (mu, sigma) of log x."""
    logx = np.log(x)
    return float(logx.mean()), float(logx.std(ddof=0))


def lognorm_pdf(x, mu, sigma):
    """Lognormal probability density."""
    return (np.exp(-((np.log(x) - mu) ** 2) / (2.0 * sigma ** 2))
            / (x * sigma * np.sqrt(2.0 * np.pi)))


def lognorm_ks(x, mu, sigma):
    """Kolmogorov-Smirnov statistic of the sample against the fitted
    lognormal. Note: at N ~ 10^4-10^5 the p-value is meaningless; the
    statistic itself is the reportable quantity."""
    xs = np.sort(x)
    z = (np.log(xs) - mu) / (sigma * np.sqrt(2.0))
    cdf = 0.5 * (1.0 + np.array([erf(v) for v in z]))
    n = len(xs)
    return float(max(np.max(np.arange(1, n + 1) / n - cdf),
                     np.max(cdf - np.arange(0, n) / n)))


def hist_mode(x, lo, hi, bins=50):
    """Mode estimated as the centre of the tallest histogram bin."""
    h, edges = np.histogram(x, bins=bins, range=(lo, hi))
    i = int(np.argmax(h))
    return float(0.5 * (edges[i] + edges[i + 1]))


# ── aggregation of per-stage json statistics ──────────────────────────────────

def aggregate_stage_stats(folder, pattern, keys):
    """Per-image means of numeric fields from *_stats.json files."""
    acc = {k: [] for k in keys}
    for path in glob.glob(os.path.join(folder, pattern)):
        with open(path) as f:
            s = json.load(f)
        for k in keys:
            if k in s and isinstance(s[k], (int, float)):
                acc[k].append(s[k])
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

    summary_rows = []
    attrition_rows = []
    all_descr = {}   # alloy -> dict of np.arrays (um and dimensionless)

    for alloy in ALLOYS:
        seg_folder = os.path.join(RESULTS_ROOT, alloy)
        files = sorted(glob.glob(os.path.join(seg_folder, "*_grains.json")))
        if not files:
            print(f"{alloy}: no data in {seg_folder}, skipping", flush=True)
            continue
        scale = SCALE_UM_PER_PX[alloy]

        rows = []
        per_image_counts = []
        n_border = n_nonconvex_resid = 0
        widths_seen = set()

        for path in files:
            stem = os.path.basename(path).replace("_grains.json", "")
            img_path = find_image_file(alloy, stem)
            if img_path is None:
                print(f"  {alloy}/{stem}: image not found, skipping", flush=True)
                continue
            with Image.open(img_path) as im:
                w, h = im.size
            widths_seen.add(w)
            with open(path) as f:
                polys = json.load(f)

            n_kept_img = 0
            for poly in polys:
                if len(poly) < 3:
                    continue
                if touches_border(poly, w, h):
                    n_border += 1
                    continue
                if not effectively_convex(poly, alloy):
                    n_nonconvex_resid += 1
                    continue
                area_px = polygon_area(poly)
                if area_px <= 0:
                    continue
                perim_px = polygon_perimeter(poly)
                fmax_px, fmin_px = feret_diameters(poly)
                if fmax_px <= 0 or fmin_px <= 0:
                    continue
                deq_px = np.sqrt(4.0 * area_px / np.pi)
                rows.append(dict(
                    image=stem,
                    area_um2=area_px * scale ** 2,
                    perimeter_um=perim_px * scale,
                    deq_um=deq_px * scale,
                    fmax_px=fmax_px, fmin_px=fmin_px,
                    fmax_um=fmax_px * scale, fmin_um=fmin_px * scale,
                    psi_a=fmin_px / fmax_px,
                    sphericity=2.0 * np.sqrt(np.pi * area_px) / perim_px,
                ))
                n_kept_img += 1
            per_image_counts.append(n_kept_img)

        d = {k: np.array([r[k] for r in rows]) for k in rows[0]
             if k != "image"}
        all_descr[alloy] = d
        n = len(rows)

        # scale sanity check
        if widths_seen != {REFERENCE_WIDTH_PX}:
            print(f"  !!! image widths {sorted(widths_seen)} px != "
                  f"{REFERENCE_WIDTH_PX} px — RE-CHECK SCALE_UM_PER_PX "
                  f"for {alloy} !!!", flush=True)

        # d_eq fit (sigma is scale-free, mu_um = mu_px + ln scale)
        mu_px, sigma = lognorm_fit(d["deq_um"] / scale)
        mu_um = mu_px + np.log(scale)
        ks = lognorm_ks(d["deq_um"], mu_um, sigma)

        summary_rows.append(dict(
            alloy=alloy,
            n_images=len(per_image_counts),
            image_width_px="/".join(map(str, sorted(widths_seen))),
            scale_um_px=scale,
            grains_total=n,
            grains_per_image_mean=round(float(np.mean(per_image_counts)), 1),
            grains_per_image_std=round(float(np.std(per_image_counts, ddof=1)), 1),
            excluded_border=n_border,
            excluded_residual_nonconvex=n_nonconvex_resid,
            deq_median_um=round(float(np.median(d["deq_um"])), 3),
            deq_mean_um=round(float(np.mean(d["deq_um"])), 3),
            lognorm_mu_px=round(mu_px, 4),
            lognorm_mu_um=round(mu_um, 4),
            lognorm_sigma=round(sigma, 4),
            lognorm_median_um=round(float(np.exp(mu_um)), 3),
            ks_stat=round(ks, 4),
            fmax_mean_px=round(float(np.mean(d["fmax_px"])), 1),
            fmin_mean_px=round(float(np.mean(d["fmin_px"])), 1),
            fmax_mean_um=round(float(np.mean(d["fmax_um"])), 3),
            fmin_mean_um=round(float(np.mean(d["fmin_um"])), 3),
            psi_a_mean=round(float(np.mean(d["psi_a"])), 3),
            psi_a_median=round(float(np.median(d["psi_a"])), 3),
            psi_a_mode=round(hist_mode(d["psi_a"], 0, 1), 3),
            sphericity_mean=round(float(np.mean(d["sphericity"])), 3),
            sphericity_mode=round(hist_mode(d["sphericity"], 0, 1), 3),
        ))

        # attrition cascade from the per-image stats
        pipe = aggregate_stage_stats(seg_folder, "*_stats.json", [
            "n_sam_masks", "n_after_intensity_filter", "n_after_flatten",
            "n_flatten_small", "n_flatten_thin", "n_flatten_dark",
            "n_nonconvex_after_coarsen", "n_too_small_to_split",
            "n_blobs_split", "n_dark_removed", "n_final_grains"])
        attrition_rows.append(dict(
            alloy=alloy,
            sam_masks=round(pipe["n_sam_masks"], 1),
            after_intensity=round(pipe["n_after_intensity_filter"], 1),
            after_flatten=round(pipe["n_after_flatten"], 1),
            flatten_small=round(pipe["n_flatten_small"], 1),
            flatten_thin=round(pipe["n_flatten_thin"], 1),
            flatten_dark=round(pipe["n_flatten_dark"], 1),
            nonconvex=round(pipe["n_nonconvex_after_coarsen"], 1),
            blobs_split=round(pipe["n_blobs_split"], 1),
            dark_removed=round(pipe["n_dark_removed"], 1),
            pipeline_final=round(pipe["n_final_grains"], 1),
            excl_border_per_img=round(n_border / max(1, len(per_image_counts)), 1),
            excl_nonconvex_per_img=round(
                n_nonconvex_resid / max(1, len(per_image_counts)), 1),
            analysis_final_per_img=round(float(np.mean(per_image_counts)), 1),
        ))

        with open(os.path.join(ANALYSIS_FOLDER, f"grains_{alloy}.csv"),
                  "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            for r in rows:
                wcsv.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in r.items()})

        print(f"{alloy}: {n} grains (border -{n_border}, "
              f"non-convex -{n_nonconvex_resid}); median d_eq "
              f"{np.median(d['deq_um']):.2f} um, sigma_log {sigma:.3f}, "
              f"KS {ks:.3f}", flush=True)

    # ── summary files ──
    def write_csv(path, rows_):
        with open(path, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows_)

    write_csv(os.path.join(ANALYSIS_FOLDER, "summary.csv"), summary_rows)
    write_csv(os.path.join(ANALYSIS_FOLDER, "attrition.csv"), attrition_rows)

    # ── LaTeX tables ──
    def grade_label(a):
        return a.replace("Ultra_Co", "Ultra").replace("6_2", "6\\_2")

    with open(os.path.join(ANALYSIS_FOLDER, "latex_tables.txt"), "w") as f:
        f.write("%% ---- tab:grain-counts ----\n")
        for s in sorted(summary_rows, key=lambda r: r["grains_per_image_mean"]):
            f.write(f"{grade_label(s['alloy'])} & "
                    f"{s['grains_per_image_mean']:.1f} & "
                    f"{s['grains_per_image_std']:.1f} \\\\\n")
        f.write("\n%% ---- tab:deq-lognormal ----\n")
        f.write("%% Grade & mu & sigma & Median d_eq (um) & N_grains\n")
        for s in summary_rows:
            f.write(f"{grade_label(s['alloy'])} & "
                    f"{s['lognorm_mu_px']:.3f} & {s['lognorm_sigma']:.3f} & "
                    f"{s['lognorm_median_um']:.2f} & "
                    f"{s['grains_total']} \\\\\n")
        f.write("\n%% ---- tab:feret-stats (px) ----\n")
        for s in summary_rows:
            f.write(f"{grade_label(s['alloy'])} & "
                    f"{s['fmax_mean_px']:.1f} & {s['fmin_mean_px']:.1f} \\\\\n")
        f.write("\n%% ---- numbers for the text ----\n")
        ks_vals = [s["ks_stat"] for s in summary_rows]
        n_vals = [s["grains_total"] for s in summary_rows]
        f.write(f"%% KS in [{min(ks_vals):.3f}, {max(ks_vals):.3f}]\n")
        f.write(f"%% N_grains in [{min(n_vals)}, {max(n_vals)}]\n")
        f.write(f"%% sigma in [{min(s['lognorm_sigma'] for s in summary_rows):.3f}, "
                f"{max(s['lognorm_sigma'] for s in summary_rows):.3f}]\n")
        smax = max(summary_rows, key=lambda s: s["fmax_mean_px"])
        smin = min(summary_rows, key=lambda s: s["fmax_mean_px"])
        f.write(f"%% F_max ratio coarse/fine = "
                f"{smax['fmax_mean_px'] / smin['fmax_mean_px']:.2f}\n")
        f.write(f"%% F_min ratio coarse/fine = "
                f"{smax['fmin_mean_px'] / smin['fmin_mean_px']:.2f}\n")
        f.write(f"%% psi_A mode by grade: "
                + ", ".join(f"{grade_label(s['alloy'])}={s['psi_a_mode']:.2f}"
                            for s in summary_rows) + "\n")
        f.write(f"%% sphericity mode by grade: "
                + ", ".join(f"{grade_label(s['alloy'])}={s['sphericity_mode']:.2f}"
                            for s in summary_rows) + "\n")

    # ── figures: a row of 5 panels per descriptor ──
    def panel_figure(key, xlabel, fname, lognorm_overlay=False):
        alloys_present = [a for a in ALLOYS if a in all_descr]
        fig, axes = plt.subplots(1, len(alloys_present),
                                 figsize=(4.2 * len(alloys_present), 3.6))
        axes = np.atleast_1d(axes)
        for ax, a in zip(axes, alloys_present):
            x = all_descr[a][key]
            ax.hist(x, bins=80, density=True, alpha=0.6)
            if lognorm_overlay:
                mu, sg = lognorm_fit(x)
                xs = np.linspace(x.min(), x.max(), 400)
                ax.plot(xs, lognorm_pdf(xs, mu, sg), "k-", lw=1.6)
            ax.set_title(a.replace("Ultra_Co", "Ultra"))
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel("density")
        fig.tight_layout()
        fig.savefig(os.path.join(ANALYSIS_FOLDER, fname), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    panel_figure("deq_um", r"$d_{eq}$, $\mu$m", "deq_lognormal.png",
                 lognorm_overlay=True)
    panel_figure("area_um2", r"area, $\mu$m$^2$", "area_all.png")
    panel_figure("perimeter_um", r"perimeter, $\mu$m", "perimeter_all.png")
    panel_figure("fmin_um", r"$F_{min}$, $\mu$m", "min_feret_all.png")
    panel_figure("fmax_um", r"$F_{max}$, $\mu$m", "max_feret_all.png")
    panel_figure("sphericity", r"sphericity $S$", "sphericity_all.png")
    panel_figure("psi_a", r"$\psi_A$", "aspect_ratio_all.png")

    print(f"\nEverything saved to {ANALYSIS_FOLDER}/", flush=True)
    print("Done!", flush=True)
